"""
[ETL 4] DUR 성분정보 Bronze → Silver  (v2: 2026-10-07 probe 검증 필드 반영)
입력: s3://bucket/bronze/api/dur_ingredient/{type}/
출력:
  s3://bucket/silver/dur_ingredient/{type}/      성분 단위 DUR 문서 (Graph 금기 엣지 + RAG)
  s3://bucket/silver/dur_ingredient/code_map/    원료코드(M/I) → DUR 성분코드(D) 공식 매핑
                                                 (허가정보 M코드를 D코드로 바꿔 CONTAINS 연결)

확인된 원본 필드 (유형마다 이름이 다름)
  병용금기 : INGR_CODE, INGR_KOR_NAME, INGR_ENG_NAME, MIX, ORI, CLASS,
             MIXTURE_INGR_CODE, MIXTURE_INGR_KOR_NAME, MIXTURE_INGR_ENG_NAME,
             MIXTURE_MIX, MIXTURE_ORI, MIXTURE_CLASS, (DUR_SEQ·FORM_NAME 없음)
  그 외    : DUR_SEQ, INGR_CODE, INGR_NAME, INGR_ENG_NAME, MIX_INGR, ORI_INGR, CLASS_NAME, FORM_NAME
  공통     : TYPE_NAME, MIX_TYPE, NOTIFICATION_DATE, PROHBT_CONTENT, REMARK, DEL_YN('정상')
  유형별   : AGE_BASE('18세 이하'), GRADE('1등급'), MAX_QTY('0.25밀리그램'),
             MAX_DOSAGE_TERM('21일'), EFFECT_CODE('해열진통소염제'), SERS_NAME('비스테로이드성 소염제')
  ORI/ORI_INGR = '[M083733]이트라코나졸제피과립/[M083734]이트라코나졸/...' → 해당 D코드에 속하는 원료코드 목록
"""
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).resolve().parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import list_keys, download_json, upload_json, silver_key

SOURCE_META = {
    "source_name": "식품의약품안전처 DUR(의약품 안전사용 서비스) 성분정보",
    "source_url": "https://www.data.go.kr/data/15056780/openapi.do",
    "api_endpoint": "https://apis.data.go.kr/1471000/DURIrdntInfoService03",
    "official_url": "https://www.dur.or.kr",
    "license": "공공누리 1유형",
    "provider": "식품의약품안전처",
}

# level: 그래프 관계의 강도 (ban = 금기, caution = 주의)
TYPES = {
    "combination_ban":       {"desc": "병용금기",       "pair": True,  "level": "ban"},
    "pregnancy_ban":         {"desc": "임부금기",       "pair": False, "level": "ban"},
    "age_specific_ban":      {"desc": "특정연령대금기", "pair": False, "level": "ban"},
    "elderly_caution":       {"desc": "노인주의",       "pair": False, "level": "caution"},
    "dose_caution":          {"desc": "용량주의",       "pair": False, "level": "caution"},
    "dosing_period_caution": {"desc": "투여기간주의",   "pair": False, "level": "caution"},
    "effect_duplicate":      {"desc": "효능군중복",     "pair": False, "level": "caution"},  # 성분 → 효능군 소속
}

# 논리 필드 → 원본 키 (유형마다 다른 이름을 모두 나열, 앞에서부터 값이 있는 키 사용)
F = {
    "dur_seq":      ["DUR_SEQ"],
    "type_name":    ["TYPE_NAME"],
    "mix_type":     ["MIX_TYPE"],
    "ingr_code":    ["INGR_CODE"],
    "ingr_kor":     ["INGR_KOR_NAME", "INGR_NAME"],
    "ingr_eng":     ["INGR_ENG_NAME"],
    "mix_ingr":     ["MIX_INGR", "MIX"],
    "ori":          ["ORI_INGR", "ORI"],
    "class":        ["CLASS_NAME", "CLASS"],
    "form_name":    ["FORM_NAME"],
    "notification": ["NOTIFICATION_DATE"],
    "prohbt":       ["PROHBT_CONTENT"],
    "remark":       ["REMARK"],
    "del_yn":       ["DEL_YN"],
    # 병용금기 상대 성분
    "ingr_code_b":  ["MIXTURE_INGR_CODE"],
    "ingr_kor_b":   ["MIXTURE_INGR_KOR_NAME", "MIXTURE_INGR_NAME"],
    "ingr_eng_b":   ["MIXTURE_INGR_ENG_NAME"],
    "mix_type_b":   ["MIXTURE_MIX_TYPE"],
    "mix_ingr_b":   ["MIXTURE_MIX_INGR", "MIXTURE_MIX"],
    "ori_b":        ["MIXTURE_ORI_INGR", "MIXTURE_ORI"],
    "class_b":      ["MIXTURE_CLASS_NAME", "MIXTURE_CLASS"],
    # 유형별
    "age_base":     ["AGE_BASE"],
    "grade":        ["GRADE"],
    "max_qty":      ["MAX_QTY"],
    "max_term":     ["MAX_DOSAGE_TERM"],
    "effect_name":  ["EFFECT_CODE", "EFFECT_NAME"],     # EFFECT_CODE의 값이 효능군 이름
    "sers_name":    ["SERS_NAME"],
}

# 유형별로 반드시 있어야 하는 필드 (결측 리포트 대상)
REQUIRED = {
    "combination_ban":       ["ingr_code", "ingr_code_b", "prohbt"],
    "pregnancy_ban":         ["ingr_code", "grade", "prohbt"],
    "age_specific_ban":      ["ingr_code", "age_base"],
    "elderly_caution":       ["ingr_code", "prohbt"],
    "dose_caution":          ["ingr_code", "max_qty"],
    "dosing_period_caution": ["ingr_code", "max_term"],
    "effect_duplicate":      ["ingr_code", "effect_name"],
}

KNOWN_KEYS = {k for ks in F.values() for k in ks}
NULLS = {"none", "null", "nan"}
ACTIVE_DEL_VALUES = {"", "정상", "N"}          # DEL_YN 값이 이 외(예: '삭제', 'Y')이면 제외

_miss = defaultdict(Counter)
_seen_keys = defaultdict(set)
_type_total = Counter()
_skip = defaultdict(Counter)        # _skip[type][사유]
_exact_dup = Counter()              # 내용까지 같은 완전 중복 (제거)
_collision = Counter()
_map_skipped = Counter()            # 복합제 ORI 중 다른 원료라서 매핑에서 제외한 수              # doc_id만 같고 내용이 다른 레코드 (보존)


# ─────────────────────────────────────
# 유틸
# ─────────────────────────────────────
def clean_text(text) -> str:
    if text is None:
        return ""
    s = " ".join(str(text).split()).strip()
    return "" if s.lower() in NULLS else s


def get(item: dict, field: str) -> str:
    for k in F[field]:
        v = clean_text(item.get(k))
        if v:
            return v
    return ""


def clean_reason(r: str) -> str:
    r = (r or "").replace("？", "·")
    r = re.sub(r"^[\-\s_]+", "", r)                      # '_'만 있던 값(원본 결측) → ""
    r = re.sub(r"안전성\s*(및)?\s*유효성", "안전성 및 유효성", r)
    r = re.sub(r"\s*-\s+", " / ", r)
    return clean_text(r)


AGE_RE = re.compile(r"(\d+)\s*(세|개월)\s*(미만|이하)")


def parse_max_age_months(*texts: str):
    """'18세 이하' → 217, '3세 미만' → 36 ('미만' 기준 개월 수로 통일). 실패 시 None"""
    for t in texts:
        m = AGE_RE.search(t or "")
        if m:
            n, unit, op = int(m.group(1)), m.group(2), m.group(3)
            months = n * 12 if unit == "세" else n
            return months + (1 if op == "이하" else 0)
    return None


CODE_RE = re.compile(r"\[([A-Z]\d+)\]\s*([^/\[]*)")


def parse_codes(raw: str) -> list[tuple[str, str]]:
    """'[M083733]이트라코나졸제피과립/[I005603]트리아졸람' → [('M083733','이트라코나졸제피과립'), ('I005603','트리아졸람')]"""
    seen, out = set(), []
    for m in CODE_RE.finditer(raw or ""):
        code, name = m.group(1), clean_text(m.group(2))
        if code not in seen:
            seen.add(code)
            out.append((code, name))
    return out


def split_class(raw: str) -> tuple[str, str]:
    """'[06290]기타의 화학요법제' → ('06290', '기타의 화학요법제')"""
    m = re.match(r"\[(\w+)\]\s*(.*)", raw or "")
    return (m.group(1), clean_text(m.group(2))) if m else ("", clean_text(raw))


def grade_num(raw: str) -> int | None:
    m = re.search(r"(\d+)", raw or "")
    return int(m.group(1)) if m else None


# ─────────────────────────────────────
# 변환
# ─────────────────────────────────────
def build_text(m: dict) -> str:
    a = m["ingredient_a_kor"] or m["ingredient_a_eng"]
    b = m["ingredient_b_kor"] or m["ingredient_b_eng"]
    text = f"[{m['dur_desc']}] 성분: {a} + {b}" if b else f"[{m['dur_desc']}] 성분: {a}"
    if m["form_name"]:
        text += f" | 제형: {m['form_name']}"
    if m["pregnancy_grade"]:
        text += f" | 등급: {m['pregnancy_grade']}등급"
    if m["age_base"]:
        text += f" | 연령기준: {m['age_base']}"
    if m["effect_name"]:
        text += f" | 효능군: {m['effect_name']}"
        if m["sers_name"]:
            text += f" ({m['sers_name']})"
    if m["max_qty"]:
        text += f" | 1일 최대투여량: {m['max_qty']}"
    if m["max_term"]:
        text += f" | 최대투여기간: {m['max_term']}"
    if m["reason"]:
        text += f" | 사유: {m['reason']}"
    if m["remark"]:
        text += f" | 비고: {m['remark']}"
    return text


def transform(item: dict, t: str, etl_at: str) -> dict | None:
    pair = TYPES[t]["pair"]

    for field in REQUIRED[t]:
        if not get(item, field):
            _miss[t][field] += 1

    if get(item, "del_yn") not in ACTIVE_DEL_VALUES:
        _skip[t][f"삭제표시({get(item, 'del_yn')})"] += 1
        return None

    code_a = get(item, "ingr_code")
    kor_a, eng_a = get(item, "ingr_kor"), get(item, "ingr_eng")
    if not (code_a or kor_a or eng_a):
        _skip[t]["성분 정보 없음"] += 1
        return None

    code_b = get(item, "ingr_code_b") if pair else ""
    kor_b = get(item, "ingr_kor_b") if pair else ""
    eng_b = get(item, "ingr_eng_b") if pair else ""
    if pair and not (code_b or kor_b or eng_b):
        _skip[t]["상대 성분 정보 없음"] += 1
        return None

    prohbt = clean_reason(get(item, "prohbt"))
    remark = clean_reason(get(item, "remark"))
    class_code_a, class_name_a = split_class(get(item, "class"))
    class_code_b, class_name_b = split_class(get(item, "class_b")) if pair else ("", "")
    age_base = get(item, "age_base") if t == "age_specific_ban" else ""
    key_a = code_a or eng_a or kor_a
    key_b = code_b or eng_b or kor_b

    metadata = {
        "dur_type": t,
        "dur_desc": TYPES[t]["desc"],
        "level": TYPES[t]["level"],
        "dur_seq": get(item, "dur_seq"),
        # 성분 A (품목정보 ETL과 같은 키 이름)
        "ingr_code_a": code_a,
        "ingredient_a_kor": kor_a,
        "ingredient_a_eng": eng_a,
        "mix_type_a": get(item, "mix_type"),
        "mix_ingr_a": get(item, "mix_ingr"),
        "ori_codes_a": [c for c, _ in parse_codes(get(item, "ori"))],
        "class_code_a": class_code_a,
        "class_name_a": class_name_a,
        # 성분 B (병용금기)
        "ingr_code_b": code_b,
        "ingredient_b_kor": kor_b,
        "ingredient_b_eng": eng_b,
        "mix_type_b": get(item, "mix_type_b") if pair else "",
        "mix_ingr_b": get(item, "mix_ingr_b") if pair else "",
        "ori_codes_b": [c for c, _ in parse_codes(get(item, "ori_b"))] if pair else [],
        "class_code_b": class_code_b,
        "class_name_b": class_name_b,
        # 무방향 관계 키 (그래프에서 A-B / B-A 중복 방지)
        "pair_key": "|".join(sorted([key_a, key_b])) if pair else "",
        # 공통
        "form_name": get(item, "form_name"),
        "reason": prohbt or remark,
        "remark": remark if prohbt else "",            # 예: 임부금기 '경구' = 투여 경로 제한
        "notification_date": get(item, "notification"),
        # 유형별
        "pregnancy_grade": grade_num(get(item, "grade")) if t == "pregnancy_ban" else None,
        "age_base": age_base,
        "max_age_months": parse_max_age_months(age_base, prohbt, remark) if t == "age_specific_ban" else None,
        "max_qty": get(item, "max_qty") if t == "dose_caution" else "",
        "max_term": get(item, "max_term") if t == "dosing_period_caution" else "",
        "effect_name": get(item, "effect_name") if t == "effect_duplicate" else "",
        "sers_name": get(item, "sers_name") if t == "effect_duplicate" else "",
        # 출처
        **SOURCE_META,
        "etl_at": etl_at,
    }

    id_parts = [t, metadata["dur_seq"], key_a, key_b if pair else "", metadata["form_name"]]
    doc_id = "dur_ingr_" + "_".join(re.sub(r"\s+", "", p) for p in id_parts if p)

    return {
        "doc_id": doc_id,
        "doc_type": "dur_ingredient",
        "text": build_text(metadata),
        "metadata": metadata,
    }


CONTENT_FIELDS = ["ingr_code_a", "ingr_code_b", "mix_ingr_a", "mix_ingr_b", "form_name", "reason", "remark",
                  "pregnancy_grade", "age_base", "max_qty", "max_term", "effect_name", "sers_name",
                  "notification_date"]


def content_sig(doc: dict) -> str:
    """doc_id가 같을 때 실제 내용이 같은지 비교하기 위한 서명"""
    m = doc["metadata"]
    raw = "\x1f".join(str(m.get(f, "")) for f in CONTENT_FIELDS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────
# 원료코드(M/I) → DUR 성분코드(D) 매핑
# ─────────────────────────────────────
NAME_NORM_RE = re.compile(r"[\s\(\)\[\]·,\-]")


def _norm_name(x: str) -> str:
    return NAME_NORM_RE.sub("", (x or "")).lower()


def _same_ingredient(src_name: str, kor: str, eng: str) -> bool:
    """복합제 레코드에서 원료명이 해당 성분명과 같은 성분인지 (이름 포함 관계로 판단)
    예) D '아세트아미노펜' ↔ M '아세트아미노펜' ✅ / M '감초' ❌
        D '세티리진염산염' ↔ M '세티리진염산염' ✅ / D '이트라코나졸' ↔ M '이트라코나졸제피과립' ✅
    """
    s = _norm_name(src_name)
    if not s:
        return False
    for target in (_norm_name(kor), _norm_name(eng)):
        if len(target) >= 2 and (target in s or s in target):
            return True
    return False


def collect_code_map(item: dict, t: str, cmap: dict):
    """원료코드(ORI) → DUR 성분코드(D) 매핑 수집

    ⚠️ 복합제 레코드의 ORI에는 해당 성분뿐 아니라 복합제에 함께 든 '모든 원료'가 나열됨
       (예: 아세트아미노펜 복합 레코드의 ORI = 아세트아미노펜, 감초, 인삼, 아스코르빈산 …)
       → 단일제 레코드: ORI 전체를 매핑
       → 복합제 레코드: ORI 중 이름이 성분명과 일치하는 원료만 매핑
    """
    sides = [("ingr_code", "ingr_kor", "ingr_eng", "ori", "mix_type")]
    if TYPES[t]["pair"]:
        sides.append(("ingr_code_b", "ingr_kor_b", "ingr_eng_b", "ori_b", "mix_type_b"))
    for code_f, kor_f, eng_f, ori_f, mix_f in sides:
        d = get(item, code_f)
        if not d:
            continue
        kor, eng = get(item, kor_f), get(item, eng_f)
        is_single = get(item, mix_f) == "단일"
        for src_code, src_name in parse_codes(get(item, ori_f)):
            if is_single:
                basis = "single"
            elif _same_ingredient(src_name, kor, eng):
                basis = "mixture_name_match"
            else:
                _map_skipped[t] += 1               # 복합제의 다른 원료 (감초, 인삼 등) → 매핑하지 않음
                continue
            entry = cmap.setdefault((src_code, d), {
                "src_code": src_code,
                "src_code_type": src_code[0],      # M = 원료, I = 기타, A/C = 혼합·기타 원료
                "src_name": src_name,
                "dur_code": d,
                "dur_name_kor": kor,
                "dur_name_eng": eng,
                "seen_in": set(),
                "basis": set(),
            })
            entry["seen_in"].add(t)
            entry["basis"].add(basis)


def finalize_code_map(cmap: dict, etl_at: str) -> list[dict]:
    d_per_src = defaultdict(set)
    for (src, d) in cmap:
        d_per_src[src].add(d)
    rows = []
    for (src, d), e in sorted(cmap.items()):
        rows.append({
            **{k: v for k, v in e.items() if k not in ("seen_in", "basis")},
            "seen_in": sorted(e["seen_in"]),
            "basis": sorted(e["basis"]),                 # single / mixture_name_match
            "ambiguous": len(d_per_src[src]) > 1,      # 원료코드 하나가 여러 D코드에 속하는 경우
            "source_url": SOURCE_META["source_url"],
            "etl_at": etl_at,
        })
    return rows


# ─────────────────────────────────────
# 리포트
# ─────────────────────────────────────
def report(type_items: dict, code_rows: list[dict]):
    logger.info("─" * 60)
    logger.info("필드 리포트")
    for t, total in _type_total.items():
        for field, cnt in _miss[t].most_common():
            rate = cnt / total * 100 if total else 0
            if rate >= 10:
                logger.warning(f"  [{t}] 필수 필드 '{field}' 결측 {rate:5.1f}%  (키: {F[field]})")
        if _skip[t]:
            logger.info(f"  [{t}] 스킵 사유: {dict(_skip[t])}")
        if _exact_dup[t] or _collision[t]:
            logger.info(f"  [{t}] 완전 중복 제거 {_exact_dup[t]}건 / id 같고 내용 달라 보존 {_collision[t]}건")
        unmapped = sorted(_seen_keys[t] - KNOWN_KEYS)
        if unmapped:
            logger.info(f"  [{t}] 매핑되지 않은 원본 키: {unmapped}")

    logger.info("─" * 60)
    for t, items in type_items.items():
        if not items:
            continue
        n = len(items)
        msg = f"  [{TYPES[t]['desc']}] {n}건"
        if TYPES[t]["pair"]:
            msg += f" | 고유 성분쌍 {len({d['metadata']['pair_key'] for d in items})}개"
        if t == "age_specific_ban":
            n_age = sum(d["metadata"]["max_age_months"] is not None for d in items)
            bases = Counter(d["metadata"]["age_base"] for d in items).most_common(5)
            msg += f" | 연령 파싱 {n_age}/{n} | 상위 기준 {bases}"
        if t == "pregnancy_ban":
            msg += f" | 등급 {dict(Counter(d['metadata']['pregnancy_grade'] for d in items))}"
        if t == "effect_duplicate":
            msg += f" | 효능군 {len({d['metadata']['effect_name'] for d in items})}개"
        logger.info(msg)

    n_amb = sum(r["ambiguous"] for r in code_rows)
    logger.info(f"  [코드 매핑] 원료코드→D코드 {len(code_rows)}건 "
                f"(원료코드 {len({r['src_code'] for r in code_rows})}종 → D코드 {len({r['dur_code'] for r in code_rows})}종)"
                f" | 다중 매핑 {n_amb}건")
    logger.info(f"  [코드 매핑] 복합제의 다른 원료라서 제외: {sum(_map_skipped.values()):,}건")
    amb = defaultdict(list)
    for r in code_rows:
        if r["ambiguous"]:
            amb[(r["src_code"], r["src_name"])].append(f"{r['dur_code']}({r['dur_name_kor']})")
    by_type = Counter(code[0] for code, _ in amb)
    logger.info(f"  [코드 매핑] 다중 매핑 원료코드 {len(amb)}종 (코드 종류별 {dict(by_type)}) — 예시:")
    for (code, name), ds in list(amb.items())[:15]:
        logger.info(f"     {code}({name}) → {' / '.join(ds)}")
    logger.info("─" * 60)


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────
def save(rows: list, prefix: str, extra: dict | None = None, batch_size: int = 500) -> int:
    for j in range(0, len(rows), batch_size):
        batch = rows[j:j + batch_size]
        upload_json({**(extra or {}), "count": len(batch), "items": batch},
                    silver_key(prefix, f"batch_{j // batch_size:04d}.json"))
    return len(rows)


def main():
    logger.info("=" * 50)
    logger.info("[ETL 4] DUR 성분정보 정제 시작 (v2)")
    logger.info("=" * 50)
    etl_at = datetime.now().isoformat()

    keys = [k for k in list_keys("bronze/api/dur_ingredient/") if k.endswith(".json")]
    logger.info(f"Bronze 파일: {len(keys)}개")

    type_items = {t: [] for t in TYPES}
    seen_ids = {t: set() for t in TYPES}
    seen_sigs = {t: set() for t in TYPES}
    cmap: dict = {}
    total_raw = dup = skipped = 0

    for key in keys:
        try:
            data = download_json(key)
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")
            continue
        if not data:
            continue
        t = data.get("type", "")
        if t not in TYPES:
            continue

        for item in data.get("items", []):
            # 수집 단계에서 벗기지 못한 {'item': {...}} 래퍼 방어
            if isinstance(item, dict) and set(item) == {"item"} and isinstance(item["item"], dict):
                item = item["item"]
            total_raw += 1
            _type_total[t] += 1
            _seen_keys[t].update(item.keys())

            doc = transform(item, t, etl_at)
            if not doc:
                skipped += 1
                continue
            collect_code_map(item, t, cmap)
            sig = content_sig(doc)
            if sig in seen_sigs[t]:                     # 내용까지 같음 → 완전 중복 제거
                _exact_dup[t] += 1
                dup += 1
                continue
            seen_sigs[t].add(sig)
            if doc["doc_id"] in seen_ids[t]:            # id만 같고 내용이 다름 → 해시를 붙여 보존
                doc["doc_id"] = f"{doc['doc_id']}_{sig[:8]}"
                _collision[t] += 1
            seen_ids[t].add(doc["doc_id"])
            type_items[t].append(doc)

    code_rows = finalize_code_map(cmap, etl_at)
    logger.info(f"원본 {total_raw}건 / 스킵 {skipped}건 / 완전 중복 제거 {dup}건 / 내용이 달라 보존 {sum(_collision.values())}건")
    report(type_items, code_rows)

    total_saved = 0
    for t, items in type_items.items():
        if items:
            total_saved += save(items, f"dur_ingredient/{t}",
                                {"dur_type": t, "description": TYPES[t]["desc"]})
            logger.info(f"  [{TYPES[t]['desc']}] ✅ {len(items)}건 저장")
    save(code_rows, "dur_ingredient/code_map", {"description": "원료코드→DUR성분코드 매핑"}, batch_size=2000)
    logger.info(f"  [코드 매핑] ✅ {len(code_rows)}건 저장")

    logger.info(f"✅ DUR 성분정보 Silver 저장 완료: {total_saved}건 + 코드 매핑 {len(code_rows)}건")
    return total_saved


if __name__ == "__main__":
    main()
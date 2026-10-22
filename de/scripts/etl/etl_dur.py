"""
[ETL 2] DUR API Bronze → Silver  (v2: 성분코드 보존 / 필드명 차이 흡수 / doc_id 고유화)

변경 사항 (v1 대비)
  1. 유형별로 다른 필드명 흡수: 성분 한글명 INGR_KOR_NAME(병용금기) vs INGR_NAME(연령금기 등)
  2. 모든 유형에서 PROHBT_CONTENT를 읽음 (v1은 노인/용량/투여기간에서 REMARK만 읽어 reason 누락)
  3. REMARK를 reason과 분리 보존 (예: 연령금기 '점안제(1%)' = 적용 범위)
  4. 그래프용 코드 필드 보존: INGR_CODE, MAIN_INGR, MIX_TYPE, MIX_INGR, CLASS_CODE (+ 상대 성분/제품)
  5. doc_id = 유형 + 제품A + 성분코드A + 제품B + 성분코드B → 고유 (v1은 상대 제품이 달라도 중복)
  6. 중복 제거 키도 doc_id와 동일 기준 (이름 → 코드)
  7. reason 정규화: '_'(원본 결측) 제거, '？' 깨짐 보정, 표기 통일
  8. 연령금기: PROHBT_CONTENT에서 'N세/개월 미만' 파싱 → max_age_months (파싱 실패 시 null)
  9. 필드 매핑 리포트: 유형별 결측률 + 매핑되지 않은 원본 키 출력 (필드명 검증용)
"""
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import list_keys, dated_keys, download_json, upload_json, silver_key

SOURCE_META = {
    "source_name": "식품의약품안전처 DUR(의약품 안전사용 서비스)",
    "source_url": "https://www.data.go.kr/data/15059486/openapi.do",  # DUR 품목정보 (v1의 15075057은 e약은요 URL이었음)
    "api_endpoint": "https://apis.data.go.kr/1471000/DURPrdlstInfoService03",
    "official_url": "https://www.dur.or.kr",
    "license": "공공누리 1유형",
    "provider": "식품의약품안전처",
}

# 논리 필드 → 원본 키 후보 (앞에서부터 처음 값이 있는 키 사용)
F = {
    # 성분 A
    "ingr_code_a":   ["INGR_CODE"],
    "ingr_kor_a":    ["INGR_KOR_NAME", "INGR_NAME"],
    "ingr_eng_a":    ["INGR_ENG_NAME"],
    "main_ingr_a":   ["MAIN_INGR"],
    "mix_type_a":    ["MIX_TYPE", "MIX"],            # 병용금기·효능군중복은 MIX에 단일/복합
    "mix_ingr_a":    ["MIX_INGR"],
    # 성분 B (병용금기 / 효능군중복)
    "ingr_code_b":   ["MIXTURE_INGR_CODE"],
    "ingr_kor_b":    ["MIXTURE_INGR_KOR_NAME", "MIXTURE_INGR_NAME"],
    "ingr_eng_b":    ["MIXTURE_INGR_ENG_NAME"],
    "main_ingr_b":   ["MIXTURE_MAIN_INGR"],
    "mix_type_b":    ["MIXTURE_MIX_TYPE", "MIXTURE_MIX"],
    "mix_ingr_b":    ["MIXTURE_MIX_INGR"],
    # 제품 A
    "item_seq":      ["ITEM_SEQ"],
    "item_name":     ["ITEM_NAME"],
    "manufacturer":  ["ENTP_NAME"],
    "class_code":    ["CLASS_CODE"],
    "class_name":    ["CLASS_NAME"],
    "etc_otc":       ["ETC_OTC_NAME"],
    "form_name":     ["FORM_NAME"],
    # 제품 B
    "mixture_item_seq":     ["MIXTURE_ITEM_SEQ"],
    "mixture_item_name":    ["MIXTURE_ITEM_NAME"],
    "mixture_manufacturer": ["MIXTURE_ENTP_NAME"],
    "mixture_class_code":   ["MIXTURE_CLASS_CODE"],
    "mixture_class_name":   ["MIXTURE_CLASS_NAME"],
    "mixture_etc_otc":      ["MIXTURE_ETC_OTC_NAME"],
    "mixture_form_name":    ["MIXTURE_FORM_NAME"],
    # 금기 / 주의 내용
    "prohbt":        ["PROHBT_CONTENT"],
    "remark":        ["REMARK"],
    "effect_name":   ["EFFECT_NAME"],
    "sers_name":     ["SERS_NAME"],
    "age_base":      ["AGE_BASE"],          # 원본에 없으면 PROHBT_CONTENT 파싱으로 대체
    # 일자 / 기타
    "dur_seq":           ["DUR_SEQ"],
    "notification_date": ["NOTIFICATION_DATE"],
    "change_date":       ["CHANGE_DATE"],
}

TYPES = {
    "combination_ban":       {"desc": "병용금기",       "pair": True},
    "effect_duplicate":      {"desc": "효능군중복",     "pair": False},  # 품목정보에서도 단일 성분 (상대 필드 없음)
    "pregnancy_ban":         {"desc": "임부금기",       "pair": False},
    "age_specific_ban":      {"desc": "특정연령대금기", "pair": False},
    "elderly_caution":       {"desc": "노인주의",       "pair": False},
    "dose_caution":          {"desc": "용량주의",       "pair": False},
    "dosing_period_caution": {"desc": "투여기간주의",   "pair": False},
}

# 매핑 리포트에서 '알려진 키'로 취급할 원본 키 (매핑 대상은 아니지만 의도적으로 무시)
IGNORED_KEYS = {"TYPE_NAME", "TYPE_CODE", "CHART", "ITEM_PERMIT_DATE", "INGR_ENG_NAME_FULL", "BIZRNO",
                "ETC_OTC_CODE", "FORM_CODE", "FORM_CODE_NAME",
                "MIXTURE_CHART", "MIXTURE_ITEM_PERMIT_DATE", "MIXTURE_INGR_ENG_NAME_FULL",
                "MIXTURE_ETC_OTC_CODE", "MIXTURE_FORM_CODE", "MIXTURE_DUR_SEQ", "MIXTURE_CHANGE_DATE"}
KNOWN_KEYS = {k for keys in F.values() for k in keys} | IGNORED_KEYS

_miss = defaultdict(Counter)    # _miss[type][field]
_seen_keys = defaultdict(set)   # 유형별 원본 키 전체
_type_total = Counter()
_exact_dup = Counter()      # 내용까지 같은 완전 중복 (제거)
_collision = Counter()      # doc_id만 같고 내용이 다른 레코드 (해시를 붙여 보존)

# ⚠️ 품목정보는 '제품 쌍' 단위 문서이므로 제품 코드가 반드시 포함되어야 함
CONTENT_FIELDS = ["item_seq", "mixture_item_seq", "ingr_code_a", "ingr_code_b", "mix_type_a", "mix_type_b", "mix_ingr_a", "mix_ingr_b",
                  "form_name", "mixture_form_name", "reason", "remark", "effect_name", "sers_name",
                  "dur_seq", "notification_date"]


def content_sig(doc: dict) -> str:
    m = doc["metadata"]
    raw = "\x1f".join(str(m.get(f, "")) for f in CONTENT_FIELDS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────
# 유틸
# ─────────────────────────────────────
def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def get(item: dict, field: str, type_key: str = "") -> str:
    for k in F[field]:
        v = item.get(k)
        if v not in (None, ""):
            return clean_text(v)
    if type_key:
        _miss[type_key][field] += 1
    return ""


def clean_reason(r: str) -> str:
    r = (r or "").replace("？", "·")
    r = re.sub(r"^[\-\s_]+", "", r)                      # 앞쪽 '-', '_' 제거 ('_'만 있던 값은 결측 처리)
    r = re.sub(r"안전성\s*(및)?\s*유효성", "안전성 및 유효성", r)
    r = re.sub(r"\s*-\s+", " / ", r)                     # '...미확립- 퀴놀론계...' 구분자 통일
    return clean_text(r)


AGE_RE = re.compile(r"(\d+)\s*(세|개월)\s*(미만|이하)")


def parse_max_age_months(*texts: str):
    """'3세 미만' → 36, '12세 이하' → 145 (미만 기준으로 통일). 실패 시 None"""
    for t in texts:
        m = AGE_RE.search(t or "")
        if m:
            n, unit, op = int(m.group(1)), m.group(2), m.group(3)
            months = n * 12 if unit == "세" else n
            return months + (1 if op == "이하" else 0)
    return None


def make_id_parts(item: dict, type_key: str) -> list[str]:
    """doc_id / 중복제거 공통 키: 코드 우선, 없으면 이름으로 대체"""
    parts = [
        get(item, "item_seq"),
        get(item, "ingr_code_a") or get(item, "ingr_eng_a") or get(item, "ingr_kor_a"),
    ]
    if TYPES[type_key]["pair"]:
        parts += [
            get(item, "mixture_item_seq"),
            get(item, "ingr_code_b") or get(item, "ingr_eng_b") or get(item, "ingr_kor_b"),
        ]
    return parts


# ─────────────────────────────────────
# 변환
# ─────────────────────────────────────
def build_text(m: dict) -> str:
    desc = m["dur_desc"]
    a = m["ingredient_a_kor"] or m["ingredient_a_eng"]
    b = m["ingredient_b_kor"] or m["ingredient_b_eng"]
    if b:
        text = f"[{desc}] 성분: {a} + {b}"
        if m["item_name"] and m["mixture_item_name"]:
            text += f" | 제품: {m['item_name']} + {m['mixture_item_name']}"
    else:
        text = f"[{desc}] 성분: {a}"
        if m["item_name"]:
            text += f" | 제품: {m['item_name']}"
    if m["effect_name"]:
        text += f" | 효능군: {m['effect_name']}"
    if m["reason"]:
        text += f" | 사유: {m['reason']}"
    if m["remark"]:
        text += f" | 비고: {m['remark']}"
    return text


def transform_dur_item(item: dict, type_key: str) -> dict | None:
    t = type_key
    ingr_a_kor, ingr_a_eng = get(item, "ingr_kor_a", t), get(item, "ingr_eng_a", t)
    if not (ingr_a_kor or ingr_a_eng):
        return None

    prohbt = clean_reason(get(item, "prohbt", t))
    remark = clean_reason(get(item, "remark"))
    effect_name = get(item, "effect_name") if t == "effect_duplicate" else ""
    reason = prohbt or (remark if not TYPES[t]["pair"] else "")

    metadata = {
        "dur_type": t,
        "dur_desc": TYPES[t]["desc"],
        # 성분 A
        "ingr_code_a": get(item, "ingr_code_a", t),
        "ingredient_a_kor": ingr_a_kor,
        "ingredient_a_eng": ingr_a_eng,
        "main_ingr_a": get(item, "main_ingr_a", t),
        "mix_type_a": get(item, "mix_type_a", t),
        "mix_ingr_a": get(item, "mix_ingr_a"),
        # 성분 B
        "ingr_code_b": get(item, "ingr_code_b", t) if TYPES[t]["pair"] else "",
        "ingredient_b_kor": get(item, "ingr_kor_b", t) if TYPES[t]["pair"] else "",
        "ingredient_b_eng": get(item, "ingr_eng_b", t) if TYPES[t]["pair"] else "",
        "main_ingr_b": get(item, "main_ingr_b", t) if TYPES[t]["pair"] else "",
        "mix_type_b": get(item, "mix_type_b") if TYPES[t]["pair"] else "",
        "mix_ingr_b": get(item, "mix_ingr_b") if TYPES[t]["pair"] else "",
        # 제품 A
        "item_seq": get(item, "item_seq", t),
        "item_name": get(item, "item_name", t),
        "manufacturer": get(item, "manufacturer"),
        "class_code": get(item, "class_code"),
        "class_name": get(item, "class_name"),
        "etc_otc": get(item, "etc_otc", t),
        "form_name": get(item, "form_name"),
        # 제품 B
        "mixture_item_seq": get(item, "mixture_item_seq", t) if TYPES[t]["pair"] else "",
        "mixture_item_name": get(item, "mixture_item_name") if TYPES[t]["pair"] else "",
        "mixture_manufacturer": get(item, "mixture_manufacturer") if TYPES[t]["pair"] else "",
        "mixture_class_code": get(item, "mixture_class_code") if TYPES[t]["pair"] else "",
        "mixture_class_name": get(item, "mixture_class_name") if TYPES[t]["pair"] else "",
        "mixture_etc_otc": get(item, "mixture_etc_otc") if TYPES[t]["pair"] else "",
        "mixture_form_name": get(item, "mixture_form_name") if TYPES[t]["pair"] else "",
        # 금기 / 주의 내용
        "reason": reason,
        "remark": remark,
        "effect_name": effect_name,
        "sers_name": get(item, "sers_name") if t == "effect_duplicate" else "",
        "max_age_months": (parse_max_age_months(get(item, "age_base"), prohbt, remark)
                           if t == "age_specific_ban" else None),
        # 일자
        "dur_seq": get(item, "dur_seq"),
        "notification_date": get(item, "notification_date"),
        "change_date": get(item, "change_date"),
        # 출처
        **SOURCE_META,
        "etl_at": datetime.now().isoformat(),
    }

    return {
        "doc_id": "dur_" + "_".join([t] + make_id_parts(item, t)),
        "doc_type": "dur_interaction",
        "text": build_text(metadata),
        "metadata": metadata,
    }


# ─────────────────────────────────────
# 리포트
# ─────────────────────────────────────
def report(type_items: dict):
    logger.info("─" * 60)
    logger.info("필드 매핑 리포트 (결측률 30% 이상만 표시)")
    for t, total in _type_total.items():
        for field, cnt in _miss[t].most_common():
            rate = cnt / total * 100 if total else 0
            if rate >= 30:
                logger.warning(f"  [{t}] {field:18} 결측 {rate:5.1f}%  (후보 키: {F[field]})")
        if _exact_dup[t] or _collision[t]:
            logger.info(f"  [{t}] 완전 중복 제거 {_exact_dup[t]:,}건 / id 같고 내용 달라 보존 {_collision[t]:,}건")
        unmapped = sorted(_seen_keys[t] - KNOWN_KEYS)
        if unmapped:
            logger.info(f"  [{t}] 매핑되지 않은 원본 키: {unmapped}")

    logger.info("─" * 60)
    for t, items in type_items.items():
        if not items:
            continue
        n_code = sum(bool(d["metadata"]["ingr_code_a"]) for d in items)
        msg = f"  [{TYPES[t]['desc']}] {len(items)}건 | 성분코드 보유 {n_code / len(items) * 100:.1f}%"
        if t == "age_specific_ban":
            n_age = sum(d["metadata"]["max_age_months"] is not None for d in items)
            msg += f" | 연령 파싱 성공 {n_age}건"
        if TYPES[t]["pair"]:
            n_code_b = sum(bool(d["metadata"]["ingr_code_b"]) for d in items)
            msg += f" | 상대 성분코드 보유 {n_code_b / len(items) * 100:.1f}%"
        logger.info(msg)
    logger.info("─" * 60)


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────
def main():
    logger.info("=" * 50)
    logger.info("[ETL 2] DUR 정제 시작 (v2)")
    logger.info("=" * 50)

    keys = [k for k in dated_keys("bronze/api/dur/") if k.endswith(".json")]
    logger.info(f"Bronze 파일: {len(keys)}개")

    type_items = {t: [] for t in TYPES}
    seen_ids = {t: set() for t in TYPES}
    seen_sigs = {t: set() for t in TYPES}
    total_raw, dup = 0, 0

    for i, key in enumerate(keys):
        try:
            data = download_json(key)
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")
            continue
        if not data:
            continue

        type_key = data.get("type", "")
        items = data.get("items", [])
        total_raw += len(items)
        if type_key not in TYPES:
            continue

        for item in items:
            _type_total[type_key] += 1
            _seen_keys[type_key].update(item.keys())

            doc = transform_dur_item(item, type_key)
            if not doc:
                continue
            sig = content_sig(doc)
            if sig in seen_sigs[type_key]:              # 내용까지 같음 → 제거
                _exact_dup[type_key] += 1
                dup += 1
                continue
            seen_sigs[type_key].add(sig)
            if doc["doc_id"] in seen_ids[type_key]:     # id만 같고 내용이 다름 → 보존
                doc["doc_id"] = f"{doc['doc_id']}_{sig[:8]}"
                _collision[type_key] += 1
            seen_ids[type_key].add(doc["doc_id"])
            type_items[type_key].append(doc)

        if (i + 1) % 1000 == 0:
            logger.info(f"  진행: {i + 1}/{len(keys)}개 파일 처리")

    logger.info(f"원본: {total_raw}건 / 완전 중복 제거: {dup}건 / 내용이 달라 보존: {sum(_collision.values())}건")
    report(type_items)

    total_saved = 0
    for type_key, items in type_items.items():
        if not items:
            continue
        desc = TYPES[type_key]["desc"]
        batch_size = 500
        for j in range(0, len(items), batch_size):
            batch = items[j:j + batch_size]
            upload_json({
                "dur_type": type_key,
                "description": desc,
                "count": len(batch),
                "items": batch,
            }, silver_key(f"dur_interactions/{type_key}", f"batch_{j // batch_size:04d}.json"))
            total_saved += len(batch)
        logger.info(f"  [{desc}] ✅ {len(items)}건 저장")

    logger.info(f"✅ DUR Silver 저장 완료: {total_saved}건")
    return total_saved


if __name__ == "__main__":
    main()
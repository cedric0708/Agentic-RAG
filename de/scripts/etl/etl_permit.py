"""
[ETL 3] 의약품 제품 허가정보 Bronze → Silver  (v2: 실제 필드명 반영, 2026-10-07 probe 검증)
입력: s3://bucket/bronze/api/drug_permit/{dtl,mcpn}/
출력:
  s3://bucket/silver/drugs/drug_permit/      제품 마스터 (doc_type=drug_permit, VectorDB 대상)
  s3://bucket/silver/drug_ingredients/       제품-주성분 (doc_type=drug_ingredient, Graph CONTAINS 원천)
                                             ※ VectorDB 대상이 아니므로 silver/drugs/ 밖에 둠

확인된 원본 필드
  dtl : ITEM_SEQ, ITEM_NAME, ITEM_ENG_NAME, ENTP_NAME, ETC_OTC_CODE(값: 전문/일반의약품),
        MAIN_ITEM_INGR('[M040702]포도당', 구분자 '/'), MAIN_INGR_ENG('Glucose/...'),
        INGR_NAME(= 첨가제! 주성분 아님), MATERIAL_NAME(성분·분량 원문),
        CANCEL_NAME('정상'/취소), MAKE_MATERIAL_FLAG(완제/원료), BAR_CODE, EDI_CODE, ATC_CODE,
        EE/UD/NB_DOC_DATA(XML), EE/UD/NB_DOC_ID(PDF URL)
  mcpn: ITEM_SEQ, PRDUCT, ENTRPS, MTRAL_SN, MTRAL_CODE, MTRAL_NM, QNT, INGD_UNIT_CD,
        MAIN_INGR_ENG(제품 단위 영문 성분 목록), TAMT_SEQ(총량 순번)

스키마 (기존 silver와 동일한 래퍼):
  - text     : 임베딩/표기용 텍스트
  - sections : 효능/용법/주의사항 원문 (refined 단계 섹션 청킹용, drug_permit만)
  - metadata : 식별/필터/출처 정보
"""
import html
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).resolve().parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import list_keys, dated_keys, download_json, upload_json, silver_key

SOURCE_META = {
    "source_name": "식품의약품안전처 의약품 제품 허가정보",
    "source_url": "https://www.data.go.kr/data/15095677/openapi.do",
    "api_endpoint": "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08",
    "official_url_template": "https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_seq}",
    "license": "공공누리 1유형",
    "provider": "식품의약품안전처",
}

# 논리 필드 → 원본 키 (probe로 확인된 키만 사용)
FIELD = {
    # dtl
    "item_seq":       ["ITEM_SEQ"],
    "item_name":      ["ITEM_NAME", "PRDUCT"],
    "item_eng_name":  ["ITEM_ENG_NAME"],
    "entp_name":      ["ENTP_NAME", "ENTRPS"],
    "etc_otc":        ["ETC_OTC_CODE"],
    "permit_date":    ["ITEM_PERMIT_DATE"],
    "permit_kind":    ["PERMIT_KIND_NAME"],
    "make_flag":      ["MAKE_MATERIAL_FLAG"],
    "cancel_name":    ["CANCEL_NAME"],
    "cancel_date":    ["CANCEL_DATE"],
    "change_date":    ["CHANGE_DATE"],
    "main_ingr":      ["MAIN_ITEM_INGR"],
    "main_ingr_eng":  ["MAIN_INGR_ENG"],
    "additives":      ["INGR_NAME"],          # ⚠️ 첨가제 (주성분 아님)
    "material_raw":   ["MATERIAL_NAME"],
    "total_content":  ["TOTAL_CONTENT"],
    "atc_code":       ["ATC_CODE"],
    "edi_code":       ["EDI_CODE"],
    "bar_code":       ["BAR_CODE"],
    "chart":          ["CHART"],
    "storage":        ["STORAGE_METHOD"],
    "valid_term":     ["VALID_TERM"],
    "rare_drug":      ["RARE_DRUG_YN"],
    "ee_doc":         ["EE_DOC_DATA"],
    "ud_doc":         ["UD_DOC_DATA"],
    "nb_doc":         ["NB_DOC_DATA"],
    "ee_pdf":         ["EE_DOC_ID"],
    "ud_pdf":         ["UD_DOC_ID"],
    "nb_pdf":         ["NB_DOC_ID"],
    # mcpn
    "ingr_sn":        ["MTRAL_SN"],
    "ingr_code":      ["MTRAL_CODE"],
    "ingr_name":      ["MTRAL_NM"],
    "amount":         ["QNT"],
    "unit":           ["INGD_UNIT_CD"],
    "tamt_seq":       ["TAMT_SEQ"],
}

NULLS = {"none", "null", "nan", "-"}
_miss = defaultdict(Counter)  # 매핑 실패 통계: _miss[op][field]


def clean_text(text) -> str:
    if text is None:
        return ""
    s = " ".join(str(text).split()).strip()
    return "" if s.lower() in NULLS else s


def pick(item: dict, field: str, op: str = "") -> str:
    for k in FIELD[field]:
        v = clean_text(item.get(k))
        if v:
            return v
    if op:
        _miss[op][field] += 1
    return ""


def split_list(raw: str) -> list[str]:
    """'a/b/c', 'a|b', 'a,b,' → ['a','b','c'] (빈값 제거)"""
    return [x for x in (clean_text(p) for p in re.split(r"[|/,]", raw or "")) if x]


def strip_doc(xml: str) -> str:
    """허가 문서(EE/UD/NB_DOC_DATA) XML → 평문"""
    if not xml:
        return ""
    s = str(xml).replace("<![CDATA[", "").replace("]]>", "")
    s = re.sub(r"<ARTICLE[^>]*title=\"([^\"]*)\"[^>]*>", r" \1: ", s)  # 조항 제목 보존
    s = re.sub(r"<[^>]+>", " ", s)
    return clean_text(html.unescape(s))


def normalize_etc_otc(v: str) -> str:
    if "일반" in v:
        return "일반의약품"
    if "전문" in v:
        return "전문의약품"
    return v


def is_active(cancel_name: str) -> bool:
    return (not cancel_name) or cancel_name == "정상"


INGR_CODE_RE = re.compile(r"\[([A-Z]?\d+)\]\s*([^|/\[]+)")


def parse_main_ingr(raw: str) -> list[tuple[str, str]]:
    """'[M222857]세티리진염산염/[M222857]세티리진염산염' → [('M222857','세티리진염산염')]
    구분자 '/' 또는 '|', 반복 코드는 순서 유지하며 중복 제거
    """
    if not raw:
        return []
    out = [(m.group(1), clean_text(m.group(2))) for m in INGR_CODE_RE.finditer(raw)]
    if not out:
        out = [("", x) for x in split_list(raw)]
    seen, uniq = set(), []
    for code, name in out:
        key = code or name
        if key and key not in seen:
            seen.add(key)
            uniq.append((code, name))
    return uniq


def load_bronze(op: str, workers: int = 16) -> list[dict]:
    """Bronze 파일 병렬 다운로드 (dtl은 문서 원문 XML이 포함되어 파일이 큼)"""
    keys = sorted(k for k in dated_keys(f"bronze/api/drug_permit/{op}/") if k.endswith(".json"))
    logger.info(f"[{op}] Bronze 파일: {len(keys)}개 (병렬 {workers})")

    def fetch(key):
        try:
            data = download_json(key)
            return key, (data or {}).get("items", []), None
        except Exception as e:
            return key, [], str(e)[:60]

    by_key, done, step = {}, 0, max(len(keys) // 10, 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, items, err in ex.map(fetch, keys):
            if err:
                logger.warning(f"  스킵 [{key}]: {err}")
            by_key[key] = items
            done += 1
            if done % step == 0 or done == len(keys):
                logger.info(f"  [{op}] 다운로드 {done}/{len(keys)}")

    items = [it for k in keys for it in by_key.get(k, [])]   # 파일 순서 유지
    logger.info(f"[{op}] 원본 레코드: {len(items)}건")
    return items


# ─────────────────────────────────────
# 1. 제품 마스터 (dtl)
# ─────────────────────────────────────
def transform_product(item: dict, etl_at: str) -> dict | None:
    item_seq = pick(item, "item_seq", "dtl")
    item_name = pick(item, "item_name", "dtl")
    if not item_seq or not item_name:
        return None

    etc_otc = normalize_etc_otc(pick(item, "etc_otc", "dtl"))
    main_ingr = pick(item, "main_ingr", "dtl")
    cancel_name = pick(item, "cancel_name")
    make_flag = pick(item, "make_flag")
    sections = {
        "efficacy": strip_doc(pick(item, "ee_doc", "dtl")),
        "usage": strip_doc(pick(item, "ud_doc", "dtl")),
        "caution": strip_doc(pick(item, "nb_doc", "dtl")),
    }
    ingr_names = [n for _, n in parse_main_ingr(main_ingr)]

    parts = [f"약품명: {item_name}", f"구분: {etc_otc}"]
    if ingr_names:
        parts.append(f"주성분: {', '.join(ingr_names)}")
    if sections["efficacy"]:
        parts.append(f"효능: {sections['efficacy']}")
    if sections["usage"]:
        parts.append(f"용법: {sections['usage']}")
    # 주의사항(NB)은 매우 길어 text에서 제외 → sections.caution으로 별도 청킹

    return {
        "doc_id": f"permit_{item_seq}",
        "doc_type": "drug_permit",
        "text": " | ".join(parts),
        "sections": sections,
        "metadata": {
            "name": item_name,
            "name_eng": pick(item, "item_eng_name"),
            "item_seq": item_seq,
            "manufacturer": pick(item, "entp_name", "dtl"),
            "etc_otc": etc_otc,
            "is_otc": etc_otc == "일반의약품",
            "is_active": is_active(cancel_name),
            "is_finished_drug": make_flag == "완제의약품",
            "make_material_flag": make_flag,
            "permit_kind": pick(item, "permit_kind"),
            "cancel_name": cancel_name,
            "cancel_date": pick(item, "cancel_date"),
            "permit_date": pick(item, "permit_date", "dtl"),
            "change_date": pick(item, "change_date"),
            "main_ingr_raw": main_ingr,
            "main_ingr_eng_raw": pick(item, "main_ingr_eng"),
            "additives_raw": pick(item, "additives"),     # 첨가제 (알레르기 확인용)
            "material_raw": pick(item, "material_raw"),   # 성분·분량 원문
            "total_content": pick(item, "total_content"),
            "atc_code": pick(item, "atc_code"),
            "edi_codes": split_list(pick(item, "edi_code")),
            "bar_codes": split_list(pick(item, "bar_code")),   # 바코드 → 복용약 입력 기능용
            "chart": pick(item, "chart"),
            "storage": pick(item, "storage"),
            "valid_term": pick(item, "valid_term"),
            "rare_drug": pick(item, "rare_drug") == "Y",
            "doc_pdf_urls": {
                "efficacy": pick(item, "ee_pdf"),
                "usage": pick(item, "ud_pdf"),
                "caution": pick(item, "nb_pdf"),
            },
            **{k: v for k, v in SOURCE_META.items() if k != "official_url_template"},
            "official_url": SOURCE_META["official_url_template"].format(item_seq=item_seq),
            "etl_at": etl_at,
        },
    }


# ─────────────────────────────────────
# 2. 제품-주성분 (mcpn 우선, 없으면 dtl의 MAIN_ITEM_INGR로 보완)
# ─────────────────────────────────────
def ingredient_doc(item_seq, product, fallback_name, sn, code, name_kor, name_eng,
                   amount, unit, tamt_seq, source, eng_source, etl_at):
    pm = product["metadata"] if product else {}
    item_name = pm.get("name") or fallback_name
    return {
        "doc_id": f"permit_ingr_{item_seq}_{code or name_kor}",
        "doc_type": "drug_ingredient",
        "text": f"제품: {item_name} | 성분: {name_kor} {amount}{unit}".strip(),
        "metadata": {
            "item_seq": item_seq,
            "item_name": item_name,
            "etc_otc": pm.get("etc_otc", ""),
            "is_otc": pm.get("is_otc", False),
            "is_active": pm.get("is_active", True),
            "is_finished_drug": pm.get("is_finished_drug", True),
            "ingr_sn": sn,
            "ingr_code": code,           # 식약처 원료코드(M코드) → DUR 성분코드(D코드)와 별도 매핑 필요
            "ingr_name_kor": name_kor,
            "ingr_name_eng": name_eng,
            "eng_source": eng_source,    # 'main_ingr_eng_order' (순서 대응) | ''
            "amount": amount,
            "unit": unit,
            "tamt_seq": tamt_seq,
            "ingr_source": source,       # 'mcpn' | 'dtl_main_item_ingr'
            **{k: v for k, v in SOURCE_META.items() if k != "official_url_template"},
            "etl_at": etl_at,
        },
    }


def _int(v: str) -> int:
    return int(v) if v and v.isdigit() else 0


def build_ingredients(mcpn_items, products_by_seq, etl_at) -> list[dict]:
    by_seq = defaultdict(list)
    for it in mcpn_items:
        seq = pick(it, "item_seq", "mcpn")
        if seq:
            by_seq[seq].append(it)

    docs = []
    eng_assigned = 0
    for seq, rows in by_seq.items():
        # 총량순번 → 원료순번 순으로 정렬 후, 성분코드 기준 중복 제거 (여러 규격의 같은 성분 반복 제거)
        rows.sort(key=lambda r: (_int(pick(r, "tamt_seq")), _int(pick(r, "ingr_sn"))))
        uniq, seen = [], set()
        for r in rows:
            key = pick(r, "ingr_code", "mcpn") or pick(r, "ingr_name", "mcpn")
            if key and key not in seen:
                seen.add(key)
                uniq.append(r)
        if not uniq:
            continue

        # 영문명: 제품 단위 MAIN_INGR_ENG를 성분 개수가 같을 때만 순서대로 대응
        eng = split_list(pick(uniq[0], "main_ingr_eng"))
        eng_ok = len(eng) == len(uniq)
        eng_assigned += len(uniq) if eng_ok else 0

        for idx, r in enumerate(uniq):
            docs.append(ingredient_doc(
                seq, products_by_seq.get(seq), pick(r, "item_name"),
                pick(r, "ingr_sn"), pick(r, "ingr_code"), pick(r, "ingr_name"),
                eng[idx] if eng_ok else "",
                pick(r, "amount"), pick(r, "unit"), pick(r, "tamt_seq"),
                "mcpn", "main_ingr_eng_order" if eng_ok else "", etl_at))
    n_mcpn = len(docs)

    # fallback: mcpn에 없는 제품은 dtl의 MAIN_ITEM_INGR 파싱
    for seq, prod in products_by_seq.items():
        if seq in by_seq:
            continue
        pm = prod["metadata"]
        pairs = parse_main_ingr(pm["main_ingr_raw"])
        eng = split_list(pm["main_ingr_eng_raw"])
        eng_ok = len(eng) == len(pairs)
        for idx, (code, name_kor) in enumerate(pairs):
            docs.append(ingredient_doc(
                seq, prod, pm["name"], str(idx + 1), code, name_kor,
                eng[idx] if eng_ok else "", "", "", "",
                "dtl_main_item_ingr", "main_ingr_eng_order" if eng_ok else "", etl_at))

    logger.info(f"주성분 레코드: mcpn {n_mcpn}건 + fallback {len(docs) - n_mcpn}건 "
                f"(영문명 순서 대응 {eng_assigned}건)")
    return docs


# ─────────────────────────────────────
# 3. 저장 / 품질 리포트
# ─────────────────────────────────────
def save(docs: list[dict], prefix: str, batch_size: int = 100) -> int:
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        upload_json({"count": len(batch), "items": batch},
                    silver_key(prefix, f"batch_{i // batch_size:04d}.json"))
    return len(docs)


def report(products: list[dict], ingredients: list[dict], n_dtl: int, n_mcpn: int):
    for op, total in (("dtl", n_dtl), ("mcpn", n_mcpn)):
        for field, cnt in _miss[op].most_common():
            rate = cnt / total * 100 if total else 0
            if rate > 50:
                logger.warning(f"  ⚠️ [{op}] '{field}' 결측 {rate:.0f}% (키: {FIELD[field]})")

    with_ingr = {d["metadata"]["item_seq"] for d in ingredients}
    otc = [p for p in products
           if p["metadata"]["is_otc"] and p["metadata"]["is_active"] and p["metadata"]["is_finished_drug"]]
    otc_cov = sum(p["metadata"]["item_seq"] in with_ingr for p in otc)
    dup = sum(1 for c in Counter(d["doc_id"] for d in ingredients).values() if c > 1)
    with_bar = sum(bool(p["metadata"]["bar_codes"]) for p in otc)

    logger.info("─" * 55)
    logger.info(f"제품 {len(products)}건 (유효 완제 일반의약품 {len(otc)}건)")
    logger.info(f"유효 완제 일반의약품 성분 커버리지: {otc_cov}/{len(otc)} "
                f"({otc_cov / max(len(otc), 1) * 100:.1f}%)")
    logger.info(f"유효 완제 일반의약품 바코드 보유: {with_bar}/{len(otc)}")
    logger.info(f"주성분 doc_id 중복: {dup}건" + (" ⚠️" if dup else ""))
    logger.info("─" * 55)


def main():
    logger.info("=" * 50)
    logger.info("[ETL 3] 의약품 제품 허가정보 정제 시작 (v2)")
    logger.info("=" * 50)
    etl_at = datetime.now().isoformat()

    dtl_raw = load_bronze("dtl")
    products_by_seq, skipped = {}, 0
    for item in dtl_raw:
        doc = transform_product(item, etl_at)
        if not doc:
            skipped += 1
            continue
        products_by_seq.setdefault(doc["metadata"]["item_seq"], doc)
    products = list(products_by_seq.values())
    logger.info(f"제품 변환 성공: {len(products)}건 / 스킵: {skipped}건")

    mcpn_raw = load_bronze("mcpn")
    ingredients = build_ingredients(mcpn_raw, products_by_seq, etl_at)

    report(products, ingredients, len(dtl_raw), len(mcpn_raw))

    n1 = save(products, "drugs/drug_permit")
    n2 = save(ingredients, "drug_ingredients")
    logger.info(f"✅ Silver 저장 완료: 제품 {n1}건 / 주성분 {n2}건")
    return n1, n2


if __name__ == "__main__":
    main()
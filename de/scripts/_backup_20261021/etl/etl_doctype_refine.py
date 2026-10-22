"""
[ETL 추가] doc_type 세분화
기존 Silver 데이터를 Agent별 최적 doc_type으로 재분류

변환 규칙:
  drug_info       → drug_efficacy / drug_usage / drug_caution / drug_side_effect
  symptom_disease → symptom_redflag / symptom_info
  dur_interaction → 유지

출력: s3://bucket/silver/refined/

수정 이력:
  v2 (symptom):
    - 문제: has_redflag를 무시하고 '|' 조각마다 넓은 키워드("위험", "심각한", "출혈", "[예방]", "[치료]" 등)로
            redflag를 다시 나눔 → 예방법·치료 설명이 위험 신호로 분류됨, 질환마다 redflag 문서가 2개씩 생김
    - 수정: etl_crawl.py(v4)가 정한 has_redflag를 그대로 사용 (True → symptom_redflag, 그 외 → symptom_info)
    - 원천별 최신 날짜 폴더만 읽음 (예전 날짜 데이터가 섞이지 않게)
    - 저장 파일명에 원천 이름을 붙임 (health_portal / health_kr_disease의 batch_0000 덮어쓰기 방지)
    - --dry-run: 업로드 없이 건수만 출력
"""
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import list_keys, download_json, upload_json, silver_key

DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/")


def latest_keys(prefix: str) -> list:
    """prefix 아래 날짜 폴더 중 최신 날짜의 .json 파일만"""
    keys = [k for k in list_keys(prefix) if k.endswith(".json")]
    dates = sorted({m.group(1) for k in keys if (m := DATE_RE.search(k))})
    if not dates:
        return keys
    return [k for k in keys if f"/{dates[-1]}/" in k]


def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


# ── 위험 증상 키워드 (Red flag 판단용) ──────────────────────
REDFLAG_KEYWORDS = [
    "즉시 병원", "응급", "119", "구급",
    "의사와 상담", "의사 진료", "전문의",
    "심각한", "위험", "사망", "쇼크",
    "호흡 곤란", "흉통", "의식 잃", "경련",
    "마비", "출혈", "고열", "40도",
    "48시간 이상", "증상이 악화", "지속될 경우",
    "병원을 방문", "즉각 중지",
]

def is_redflag(text: str) -> bool:
    """Red flag 증상 여부 판단"""
    return any(kw in text for kw in REDFLAG_KEYWORDS)


# ── drug_info 세분화 ─────────────────────────────────────────

def split_drug_info(item: dict) -> list:
    """
    drug_info 1건 → 최대 4건으로 세분화
    (효능 / 용법 / 주의사항 / 부작용)
    """
    results = []
    text = item.get("text", "")
    meta = item.get("metadata", {})
    doc_id = item.get("doc_id", "")

    # text에서 섹션별 추출
    sections = {
        "efficacy":    [],
        "usage":       [],
        "caution":     [],
        "side_effect": [],
    }

    # | 구분자로 파싱
    parts = text.split("|")
    for part in parts:
        part = clean_text(part)
        if not part:
            continue

        if any(kw in part for kw in ["효능:", "효과:", "적응:"]):
            sections["efficacy"].append(part)
        elif any(kw in part for kw in ["용법:", "용량:", "복용:"]):
            sections["usage"].append(part)
        elif any(kw in part for kw in ["주의사항:", "주의:", "경고:"]):
            sections["caution"].append(part)
        elif any(kw in part for kw in ["부작용:", "이상반응:"]):
            sections["side_effect"].append(part)

    # 약품명 파트 (공통)
    name_part = ""
    for part in parts:
        part = clean_text(part)
        if part.startswith("약품명:"):
            name_part = part
            break

    # 섹션별 문서 생성
    type_map = {
        "efficacy":    "drug_efficacy",
        "usage":       "drug_usage",
        "caution":     "drug_caution",
        "side_effect": "drug_side_effect",
    }

    for section_key, doc_type in type_map.items():
        section_text = " | ".join(sections[section_key])
        if not section_text:
            continue

        # 약품명 + 섹션 내용
        full_text = f"{name_part} | {section_text}" if name_part else section_text

        new_item = {
            "doc_id":   f"{doc_id}_{section_key}",
            "doc_type": doc_type,
            "text":     full_text,
            "metadata": {
                **meta,
                "doc_type":        doc_type,
                "parent_doc_id":   doc_id,
                "section":         section_key,
            }
        }

        # drug_usage에 structured_usage 포함
        if section_key == "usage" and "structured_usage" in meta:
            new_item["metadata"]["structured_usage"] = meta["structured_usage"]

        results.append(new_item)

    # 섹션이 하나도 없으면 원본 유지 (drug_efficacy로 분류)
    if not results:
        results.append({
            "doc_id":   f"{doc_id}_efficacy",
            "doc_type": "drug_efficacy",
            "text":     text,
            "metadata": {**meta, "doc_type": "drug_efficacy", "parent_doc_id": doc_id},
        })

    return results


# ── symptom_disease 세분화 ───────────────────────────────────

def split_symptom(item: dict) -> list:
    """
    symptom_disease 1건 → symptom_redflag 또는 symptom_info (1건 그대로)
    분류는 etl_crawl.py(v4)가 정한 metadata.has_redflag를 따름
      - True  : "병원/응급실에 가야 하는 경우" 문장만 모은 문서 → symptom_redflag
      - 그 외 : 질환 설명 전체 → symptom_info (has_redflag가 없는 약학정보원 질병정보 포함)
    """
    text = item.get("text", "")
    meta = item.get("metadata", {})
    doc_id = item.get("doc_id", "")
    if not clean_text(text):
        return []

    doc_type = "symptom_redflag" if meta.get("has_redflag") is True else "symptom_info"
    new_meta = {**meta, "doc_type": doc_type, "parent_doc_id": doc_id}
    if doc_type == "symptom_redflag":
        new_meta["is_redflag"] = True
    return [{
        "doc_id":   doc_id,        # etl_crawl에서 이미 _info / _redflag로 끝남
        "doc_type": doc_type,
        "text":     text,
        "metadata": new_meta,
    }]


# ── 전체 처리 ────────────────────────────────────────────────

def process_drug_info() -> int:
    logger.info("▶ [drug_info] 세분화 시작")
    prefixes = [
        "silver/drugs/public_drug/",
        "silver/drugs/health_kr_drug/",
    ]
    total = 0
    for prefix in prefixes:
        keys = [k for k in list_keys(prefix) if k.endswith(".json")]
        logger.info(f"  {prefix}: {len(keys)}개 파일")

        for key in keys:
            try:
                data = download_json(key)
                items = data.get("items", [])
                refined_items = []

                for item in items:
                    refined_items.extend(split_drug_info(item))

                # silver/refined/drugs/ 에 저장
                filename = key.split("/")[-1]
                s_key = silver_key("refined/drugs", filename)
                upload_json({
                    "count": len(refined_items),
                    "items": refined_items,
                }, s_key)
                total += len(refined_items)

            except Exception as e:
                logger.error(f"  [X] {key}: {str(e)[:80]}")

    logger.info(f"  [O] drug_info 세분화 완료: {total}건")
    return total


def process_symptom(dry_run: bool = False) -> int:
    logger.info("▶ [symptom_disease] 세분화 시작 (v2: has_redflag 기준)")
    counts = {}
    prefixes = [
        "silver/symptoms/health_portal/",
        "silver/symptoms/health_kr_disease/",
    ]
    total = 0
    for prefix in prefixes:
        keys = latest_keys(prefix)
        source = prefix.rstrip("/").split("/")[-1]
        logger.info(f"  {prefix}: {len(keys)}개 파일 (최신 날짜만)")

        for key in keys:
            try:
                data = download_json(key)
                items = data.get("items", [])
                refined_items = []

                for item in items:
                    refined_items.extend(split_symptom(item))

                for it in refined_items:
                    counts[it["doc_type"]] = counts.get(it["doc_type"], 0) + 1

                filename = f"{source}_{key.split('/')[-1]}"   # 원천끼리 batch_0000 덮어쓰기 방지
                if not dry_run:
                    s_key = silver_key("refined/symptoms", filename)
                    upload_json({
                        "count": len(refined_items),
                        "items": refined_items,
                    }, s_key)
                total += len(refined_items)

            except Exception as e:
                logger.error(f"  ❌ {key}: {str(e)[:80]}")

    logger.info(f"  ✅ symptom 세분화 완료: {total}건 {counts}" + (" (dry-run, 업로드 안 함)" if dry_run else ""))
    return total


def process_dur() -> int:
    """DUR은 그대로 복사 (세분화 불필요)"""
    logger.info("▶ [dur_interaction] 복사 (변경 없음)")
    prefixes = [
        "silver/dur_interactions/combination_ban/",
        "silver/dur_interactions/pregnancy_ban/",
        "silver/dur_interactions/elderly_caution/",
        "silver/dur_interactions/dose_caution/",
        "silver/dur_interactions/effect_duplicate/",
        "silver/dur_interactions/age_specific_ban/",
        "silver/dur_interactions/dosing_period_caution/",
    ]
    total = 0
    for prefix in prefixes:
        keys = [k for k in list_keys(prefix) if k.endswith(".json")]
        for key in keys:
            try:
                data = download_json(key)
                filename = key.split("/")[-1]
                dur_type = prefix.split("/")[-2]
                s_key = silver_key(f"refined/dur/{dur_type}", filename)
                upload_json(data, s_key)
                total += data.get("count", 0)
            except Exception as e:
                logger.error(f"  ❌ {key}: {str(e)[:80]}")

    logger.info(f"  ✅ DUR 복사 완료: {total}건")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="S3에 올리지 않고 건수만 출력")
    args = ap.parse_args()

    logger.info("=" * 50)
    logger.info("[ETL] doc_type 세분화 시작")
    logger.info("=" * 50)

    total = 0
    # total += process_drug_info()
    total += process_symptom(dry_run=args.dry_run)
    # total += process_dur()

    logger.info("=" * 50)
    logger.info(f"전체 완료: {total}건")
    logger.info("=" * 50)

    # 결과 요약
    logger.info("\n=== doc_type 세분화 결과 ===")
    logger.info("drug_info     → drug_efficacy / drug_usage / drug_caution / drug_side_effect")
    logger.info("symptom       → symptom_info / symptom_redflag")
    logger.info("dur           → 유지")
    logger.info("\nS3 저장 경로: silver/refined/")


if __name__ == "__main__":
    main()
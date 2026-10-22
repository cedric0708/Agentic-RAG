"""
[ETL 1] 공공데이터 의약품 API (e약은요) Bronze → Silver
입력: s3://bucket/bronze/api/public_data/
출력: s3://bucket/silver/drugs/public_drug/

스키마:
  - text     : VectorDB 임베딩 대상 텍스트
  - metadata : 출처, 제조사 등 필터링/표기용 메타데이터
"""
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import list_keys, download_json, upload_json, silver_key

# 공식 출처 정보
SOURCE_META = {
    "source_name": "식품의약품안전처 e약은요",
    "source_url": "https://www.data.go.kr/data/15075057/openapi.do",
    "official_url_template": "https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?openDataInfoSeq=11",
    "license": "공공누리 1유형",
    "license_url": "https://www.data.go.kr/ugs/selectPublicDataDetailView.do",
    "provider": "식품의약품안전처",
}


def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def build_text(item: dict) -> str:
    """VectorDB 임베딩용 텍스트 생성"""
    parts = []
    name = clean_text(item.get("itemName", ""))
    if name:
        parts.append(f"약품명: {name}")

    efficacy = clean_text(item.get("efcyQesitm", ""))
    if efficacy:
        parts.append(f"효능: {efficacy}")

    usage = clean_text(item.get("useMethodQesitm", ""))
    if usage:
        parts.append(f"용법: {usage}")

    caution = clean_text(item.get("atpnQesitm", ""))
    if caution:
        parts.append(f"주의사항: {caution}")

    warn = clean_text(item.get("atpnWarnQesitm", ""))
    if warn:
        parts.append(f"경고: {warn}")

    interaction = clean_text(item.get("intrcQesitm", ""))
    if interaction:
        parts.append(f"상호작용: {interaction}")

    side_effect = clean_text(item.get("seQesitm", ""))
    if side_effect:
        parts.append(f"부작용: {side_effect}")

    storage = clean_text(item.get("depositMethodQesitm", ""))
    if storage:
        parts.append(f"보관: {storage}")

    return " | ".join(parts)


def transform_item(item: dict) -> dict | None:
    item_seq = item.get("itemSeq", "")
    item_name = clean_text(item.get("itemName", ""))
    if not item_seq or not item_name:
        return None

    text = build_text(item)
    if not text:
        return None

    return {
        "doc_id": f"public_{item_seq}",
        "doc_type": "drug_info",
        "text": text,
        "metadata": {
            # 식별 정보
            "name": item_name,
            "item_seq": item_seq,
            # 제조사
            "manufacturer": clean_text(item.get("entpName", "")),
            # 이미지
            "image_url": item.get("itemImage") or "",
            # 업데이트
            "updated_at": item.get("updateDe", ""),
            # 출처 정보
            **SOURCE_META,
            # ETL
            "etl_at": datetime.now().isoformat(),
        }
    }


def main():
    logger.info("=" * 50)
    logger.info("[ETL 1] 공공데이터 의약품 API 정제 시작")
    logger.info("=" * 50)

    keys = [k for k in list_keys("bronze/api/public_data/") if k.endswith(".json")]
    logger.info(f"Bronze 파일: {len(keys)}개")

    all_items = {}
    for key in keys:
        try:
            data = download_json(key)
            if not data:
                continue
            for item in data.get("items", []):
                item_seq = item.get("itemSeq", "")
                if item_seq and item_seq not in all_items:
                    all_items[item_seq] = item
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")

    logger.info(f"중복 제거 후: {len(all_items)}건")

    transformed, skipped = [], 0
    for item_seq, item in all_items.items():
        result = transform_item(item)
        if result:
            transformed.append(result)
        else:
            skipped += 1

    logger.info(f"변환 성공: {len(transformed)}건 / 스킵: {skipped}건")

    batch_size = 100
    saved = 0
    for i in range(0, len(transformed), batch_size):
        batch = transformed[i:i+batch_size]
        s_key = silver_key("drugs/public_drug", f"batch_{i//batch_size:04d}.json")
        upload_json({"count": len(batch), "items": batch}, s_key)
        saved += len(batch)

    logger.info(f"✅ Silver 저장 완료: {saved}건")
    return saved


if __name__ == "__main__":
    main()
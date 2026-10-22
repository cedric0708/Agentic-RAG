"""
[수집 2] 공공데이터포털 - 의약품 제품 허가정보 (DrugPrdtPrmsnInfoService08)
  - dtl  : getDrugPrdtPrmsnDtlInq08  허가 상세 (품목, 주성분, 전문/일반, 허가·취소, 효능/용법/주의사항 문서)
  - mcpn : getDrugPrdtMcpnDtlInq08   주성분 상세 (제품 1 : 성분 N)  → 그래프 CONTAINS 원천
라이선스: 공공누리 1유형 (상업적 이용 가능)
저장경로: s3://bucket/bronze/api/drug_permit/{dtl|mcpn}/YYYY-MM-DD/

사용법:
  python collect_drug_permit_api.py --probe        # 응답 필드명 확인 (오퍼레이션별 3건)
  python collect_drug_permit_api.py                # 전체 수집 (dtl + mcpn)
  python collect_drug_permit_api.py --op mcpn      # 특정 오퍼레이션만 수집
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import bronze_key, upload_json, key_exists

API_KEY = os.getenv("PUBLIC_API_KEY")
BASE_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08"
OPERATIONS = {
    "dtl": "getDrugPrdtPrmsnDtlInq08",
    "mcpn": "getDrugPrdtMcpnDtlInq08",
}
NUM_OF_ROWS = 100
DAILY_LIMIT = 9500  # 오퍼레이션별 일일 트래픽 10,000건 → 여유분 확보


class ApiError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def fetch_page(op: str, page_no: int, rows: int = NUM_OF_ROWS) -> dict:
    params = {
        "serviceKey": API_KEY,  # 포털의 Decoding 키 사용 (requests가 인코딩 처리)
        "pageNo": page_no,
        "numOfRows": rows,
        "type": "json",
    }
    res = requests.get(f"{BASE_URL}/{OPERATIONS[op]}", params=params, timeout=60)
    res.raise_for_status()
    try:
        data = res.json()
    except ValueError:
        # 인증키 오류·트래픽 초과는 XML로 내려오는 경우가 있음
        raise ApiError(f"JSON 파싱 실패 (인증키/트래픽 확인): {res.text[:200]}")

    header = data.get("header") or data.get("response", {}).get("header", {}) or {}
    code = str(header.get("resultCode", "00"))
    if code not in ("00", "0"):
        raise ApiError(f"API 오류 resultCode={code} msg={header.get('resultMsg')}")
    return data


def parse_body(data: dict) -> tuple[list, int]:
    """응답 구조 차이 흡수
    {body:{items:[...]}} / {body:{items:{item:[...]}}} / {response:{body:...}}
    """
    body = data.get("body") or data.get("response", {}).get("body", {}) or {}
    items = body.get("items") or []
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):  # 1건이면 dict로 오는 경우
        items = [items]
    total = int(body.get("totalCount", 0) or 0)
    return items, total


def probe():
    """각 오퍼레이션의 실제 필드명 확인용 (ETL의 FIELD 매핑 검증)"""
    for op in OPERATIONS:
        data = fetch_page(op, 1, rows=3)
        items, total = parse_body(data)
        logger.info(f"\n[{op}] {OPERATIONS[op]} | totalCount={total}")
        if not items:
            logger.warning("  items 없음 - 응답 원문 확인:")
            logger.warning(json.dumps(data, ensure_ascii=False)[:500])
            continue
        for k, v in items[0].items():
            logger.info(f"  {k:28} | {str(v)[:70]!r}")


def collect(op: str):
    logger.info("=" * 50)
    logger.info(f"의약품 허가정보 수집 시작: {op} ({OPERATIONS[op]})")
    logger.info("=" * 50)

    first = fetch_page(op, 1)
    calls = 1
    _, total_count = parse_body(first)
    total_pages = math.ceil(total_count / NUM_OF_ROWS)
    logger.info(f"전체 데이터: {total_count}건 / {total_pages}페이지")

    if total_pages > DAILY_LIMIT:
        logger.warning(f"페이지 수가 일일 한도({DAILY_LIMIT})를 넘습니다. 나머지는 다음 실행에서 이어 받습니다.")

    collected, failed = 0, []
    for page_no in range(1, total_pages + 1):
        s3_key = bronze_key(f"api/drug_permit/{op}", f"all_p{page_no:05d}.json")

        if key_exists(s3_key):
            logger.debug(f"  스킵 (이미 존재): p{page_no}")
            continue

        if calls >= DAILY_LIMIT:
            logger.warning(f"  일일 트래픽 한도 도달 → p{page_no}부터 다음 실행에서 재개")
            break

        try:
            if page_no == 1:
                data = first
            else:
                data = fetch_page(op, page_no)
                calls += 1
        except Exception as e:
            logger.error(f"  ❌ 페이지 {page_no} 실패: {e}")
            failed.append(page_no)
            continue

        items, _ = parse_body(data)
        if not items:
            logger.info(f"  빈 페이지 도달: p{page_no} → 종료")
            break

        payload = {
            "source": "의약품제품허가정보_공공데이터포털",
            "operation": OPERATIONS[op],
            "license": "공공누리1유형",
            "page_no": page_no,
            "total_count": total_count,
            "num_items": len(items),
            "items": items,
        }
        upload_json(payload, s3_key)
        collected += len(items)
        logger.info(f"  📄 [{op}] 페이지 {page_no}/{total_pages}: {len(items)}건 (누적: {collected}건)")

        time.sleep(0.3)

    logger.info(f"\n✅ [{op}] 수집 완료: {collected}건 / API 호출 {calls}회")
    if failed:
        logger.warning(f"실패 페이지 {len(failed)}개: {failed[:20]} … (재실행 시 자동 재시도)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=list(OPERATIONS), help="특정 오퍼레이션만 수집")
    parser.add_argument("--probe", action="store_true", help="필드명 확인 (각 3건)")
    args = parser.parse_args()

    if not API_KEY:
        logger.error("PUBLIC_API_KEY 환경변수가 없습니다.")
        sys.exit(1)

    if args.probe:
        probe()
        return

    for op in ([args.op] if args.op else OPERATIONS):
        collect(op)


if __name__ == "__main__":
    main()
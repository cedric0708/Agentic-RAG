"""
[수집 4] 공공데이터포털 - 의약품안전사용서비스(DUR) 성분정보 (DURIrdntInfoService03)
  https://www.data.go.kr/data/15056780/openapi.do
  품목정보(제품 쌍 단위, 병용금기 약 65만 행)와 달리 '성분 단위'로 DUR 정보를 제공
  → 그래프 금기 엣지(DUR 성분코드 D코드 간 관계)의 기준 데이터
라이선스: 공공누리 1유형 (상업적 이용 가능)
저장경로: s3://bucket/bronze/api/dur_ingredient/{type}/YYYY-MM-DD/

사용법:
  python collect_dur_ingredient_api.py --probe                   # 응답 필드명 확인 (유형별 2건)
  python collect_dur_ingredient_api.py                           # 전체 유형 수집
  python collect_dur_ingredient_api.py --type combination_ban    # 특정 유형만
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(Path(__file__).resolve().parent.parent.parent / "config" / ".env", override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import bronze_key, upload_json, key_exists

API_KEY = os.getenv("PUBLIC_API_KEY")
BASE_URL = "https://apis.data.go.kr/1471000/DURIrdntInfoService03"
OPERATIONS = {
    # type_key: (오퍼레이션명, 설명)  — 품목정보 ETL과 같은 type_key 사용
    "combination_ban":       ("getUsjntTabooInfoList02",       "병용금기"),
    "age_specific_ban":      ("getSpcifyAgrdeTabooInfoList02", "특정연령대금기"),
    "pregnancy_ban":         ("getPwnmTabooInfoList02",        "임부금기"),
    "dose_caution":          ("getCpctyAtentInfoList02",       "용량주의"),
    "dosing_period_caution": ("getMdctnPdAtentInfoList02",     "투여기간주의"),
    "elderly_caution":       ("getOdsnAtentInfoList02",        "노인주의"),
    "effect_duplicate":      ("getEfcyDplctInfoList02",        "효능군중복"),
}
NUM_OF_ROWS = 100
DAILY_LIMIT = 9500  # 오퍼레이션별 일일 트래픽 10,000건 → 여유분 확보


class ApiError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def fetch_page(type_key: str, page_no: int, rows: int = NUM_OF_ROWS) -> dict:
    op = OPERATIONS[type_key][0]
    params = {"serviceKey": API_KEY, "pageNo": page_no, "numOfRows": rows, "type": "json"}
    res = requests.get(f"{BASE_URL}/{op}", params=params, timeout=60)
    res.raise_for_status()
    try:
        data = res.json()
    except ValueError:
        # 인증키 오류·미신청·트래픽 초과는 XML로 내려오는 경우가 있음
        raise ApiError(f"JSON 파싱 실패 (인증키/활용신청 확인): {res.text[:200]}")

    header = data.get("header") or data.get("response", {}).get("header", {}) or {}
    code = str(header.get("resultCode", "00"))
    if code not in ("00", "0"):
        raise ApiError(f"API 오류 resultCode={code} msg={header.get('resultMsg')}")
    return data


def parse_body(data: dict) -> tuple[list, int]:
    """응답 구조 차이 흡수: {body:{items:[...]}} / {body:{items:{item:[...]}}} / {response:{body:...}}"""
    body = data.get("body") or data.get("response", {}).get("body", {}) or {}
    items = body.get("items") or []
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    return unwrap_items(items), int(body.get("totalCount", 0) or 0)


def unwrap_items(items: list) -> list:
    """성분정보 API는 레코드를 {'item': {...}}로 한 겹 더 감싸서 반환 → 벗겨서 평탄화"""
    out = []
    for x in items:
        if isinstance(x, dict) and set(x) == {"item"} and isinstance(x["item"], dict):
            out.append(x["item"])
        else:
            out.append(x)
    return out


def probe():
    ok, ng = [], []
    for type_key, (op, desc) in OPERATIONS.items():
        try:
            items, total = parse_body(fetch_page(type_key, 1, rows=2))
        except Exception as e:
            logger.error(f"\n[{type_key}] {op} ({desc}) ❌ {e}")
            ng.append(type_key)
            continue
        ok.append(type_key)
        logger.info(f"\n[{type_key}] {op} ({desc}) ✅ totalCount={total}")
        if items:
            for k, v in items[0].items():
                logger.info(f"  {k:28} | {str(v)[:70]!r}")
    logger.info(f"\n성공 {len(ok)}개 / 실패 {len(ng)}개 {ng if ng else ''}")


def collect(type_key: str):
    op, desc = OPERATIONS[type_key]
    logger.info("=" * 50)
    logger.info(f"DUR 성분정보 수집: {desc} ({op})")
    logger.info("=" * 50)

    first = fetch_page(type_key, 1)
    calls = 1
    _, total_count = parse_body(first)
    total_pages = math.ceil(total_count / NUM_OF_ROWS)
    logger.info(f"전체 데이터: {total_count}건 / {total_pages}페이지")

    collected, failed = 0, []
    for page_no in range(1, total_pages + 1):
        s3_key = bronze_key(f"api/dur_ingredient/{type_key}", f"p{page_no:05d}.json")
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
                data = fetch_page(type_key, page_no)
                calls += 1
        except Exception as e:
            logger.error(f"  ❌ 페이지 {page_no} 실패: {e}")
            failed.append(page_no)
            continue

        items, _ = parse_body(data)
        if not items:
            logger.info(f"  빈 페이지 도달: p{page_no} → 종료")
            break

        upload_json({
            "source": "DURIrdntInfoService03_식약처",
            "operation": op,
            "license": "공공누리1유형",
            "type": type_key,          # 품목정보 bronze와 같은 키 구조
            "description": desc,
            "page_no": page_no,
            "total_count": total_count,
            "num_items": len(items),
            "items": items,
        }, s3_key)
        collected += len(items)
        logger.info(f"  📄 [{type_key}] {page_no}/{total_pages}: {len(items)}건 (누적 {collected}건)")
        time.sleep(0.3)

    logger.info(f"✅ [{desc}] 수집 완료: {collected}건 / API 호출 {calls}회")
    if failed:
        logger.warning(f"실패 페이지 {len(failed)}개: {failed[:20]} (재실행 시 자동 재시도)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=list(OPERATIONS), help="특정 유형만 수집")
    parser.add_argument("--probe", action="store_true", help="필드명 확인 (유형별 2건)")
    args = parser.parse_args()

    if not API_KEY:
        logger.error("PUBLIC_API_KEY 환경변수가 없습니다.")
        sys.exit(1)
    if args.probe:
        probe()
        return
    for t in ([args.type] if args.type else OPERATIONS):
        collect(t)


if __name__ == "__main__":
    main()
"""
[ETL 추가] 용법 필드 구조화 v2
입력: s3://bucket/silver/drugs/
출력: 각 약품 JSON에 structured_usage 필드 추가

변경 이력:
  v2: 추가 패턴
      - frequency: 범위형 횟수 (1일 1~2회, 하루에 2번 섭취)
      - dose: 소문자 ml, 앰플, 하이픈 범위
      - age_groups: 범위형 연령 (6세에서 12세, 7세~14세)
      - timing: 승차 전, 배변 후 등 특수 패턴
      - 최대 일일 용량 파싱 추가
"""
import re
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import list_keys, download_json, upload_json, silver_key


# ── 정규식 패턴 ──────────────────────────────────────────────

# 연령/대상
AGE_PATTERNS = [
    r"(만\s*\d+세\s*이상)",
    r"(\d+세\s*이상)",
    r"(\d+세\s*초과\s*[~～]\s*만?\s*\d+세\s*미만)",  # "7세 초과~15세 미만"
    r"(\d+세\s*(?:에서|[~～-])\s*\d+세)",            # "6세에서 12세", "7세~14세"
    r"(15세이상)",
    r"(성인)",
    r"(소아)",
    r"(영아|유아|어린이|노인|청소년)",
]

# 1회 용량
DOSE_PATTERNS = [
    r"1회\s*(\d+(?:[~～-]\d+)?)\s*(정|캡슐|캅셀|포|병|앰플|패치|매)",
    r"1회\s*(\d+(?:[~～-]\d+)?)\s*(mL|ml|ML|mg|g)",
    r"1회\s*(\d+(?:\.\d+)?)\s*(mL|ml|ML|mg|g)",
    r"(\d+(?:[~～-]\d+)?)\s*(mL|ml|ML|mg|g)\s*씩",
    r"(\d+(?:[~～-]\d+)?)\s*(정|캡슐|캅셀|앰플)\s*씩",
    r"1일\s*(\d+(?:[~～-]\d+)?)\s*(앰플|정|캡슐|포)",  # "1일 2-3앰플"
]

# 1일 횟수
FREQ_PATTERNS = [
    r"1일\s*(\d+[~～-]\d+)\s*회",           # "1일 1~2회" (범위형)
    r"1일\s*(\d+)\s*회",                     # "1일 3회"
    r"하루\s*(?:에\s*)?(\d+)\s*(?:번|회|차례)\s*(?:섭취|복용)?",  # "하루에 2번 섭취"
    r"씩\s*(\d+)\s*번\s*섭취",              # "씩 2번 섭취"
    r"1일\s*(?:복용\s*)?(?:회수|횟수)는?\s*(\d+)\s*회",           # "1일 복용회수는 2회"
    r"(\d+)일\s*1회",                        # "3일 1회"
    r"매일\s*1일\s*(\d+[~～-]\d+)\s*(?:앰플|정|캡슐|포)",        # "매일 1일 2-3앰플"
]

# 복용 시기
TIMING_PATTERNS = [
    (r"식후\s*(\d+)?\s*분?", "식후"),
    (r"식전\s*(\d+)?\s*분?", "식전"),
    (r"식간", "식간"),
    (r"취침\s*(?:시|전)", "취침전"),
    (r"공복", "공복"),
    (r"식사\s*(?:와\s*)?(?:함께|직후)", "식사와함께"),
    (r"배변\s*후", "배변후"),                          # 추가
    (r"승차\s*(\d+)분?\s*전", "승차전"),               # 추가
    (r"기상\s*(?:직)?후", "기상후"),                   # 추가
    (r"증상\s*(?:이\s*)?(?:시작|발현).*?즉시", "증상시즉시"),  # 추가
]

# 복용 간격
INTERVAL_PATTERNS = [
    r"(\d+)\s*시간\s*이상",
    r"매\s*(\d+)\s*시간",
    r"(\d+)\s*시간\s*간격",
]

# 최대 복용일
MAX_DAYS_PATTERNS = [
    r"(\d+)일\s*이상\s*복용하지",
    r"(\d+)일\s*초과하여\s*복용",
    r"최대\s*(\d+)일",
    r"(\d+)일\s*이상\s*사용하지",
    r"(\d+)일간",
]

# 최대 1일 용량 (추가)
MAX_DAILY_PATTERNS = [
    r"1일\s*최대\s*(\d+)\s*(정|캡슐|mg|mL|ml|포)",
    r"하루\s*최대\s*(\d+)\s*(정|캡슐|mg|mL|ml|포)",
]


def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def parse_age_groups(text: str) -> list:
    """연령/대상 파싱 (중복 제거)"""
    groups = []
    for pattern in AGE_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            val = m.strip() if isinstance(m, str) else m
            if not val:
                continue
            # "만 N세 이상"이 있으면 "N세 이상" 중복 스킵
            is_sub = any(
                val in existing and val != existing
                for existing in groups
            )
            if not is_sub and val not in groups:
                groups.append(val)
    return groups


def parse_dose(text: str) -> dict:
    """1회 용량 파싱"""
    for pattern in DOSE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            groups = m.groups()
            # 하이픈을 ~ 로 통일
            amount = groups[0].replace("-", "~") if groups else ""
            return {
                "amount": amount,
                "unit":   groups[1] if len(groups) > 1 else "",
                "raw":    m.group(0),
            }
    return {}


def parse_frequency(text: str) -> dict:
    """1일 횟수 파싱 (범위형 포함)"""
    for pattern in FREQ_PATTERNS:
        m = re.search(pattern, text)
        if m:
            raw_val = m.group(1)
            # 범위형 ("1~2", "2-3") 처리
            if re.search(r"[~～-]", raw_val):
                parts = re.split(r"[~～-]", raw_val)
                try:
                    return {
                        "times_per_day_min": int(parts[0]),
                        "times_per_day_max": int(parts[1]),
                        "times_per_day":     f"{parts[0]}~{parts[1]}",
                        "raw": m.group(0),
                    }
                except Exception:
                    pass
            else:
                try:
                    return {
                        "times_per_day": int(raw_val),
                        "raw": m.group(0),
                    }
                except Exception:
                    pass
    return {}


def parse_timing(text: str) -> list:
    """복용 시기 파싱"""
    timings = []
    for pattern, label in TIMING_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result = {"type": label}
            groups = m.groups()
            if groups and groups[0]:
                try:
                    result["minutes"] = int(groups[0])
                except Exception:
                    pass
            timings.append(result)
    return timings


def parse_interval(text: str) -> dict:
    """복용 간격 파싱"""
    for pattern in INTERVAL_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return {
                "min_hours": int(m.group(1)),
                "raw": m.group(0),
            }
    return {}


def parse_max_days(text: str) -> dict:
    """최대 복용일 파싱"""
    for pattern in MAX_DAYS_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return {
                "max_days": int(m.group(1)),
                "raw": m.group(0),
            }
    return {}


def parse_max_daily_dose(text: str) -> dict:
    """최대 1일 용량 파싱 (추가)"""
    for pattern in MAX_DAILY_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return {
                "amount": m.group(1),
                "unit":   m.group(2),
                "raw":    m.group(0),
            }
    return {}


def parse_usage(usage_text: str) -> dict:
    """용법 텍스트 전체 파싱"""
    if not usage_text:
        return {}

    text = clean_text(usage_text)

    result = {
        "age_groups":     parse_age_groups(text),
        "dose":           parse_dose(text),
        "frequency":      parse_frequency(text),
        "timing":         parse_timing(text),
        "interval":       parse_interval(text),
        "max_days":       parse_max_days(text),
        "max_daily_dose": parse_max_daily_dose(text),  # 추가
        "raw":            text,
    }

    has_data = any([
        result["age_groups"],
        result["dose"],
        result["frequency"],
        result["timing"],
        result["interval"],
        result["max_days"],
        result["max_daily_dose"],
    ])

    return result if has_data else {"raw": text}


# ── 테스트 케이스 (v2 추가분 포함) ──────────────────────────

TEST_CASES = [
    # 기존
    "성인 및 12세 이상은 1회 1~2정, 1일 3회, 식후 30분에 복용합니다. 복용 간격은 4시간 이상으로 합니다.",
    "만 15세 이상 및 성인은 1회 1병(75 mL), 1일 3회 식후에 복용합니다.",
    "성인은 1회 2정을 취침 전에 복용합니다. 3일 이상 복용하지 마시오.",
    # v2 신규
    "15세이상 청소년 및 성인은 1회 2정씩, 1일 1~2회, 7세~14세 소아는 1회 1정씩, 1일 1~2회 복용합니다.",
    "6세에서 12세 소아환자: 하루에 5mg씩 2번 섭취(하루에 0.5정을 2번 섭취)",
    "성인 1회 2정씩 1일 최대 8정까지 복용하며, 복용간격은 4시간 이상으로 한다.",
    "1일 1~2회 환부에 부착한다.",
    "1. 성인 : 2-3주동안 매일 1일 2-3앰플 복용",
    "멀미의 예방에는 승차 30분 전에 1회 2정을 복용한다.",
    "1일 2회, 가능한 한 배변 후 적량을 환부에 바른다.",
    "성인 1회 30ml 1일 3회 식후 30분에 복용한다.",
]


def run_test():
    import json
    logger.info("=== 용법 파싱 테스트 (v2) ===")
    for i, text in enumerate(TEST_CASES, 1):
        result = parse_usage(text)
        logger.info(f"\n[케이스 {i}] {text[:60]}...")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))


# ── Silver 전체 처리 ──────────────────────────────────────────

def process_silver_drugs():
    logger.info("=" * 50)
    logger.info("[ETL 추가] 용법 필드 구조화 v2 시작")
    logger.info("=" * 50)

    prefixes = [
        "silver/drugs/public_drug/",
        "silver/drugs/health_kr_drug/",
    ]

    total_parsed = 0
    total_failed = 0

    for prefix in prefixes:
        keys = [k for k in list_keys(prefix) if k.endswith(".json")]
        logger.info(f"\n▶ {prefix}: {len(keys)}개 파일")

        for key in keys:
            try:
                data = download_json(key)
                items = data.get("items", [])
                updated = False

                for item in items:
                    usage_text = (
                        item.get("metadata", {}).get("usage", "") or
                        item.get("usage", "") or ""
                    )

                    if not usage_text:
                        text = item.get("text", "")
                        m = re.search(r"용법:\s*(.+?)(?:\s*\||\s*$)", text)
                        if m:
                            usage_text = m.group(1)

                    if usage_text:
                        parsed = parse_usage(usage_text)
                        if parsed:
                            if "metadata" in item:
                                item["metadata"]["structured_usage"] = parsed
                            else:
                                item["structured_usage"] = parsed
                            total_parsed += 1
                            updated = True
                    else:
                        total_failed += 1

                if updated:
                    upload_json(data, key)

            except Exception as e:
                logger.error(f"  ❌ {key}: {str(e)[:80]}")

    logger.info("=" * 50)
    logger.info(f"파싱 성공: {total_parsed}건")
    logger.info(f"용법 없음: {total_failed}건")
    logger.info("=" * 50)


def main():
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "test":
        run_test()
    else:
        run_test()
        process_silver_drugs()


if __name__ == "__main__":
    main()
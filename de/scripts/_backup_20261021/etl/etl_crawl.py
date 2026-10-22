"""
[ETL 3] 크롤링 데이터 Bronze → Silver
입력:
  - s3://bucket/bronze/crawl/health_portal/
  - s3://bucket/bronze/crawl/health_kr/
출력:
  - s3://bucket/silver/symptoms/
  - s3://bucket/silver/drugs/crawl_drug/

수정 이력:
  v2: 필터링 강화
      - content 길이 최소값 적용 (100자 이하 제외)
      - URL에 gnrlzHealthInfoDtl 패턴 없으면 제외
      - 중복 이름 제거
      - 파일 크기 기반 필터 (실질 내용 없는 파일 제외)
  v4: 위험 신호(redflag) 재분류
      - 문제: 크롤러의 redflag_content에 [예방]·[요약문] 같은 일반 설명이 통째로 들어가 있었음
              (예: 두통의 위험 신호 문서 = 편두통 예방법) + 그 구간은 일반 문서(info)에서 빠져 있었음
      - 수정: content + redflag_content의 모든 구간을 일반 문서(info)에 넣고,
              위험 신호 문서(redflag)는 "언제 병원/응급실에 가야 하는지" 문장만 골라서 새로 만듦
      - 실행: python etl_crawl.py --dry-run --sample 두통 --show-lost 15   (업로드 없이 결과 미리보기)
      - "다음과 같은 증상이 있으면…" 뒤의 증상 목록도 함께 포함
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

SOURCE_META = {
    "health_portal": {
        "source_name": "국가건강정보포털",
        "source_url": "https://health.kdca.go.kr",
        "license": "공공누리",
        "provider": "질병관리청",
    },
    "health_kr_disease": {
        "source_name": "약학정보원 질병정보",
        "source_url": "https://www.health.kr/researchInfo/disease.asp",
        "license": "비상업적 참조 목적",
        "provider": "약학정보원",
    },
    "health_kr_drug": {
        "source_name": "약학정보원 의약품정보",
        "source_url": "https://www.health.kr/searchDrug/search_detail.asp",
        "license": "비상업적 참조 목적",
        "provider": "약학정보원",
    },
}

# 유효한 질환 상세 URL 패턴
VALID_PORTAL_URL_KEYWORDS = ["gnrlzHealthInfoDtl"]

# 제외할 이름 패턴
SKIP_NAMES = {
    "FAQ", "뉴스레터", "카드뉴스", "이용안내", "소개",
    "사이트맵", "공지사항", "저작권", "개인정보", "관련링크",
    "이미지자료실", "동영상자료실", "OpenAPI", "건강담기",
    "검진기관", "병원약국", "노인 건강정보", "청소년 건강정보",
    "생애주기별 건강정보", "자료실", "건강소식",
}

MIN_CONTENT_LENGTH = 100  # 최소 내용 길이


def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def merge_content(content: dict) -> str:
    if not content:
        return ""
    parts = []
    for section, text in content.items():
        cleaned = clean_text(text)
        if cleaned:
            parts.append(f"[{section}] {cleaned}")
    return " | ".join(parts)


# ── 위험 신호(redflag) 문장 추출 ─────────────────────────────
# "언제 병원/응급실에 가야 하는지"를 말하는 문장만 위험 신호로 봄
URGENT_RE = re.compile(r"(응급실|응급\s*의료|119|구급차|즉시|곧바로|지체\s*없이|서둘러|신속하게)")
ACTION_RE = re.compile(
    r"(병원|의료기관|응급실|진료실)(을|에|으로|를)?\s*(방문|찾|가|내원|이송)"
    r"|(진료|진찰)(를|을)?\s*(꼭\s*|반드시\s*|즉시\s*|바로\s*)?받"
    r"|의사(와|에게|의)\s*(상담|상의|진료|진찰|진단)"
    r"|(전문의|의료진|전문가)(와|에게|의)?\s*(상담|상의|진료|진찰|진단)"
    r"|(진료|진찰|응급\s*처치|응급\s*치료)(가|이)\s*(꼭\s*|반드시\s*|즉시\s*)?필요"
    r"|내원(해야|하여야|하십시오|하세요|이\s*필요)"
)
CONDITION_RE = re.compile(r"(경우|때|면\s|으면|다면|되면|있으면|없으면|지속|심해|악화|동반|갑자기|갑작스러운)")
# 구간 제목 자체가 위험 신호를 뜻하면 구간 전체를 위험 신호로 사용
REDFLAG_SECTION_RE = re.compile(r"(응급|위험\s*신호|경고\s*증상|병원을?\s*찾|진료가\s*필요|이럴\s*때)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*•\s*|\s+(?=\d+\)\s)")
# "다음과 같은 증상이 있으면 즉시 병원에…" 처럼 뒤에 목록이 오는 문장 → 뒤 문장들도 함께 가져옴
LEAD_IN_RE = re.compile(r"(다음과\s*같|아래와\s*같|다음의|다음\s*증상|경우는\s*\?|[:：]\s*$)")
LEAD_IN_FOLLOW_SENTS = 12    # 뒤따르는 문장 최대 개수 (목록 '1. … 2. …'은 번호마다 문장이 나뉨)
LEAD_IN_FOLLOW_CHARS = 900   # 뒤따르는 문장 최대 길이 합
NEXT_QUESTION_RE = re.compile(r"^Q[.\s]")   # FAQ의 다음 질문이 시작되면 거기서 멈춤
# 놓친 질환 점검용 (--show-lost): 이 단어가 있는데 위험 신호로 안 뽑힌 문장을 보여줌
WEAK_HINT_RE = re.compile(r"(병원|의사|응급|즉시|진료|진찰|119|전문의|의료기관)")
MIN_REDFLAG_LENGTH = 20
MAX_REDFLAG_SECTION = 800  # 위험 신호 제목 구간이라도 너무 길면 문장 단위로만 사용


def is_redflag_sentence(sent: str) -> bool:
    if URGENT_RE.search(sent) and ACTION_RE.search(sent):
        return True
    return bool(ACTION_RE.search(sent) and CONDITION_RE.search(sent))


def extract_redflag(sections: dict) -> str:
    """모든 구간에서 위험 신호 문장만 뽑아 '[구간] 문장 …' 형태로 반환"""
    parts, seen = [], set()
    for section, text in sections.items():
        cleaned = clean_text(text)
        if not cleaned:
            continue
        if REDFLAG_SECTION_RE.search(str(section)) and len(cleaned) <= MAX_REDFLAG_SECTION:
            picked = [cleaned]
        else:
            sents = [x.strip() for x in SENT_SPLIT_RE.split(cleaned) if x.strip()]
            picked, i = [], 0
            while i < len(sents):
                sent = sents[i]
                if 10 <= len(sent) <= 400 and is_redflag_sentence(sent):
                    picked.append(sent)
                    if LEAD_IN_RE.search(sent):        # 뒤에 나오는 증상 목록까지 포함
                        total, j = 0, i + 1
                        while (j < len(sents) and j <= i + LEAD_IN_FOLLOW_SENTS and total + len(sents[j]) <= LEAD_IN_FOLLOW_CHARS
                               and not NEXT_QUESTION_RE.match(sents[j])):
                            picked.append(sents[j]); total += len(sents[j]); j += 1
                        i = j
                        continue
                i += 1
        picked = [p for p in picked if p not in seen]
        seen.update(picked)
        if picked:
            parts.append(f"[{section}] " + " ".join(picked))
    return " | ".join(parts)


def all_sections(content: dict, redflag_content: dict) -> dict:
    """content + redflag_content를 합침 (같은 구간 이름이면 내용이 다를 때만 이어 붙임)"""
    merged = dict(content or {})
    for section, text in (redflag_content or {}).items():
        if section in merged:
            if clean_text(text) and clean_text(text) not in clean_text(merged[section]):
                merged[section] = f"{merged[section]} {text}"
        else:
            merged[section] = text
    return merged


def is_valid_portal_item(data: dict) -> bool:
    """국가건강정보포털 항목 유효성 검사"""
    # 오류 항목 제외
    if "error" in data:
        return False

    # URL 패턴 검사 (질환 상세 페이지만)
    url = data.get("url", "")
    if not any(kw in url for kw in VALID_PORTAL_URL_KEYWORDS):
        return False

    # 이름 필터
    name = clean_text(data.get("name", ""))
    if not name or name in SKIP_NAMES:
        return False

    # 내용 최소 길이
    content = data.get("content", {})
    merged = merge_content(content)
    if len(merged) < MIN_CONTENT_LENGTH:
        return False

    return True


# ── 국가건강정보포털 ────────────────────────────────────────
def weak_hints(sections: dict, limit: int = 4) -> list:
    """위험 신호로 안 뽑혔지만 병원/의사 같은 단어가 있는 문장 (기준 조정용)"""
    out = []
    for section, text in sections.items():
        for sent in SENT_SPLIT_RE.split(clean_text(text)):
            sent = sent.strip()
            if 10 <= len(sent) <= 300 and WEAK_HINT_RE.search(sent) and not is_redflag_sentence(sent):
                out.append(f"[{section}] {sent}")
                if len(out) >= limit:
                    return out
    return out


def etl_health_portal(dry_run: bool = False, sample: str = "", show_lost: int = 0) -> int:
    """
    v4: 위험 신호 재분류
    content + redflag_content 전체 → symptom_disease (has_redflag=False, 일반 설명)
    위 전체에서 "병원/응급실에 가야 하는 경우" 문장만 → symptom_disease (has_redflag=True)
    """
    logger.info("▶ [국가건강정보포털] 정제 시작 (v4)")
    keys = [k for k in list_keys("bronze/crawl/health_portal/") if k.endswith(".json")]
    logger.info(f"  전체 파일: {len(keys)}개")

    transformed = []
    skipped = 0
    seen_names = set()
    stats = {"old_redflag": 0, "new_redflag": 0, "only_old": [], "only_new": []}

    for key in keys:
        try:
            data = download_json(key)
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")
            skipped += 1
            continue

        # 오류 항목 제외
        if "error" in data:
            skipped += 1
            continue

        name = clean_text(data.get("name", ""))
        if not name or name in SKIP_NAMES:
            skipped += 1
            continue

        # 이름 기준 중복 제거
        if name in seen_names:
            skipped += 1
            continue
        seen_names.add(name)

        content = data.get("content", {}) or {}
        redflag_content = data.get("redflag_content", {}) or {}

        # 일반 설명: 모든 구간 (예전에 redflag로만 가던 [예방]·[요약문]도 여기로)
        sections = all_sections(content, redflag_content)
        merged = merge_content(sections)
        # 위험 신호: 모든 구간에서 해당 문장만 추출
        merged_redflag = extract_redflag(sections)

        if len(merged) < MIN_CONTENT_LENGTH and len(merged_redflag) < MIN_REDFLAG_LENGTH:
            skipped += 1
            continue

        old_has = len(merge_content(redflag_content)) >= MIN_CONTENT_LENGTH
        new_has = len(merged_redflag) >= MIN_REDFLAG_LENGTH
        stats["old_redflag"] += old_has
        stats["new_redflag"] += new_has
        if old_has and not new_has:
            stats["only_old"].append(name)
            if len(stats["only_old"]) <= show_lost:
                hints = weak_hints(sections)
                logger.info(f"  ── 놓친 질환: {name} (기존 구간 {list(redflag_content.keys())})")
                for h in hints or ["(병원/의사 관련 문장 없음 → 위험 신호가 없는 질환일 가능성)"]:
                    logger.info(f"     · {h[:200]}")
        if new_has and not old_has:
            stats["only_new"].append(name)

        base_meta = {
            "name": name,
            "system_code": data.get("system_code", ""),
            "system_name": data.get("system_name", ""),
            "item_url": data.get("url", ""),
            "brd_sid": data.get("brd_sid", ""),
            **SOURCE_META["health_portal"],
            "etl_at": datetime.now().isoformat(),
        }

        # 일반 내용 저장
        if len(merged) >= MIN_CONTENT_LENGTH:
            transformed.append({
                "doc_id": f"portal_{data.get('system_code','')}_{name}_info",
                "doc_type": "symptom_disease",
                "text": f"질환명: {name} | {merged}",
                "metadata": {**base_meta, "has_redflag": False},
            })

        # 위험 신호 저장 (문장 추출 결과)
        if new_has:
            transformed.append({
                "doc_id": f"portal_{data.get('system_code','')}_{name}_redflag",
                "doc_type": "symptom_disease",
                "text": f"질환명: {name} | [위험 신호] {merged_redflag}",
                "metadata": {**base_meta, "has_redflag": True, "redflag_method": "sentence_rule_v4"},
            })

        if sample and sample == name:
            logger.info(f"  ── 샘플: {name} ──")
            logger.info(f"  구간: {list(sections.keys())}")
            logger.info(f"  [기존 redflag 구간] {list(redflag_content.keys())}")
            logger.info(f"  [새 redflag] {merged_redflag[:800] or '(없음)'}")

    logger.info(f"  변환: {len(transformed)}건 / 스킵: {skipped}건")
    logger.info(f"  위험 신호 문서: 기존 {stats['old_redflag']}개 → 새 기준 {stats['new_redflag']}개")
    logger.info(f"  기존에만 있던 질환 {len(stats['only_old'])}개 예: {stats['only_old'][:15]}")
    logger.info(f"  새로 생긴 질환 {len(stats['only_new'])}개 예: {stats['only_new'][:15]}")

    if dry_run:
        logger.info("  (dry-run) 업로드 안 함")
        return len(transformed)

    saved = 0
    for i in range(0, len(transformed), 100):
        batch = transformed[i:i+100]
        s_key = silver_key("symptoms/health_portal", f"batch_{i//100:04d}.json")
        upload_json({"count": len(batch), "items": batch}, s_key)
        saved += len(batch)

    logger.info(f"  ✅ {saved}건 저장")
    return saved


# ── 약학정보원 질병정보 ─────────────────────────────────────
def etl_health_kr_disease() -> int:
    logger.info("▶ [약학정보원 질병정보] 정제 시작")
    keys = [k for k in list_keys("bronze/crawl/health_kr/")
            if "disease_" in k and k.endswith(".json")]
    logger.info(f"  전체 파일: {len(keys)}개")

    transformed, skipped = [], 0

    for key in keys:
        try:
            data = download_json(key)
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")
            skipped += 1
            continue

        if not data or "error" in data:
            skipped += 1
            continue

        content = data.get("content", {})
        basic_info = data.get("basic_info", {})
        merged = merge_content(content)

        # 최소 내용 길이 필터
        if len(merged) < MIN_CONTENT_LENGTH and not basic_info:
            skipped += 1
            continue

        name = clean_text(data.get("name", ""))
        text = f"질환명: {name}"
        korean_name = basic_info.get("질환명 (한글)", "")
        if korean_name:
            text += f" ({clean_text(korean_name)})"
        if merged:
            text += f" | {merged}"

        transformed.append({
            "doc_id": f"disease_{data.get('idx', '')}",
            "doc_type": "symptom_disease",
            "text": text,
            "metadata": {
                "name": name,
                "idx": data.get("idx"),
                "item_url": data.get("url", ""),
                "basic_info": {k: clean_text(v) for k, v in basic_info.items() if v},
                **SOURCE_META["health_kr_disease"],
                "etl_at": datetime.now().isoformat(),
            }
        })

    logger.info(f"  변환: {len(transformed)}건 / 스킵: {skipped}건")

    saved = 0
    for i in range(0, len(transformed), 100):
        batch = transformed[i:i+100]
        s_key = silver_key("symptoms/health_kr_disease", f"batch_{i//100:04d}.json")
        upload_json({"count": len(batch), "items": batch}, s_key)
        saved += len(batch)

    logger.info(f"  ✅ {saved}건 저장")
    return saved


# ── 약학정보원 일반의약품 ───────────────────────────────────
def etl_health_kr_drug() -> int:
    logger.info("▶ [약학정보원 일반의약품] 정제 시작")
    keys = [k for k in list_keys("bronze/crawl/health_kr/")
            if "drug_" in k and k.endswith(".json")]
    logger.info(f"  전체 파일: {len(keys)}개")

    transformed, skipped = [], 0
    seen_names = set()

    for key in keys:
        try:
            data = download_json(key)
        except Exception as e:
            logger.warning(f"  스킵 [{key}]: {str(e)[:60]}")
            skipped += 1
            continue

        if not data:
            skipped += 1
            continue

        content = data.get("content", {})
        name = clean_text(data.get("name", ""))

        if not name or not content:
            skipped += 1
            continue

        if name in seen_names:
            skipped += 1
            continue
        seen_names.add(name)

        parts = [f"약품명: {name}"]
        field_labels = {
            "효능효과": "효능",
            "용법용량": "용법",
            "주의사항": "주의사항",
            "부작용": "부작용",
            "성분함량": "성분",
            "보관방법": "보관",
        }
        for field, label in field_labels.items():
            val = clean_text(content.get(field, ""))
            if val:
                parts.append(f"{label}: {val}")

        text = " | ".join(parts)

        # 최소 내용 길이 필터
        if len(text) < MIN_CONTENT_LENGTH:
            skipped += 1
            continue

        transformed.append({
            "doc_id": f"drug_kr_{name}",
            "doc_type": "drug_info",
            "text": text,
            "metadata": {
                "name": name,
                "item_url": data.get("url", ""),
                "images": data.get("images", []),
                **SOURCE_META["health_kr_drug"],
                "etl_at": datetime.now().isoformat(),
            }
        })

    logger.info(f"  변환: {len(transformed)}건 / 스킵: {skipped}건")

    saved = 0
    for i in range(0, len(transformed), 200):
        batch = transformed[i:i+200]
        s_key = silver_key("drugs/health_kr_drug", f"batch_{i//200:04d}.json")
        upload_json({"count": len(batch), "items": batch}, s_key)
        saved += len(batch)

    logger.info(f"  ✅ {saved}건 저장")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="S3에 올리지 않고 결과만 출력")
    ap.add_argument("--sample", default="", help="이 질환명의 위험 신호 추출 결과를 출력 (예: 두통)")
    ap.add_argument("--show-lost", type=int, default=0, help="위험 신호가 없어진 질환 N개의 후보 문장 출력")
    args = ap.parse_args()

    logger.info("=" * 50)
    logger.info("[ETL 3] 크롤링 데이터 정제 시작 (v4 - 위험 신호 재분류)")
    logger.info("=" * 50)

    total = 0
    total += etl_health_portal(dry_run=args.dry_run, sample=args.sample, show_lost=args.show_lost)
    # total += etl_health_kr_disease()
    # total += etl_health_kr_drug()

    logger.info("=" * 50)
    logger.info(f"전체 완료: {total}건")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
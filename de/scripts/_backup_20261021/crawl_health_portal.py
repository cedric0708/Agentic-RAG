"""
[수집 3] 국가건강정보포털 - 질환별 증상·진료 정보
로컬 실행 전용
라이선스: 공공누리 (질병관리청 산하)
저장경로: s3://bucket/bronze/crawl/health_portal/YYYY-MM-DD/

확인된 구조:
  목록: searchFilter('bdySystem', 'CODE') → AJAX 로드
  상세: gnrlzHealthInfoView.do?brdSid=XXXX
  본문: div id="contentsDivN" > h3 + p 구조
"""
import sys, time, re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup
from loguru import logger
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_s3 import bronze_key, upload_json, key_exists

BASE_URL = "https://health.kdca.go.kr"
MAIN_URL = f"{BASE_URL}/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoMain.do"
VIEW_URL = f"{BASE_URL}/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do"

# 제외할 이름 패턴
SKIP_NAMES = {
    "건강문제", "치료방법", "검사방법", "생활습관 관리",
    "전체", "본문으로 바로가기", "주메뉴 바로가기",
    "건강담기", "목록", "이전", "다음", "인쇄",
    "청소년", "노인", "생애주기별 건강정보",
}

BDY_SYSTEMS = {
    "NE": "뇌신경",
    "JU": "정신건강",
    "KO": "귀코목",
    "KU": "구강",
    "BB": "뼈근육",
    "PB": "피부",
    "NB": "내분비",
    "HH": "호흡기",
    "SO": "순환기",
    "SW": "소화기",
    "MH": "면역",
    "SA": "비뇨기",
    "SS": "생식기",
}

# Red flag 키워드
REDFLAG_KEYWORDS = [
    "즉시 병원", "응급", "119", "즉각 중지",
    "의사와 상담", "전문의", "심각한", "위험",
    "호흡 곤란", "흉통", "의식 잃", "경련",
    "마비", "출혈", "고열", "48시간 이상",
    "증상이 악화", "병원을 방문",
]


def init_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def get_disease_list(driver: webdriver.Chrome, system_code: str) -> list:
    """신체계통 버튼(id=bdySystemHH) 클릭 후 fn_goView 패턴으로 질환 목록 수집"""
    driver.get(MAIN_URL)
    time.sleep(2)

    # id="bdySystemHH" 형태로 버튼 클릭
    btn_id = f"bdySystem{system_code.upper()}"
    try:
        btn = driver.find_element(By.ID, btn_id)
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(3)
    except Exception as e:
        logger.warning(f"  [{system_code}] 버튼({btn_id}) 클릭 실패: {e}")
        return []

    soup = BeautifulSoup(driver.page_source, "lxml")
    diseases = []
    seen_brd_sids = set()

    # fn_goView('brdSid', '질환명') 패턴 추출
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        match = re.search(r"fn_goView\('(\d+)'\s*,\s*'([^']+)'\)", href)
        if not match:
            continue

        brd_sid = match.group(1)
        name = match.group(2).strip()

        if not name or name in SKIP_NAMES:
            continue
        if brd_sid in seen_brd_sids:
            continue

        seen_brd_sids.add(brd_sid)
        diseases.append({
            "name": name,
            "url": f"{VIEW_URL}?brdSid={brd_sid}",
            "brd_sid": brd_sid,
        })

    logger.info(f"  [{system_code}] 질환: {len(diseases)}개")
    return diseases


def get_disease_detail(driver: webdriver.Chrome, disease: dict) -> dict:
    """질환 상세 페이지 수집 - fn_goView JS 실행 방식"""
    try:
        driver.set_page_load_timeout(15)

        # 직접 URL 접근 시 메인으로 리다이렉트됨
        # fn_goView JS 실행으로 접근
        brd_sid = disease.get("brd_sid", "")
        name = disease.get("name", "")
        driver.execute_script(f"fn_goView(\'{brd_sid}\',\'{name}\');")
        time.sleep(2)

        # 상세 페이지 진입 확인
        if "gnrlzHealthInfoView" not in driver.current_url:
            driver.get(MAIN_URL)
            time.sleep(1)
            driver.execute_script(f"fn_goView(\'{brd_sid}\',\'{name}\');")
            time.sleep(2)

        if "gnrlzHealthInfoView" not in driver.current_url:
            return {"error": f"상세 페이지 접근 실패"}

        soup = BeautifulSoup(driver.page_source, "lxml")

        content = {}
        redflag_content = {}

        # contentsDivN 구조 파싱
        # div id가 "contentsDiv" + 숫자 패턴
        for div in soup.find_all("div", id=re.compile(r"contentsDiv\d+")):
            h3 = div.find("h3")
            if not h3:
                continue
            section_title = h3.get_text(strip=True)
            if not section_title:
                continue

            # h3 제거 후 텍스트 추출
            h3.decompose()
            section_text = div.get_text(separator=" ", strip=True)[:2000]
            if not section_text:
                continue

            # Red flag 여부 판단
            if any(kw in section_text for kw in REDFLAG_KEYWORDS):
                redflag_content[section_title] = section_text
            else:
                content[section_title] = section_text

        # fallback: class="contents-Div" 패턴
        if not content and not redflag_content:
            for div in soup.find_all("div", class_="contents-Div"):
                h3 = div.find("h3")
                if not h3:
                    continue
                section_title = h3.get_text(strip=True)
                h3.decompose()
                section_text = div.get_text(separator=" ", strip=True)[:2000]
                if section_text:
                    if any(kw in section_text for kw in REDFLAG_KEYWORDS):
                        redflag_content[section_title] = section_text
                    else:
                        content[section_title] = section_text

        if not content and not redflag_content:
            return {"error": "내용 없음"}

        return {
            "name": disease["name"],
            "url": disease["url"],
            "brd_sid": disease.get("brd_sid", ""),
            "content": content,
            "redflag_content": redflag_content,
            "has_redflag": len(redflag_content) > 0,
            "crawled_at": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"error": str(e).split("\n")[0][:100]}


def collect_system(driver: webdriver.Chrome, system_code: str, system_name: str) -> tuple:
    logger.info(f"\n▶ [{system_name}({system_code})] 수집 시작")

    diseases = get_disease_list(driver, system_code)
    if not diseases:
        logger.warning(f"  질환 없음")
        return 0, 0

    ok, fail = 0, 0
    for i, disease in enumerate(diseases, 1):
        s3_key = bronze_key(
            "crawl/health_portal",
            f"{system_code}_{disease['name'][:20].replace(' ','_')}.json"
        )

        if key_exists(s3_key):
            logger.debug(f"  스킵: {disease['name']}")
            continue

        # fn_goView 실행을 위해 목록 페이지로 이동 후 신체계통 필터 적용
        driver.get(MAIN_URL)
        time.sleep(1)
        try:
            btn = driver.find_element(By.ID, f"bdySystem{system_code.upper()}")
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(2)
        except Exception:
            pass

        detail = get_disease_detail(driver, disease)

        if "error" in detail:
            fail += 1
            logger.debug(f"  [{i}/{len(diseases)}] {disease['name']} ❌ {detail['error']}")
        else:
            detail["system_code"] = system_code
            detail["system_name"] = system_name
            upload_json(detail, s3_key)
            ok += 1
            redflag = "🚨" if detail.get("has_redflag") else "✅"
            logger.info(f"  [{i}/{len(diseases)}] {disease['name']} {redflag}")

        time.sleep(1)

    return ok, fail


def main():
    logger.info("=" * 50)
    logger.info("국가건강정보포털 크롤링 시작 (v3)")
    logger.info("확인된 구조: gnrlzHealthInfoView.do + contentsDivN")
    logger.info("=" * 50)

    driver = init_driver()
    total_ok, total_fail = 0, 0

    try:
        for code, name in BDY_SYSTEMS.items():
            ok, fail = collect_system(driver, code, name)
            total_ok += ok
            total_fail += fail
            logger.info(f"  [{name}] 성공: {ok} / 실패: {fail}")
            time.sleep(1)
    finally:
        driver.quit()

    logger.info("=" * 50)
    logger.info(f"완료 → 성공: {total_ok} / 실패: {total_fail}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
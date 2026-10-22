"""
[수집 3] 국가건강정보포털 - 질환별 증상·진료 정보
로컬 실행 전용 (Chrome + Selenium)
라이선스: 공공누리 (질병관리청)
저장경로: s3://bucket/bronze/crawl/health_portal/{RUN_DATE 또는 오늘}/

확인된 구조:
  목록: 신체계통 버튼(id=bdySystemXX) 클릭 → AJAX 목록, 여러 페이지 (페이지 번호 링크 href="#")
  상세: fn_goView('brdSid', '질환명') → gnrlzHealthInfoView.do
  본문: div id="contentsDivN" > h3 + 본문

v4 (2026-10-21) 변경
  1. 목록 페이지 넘김 추가 → 예전에는 신체계통마다 첫 페이지만 수집 (전체 214개)
  2. 신체계통 '눈(NU)' 추가
  3. 위험 신호 구간 분류 제거 → 모든 구간을 content에 저장
     (위험 신호 문장은 etl_crawl.py v4가 문장 단위로 추출. redflag_content는 호환용 빈 값)
  4. 구간 본문 2,000자 자르기 제거 (최대 20,000자)
  5. 심뇌혈관 특별 페이지 수집: 심근경색 / 뇌졸중 / 뇌졸중 이후의 삶 / 응급상황 / 심폐소생술
     (일반 질환 목록에 없어서 예전 데이터에 빠져 있던 응급 질환)
  6. 저장 파일명을 brdSid 기준으로 변경 (질환명 20자 자르기로 생기던 이름 충돌 방지)

사용법:
  python crawl_health_portal.py --list-only              # 목록만 확인 (신체계통별 질환 수, 페이지 수) ★ 먼저 실행
  python crawl_health_portal.py --list-only --systems SO # 순환기만
  python crawl_health_portal.py --special-only           # 심뇌혈관 특별 페이지만
  python crawl_health_portal.py                          # 전체 수집 (같은 RUN_DATE로 다시 실행하면 이어서)
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv(Path(__file__).resolve().parent.parent.parent / "config" / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import bronze_key, upload_json, key_exists  # noqa: E402

BASE_URL = "https://health.kdca.go.kr"
MAIN_URL = f"{BASE_URL}/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoMain.do"
VIEW_URL = f"{BASE_URL}/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do"

SKIP_NAMES = {
    "건강문제", "치료방법", "검사방법", "생활습관 관리",
    "전체", "본문으로 바로가기", "주메뉴 바로가기",
    "건강담기", "목록", "이전", "다음", "인쇄",
    "청소년", "노인", "생애주기별 건강정보",
}

BDY_SYSTEMS = {
    "NE": "뇌신경",
    "JU": "정신건강",
    "NU": "눈",          # v4 추가
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
SYSTEM_NAMES = {**BDY_SYSTEMS, "ETC": "기타", "EM": "응급"}

# 일반 질환 목록에 없는 심뇌혈관·응급 페이지 (메뉴 링크로만 존재)
SPECIAL_PAGES = {
    "심근경색증":        ("SO", f"{BASE_URL}/healthinfo/biz/health/ccvdInfo/cvcdInfo/miInfoMain.do"),
    "뇌졸중":            ("NE", f"{BASE_URL}/healthinfo/biz/health/ccvdInfo/ccvcdInfo/cbvcacdInfoMain.do"),
    "뇌졸중 이후의 삶":   ("NE", f"{BASE_URL}/healthinfo/biz/health/ccvdInfo/ccvcdInfo/cbvcacdAfterMain.do"),
    "응급상황 대처":      ("EM", f"{BASE_URL}/healthinfo/biz/health/gnrlzHealthInfo/healthInfo/emgncySittnInfoMain.do"),
    "심폐소생술":         ("EM", f"{BASE_URL}/healthinfo/biz/health/ccvdInfo/cvcdInfo/cprInfoMain.do"),
}

MAX_SECTION_CHARS = 20000
MAX_LIST_PAGES = 60
GOVIEW_RE = re.compile(r"fn_goView\('(\d+)'\s*,\s*'([^']+)'\)")
TOTAL_RE = re.compile(r"총\s*([\d,]+)\s*건")


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


def open_system(driver: webdriver.Chrome, system_code: str) -> bool:
    """메인 페이지에서 신체계통 버튼 클릭 (ETC: 필터 없이 전체 목록)"""
    driver.get(MAIN_URL)
    time.sleep(2)
    if system_code == "ETC":
        return True
    try:
        btn = driver.find_element(By.ID, f"bdySystem{system_code.upper()}")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(3)
        return True
    except Exception as e:
        logger.warning(f"  [{system_code}] 버튼 클릭 실패: {str(e)[:80]}")
        return False


def parse_list(html: str) -> list:
    """현재 목록 화면의 fn_goView 링크 → [(brdSid, 질환명)]"""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        m = GOVIEW_RE.search(a.get("href", "") + " " + (a.get("onclick") or ""))
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def goto_page(driver: webdriver.Chrome, page: int) -> bool:
    """페이지 번호 링크 클릭. 번호가 안 보이면 '다음 묶음' 버튼을 누르고 다시 시도"""
    num_xpaths = [
        f"//*[contains(@class,'pag')]//a[normalize-space()='{page}']",
        f"//*[contains(@class,'pag')]//button[normalize-space()='{page}']",
        f"//*[contains(@class,'Pag')]//a[normalize-space()='{page}']",
    ]
    next_xpaths = [
        "//*[contains(@class,'pag')]//a[contains(@class,'next') and not(contains(@class,'end')) and not(contains(@class,'last'))]",
        "//*[contains(@class,'pag')]//a[normalize-space()='다음' or normalize-space()='>' or normalize-space()='›' or @title='다음']",
    ]
    for attempt in range(2):
        for xp in num_xpaths:
            els = driver.find_elements(By.XPATH, xp)
            if els:
                driver.execute_script("arguments[0].click();", els[0])
                return True
        if attempt == 0:   # 번호가 없으면 다음 묶음으로 넘긴 뒤 한 번 더
            moved = False
            for xp in next_xpaths:
                els = driver.find_elements(By.XPATH, xp)
                if els:
                    driver.execute_script("arguments[0].click();", els[0])
                    time.sleep(2)
                    moved = True
                    break
            if not moved:
                return False
    return False


def get_disease_list(driver: webdriver.Chrome, system_code: str) -> tuple:
    """신체계통의 모든 페이지를 넘기며 질환 목록 수집 → (질환 목록, 화면에 표시된 총 건수, 페이지 수)"""
    if not open_system(driver, system_code):
        return [], None, 0
    m = TOTAL_RE.search(BeautifulSoup(driver.page_source, "lxml").get_text(" "))
    shown_total = int(m.group(1).replace(",", "")) if m else None

    found, order, pages = {}, [], 0
    for page in range(1, MAX_LIST_PAGES + 1):
        if page > 1:
            if not goto_page(driver, page):
                break
            time.sleep(2)
        new = 0
        for sid, name in parse_list(driver.page_source):
            if sid not in found and name and name not in SKIP_NAMES:
                found[sid] = name
                order.append(sid)
                new += 1
        pages = page
        if new == 0:   # 페이지가 안 바뀌었거나 끝 → 중단
            break

    diseases = [{"brd_sid": sid, "name": found[sid], "url": f"{VIEW_URL}?brdSid={sid}"} for sid in order]
    gap = "" if shown_total in (None, len(diseases)) else f" ⚠️ 화면 총 {shown_total}건과 다름"
    logger.info(f"  [{system_code}] 질환 {len(diseases)}개 / {pages}페이지{gap}")
    return diseases, shown_total, pages


def get_all_list(driver: webdriver.Chrome) -> tuple:
    """신체계통 필터 없이 전체 목록 (여러 페이지). 신체계통이 지정되지 않은 건강정보가 많음"""
    return get_disease_list(driver, "ETC")


def parse_sections(soup: BeautifulSoup) -> dict:
    """contentsDivN(h3 + 본문) 구조 → {구간 제목: 본문}"""
    content = {}
    divs = soup.find_all("div", id=re.compile(r"contentsDiv\d+")) or soup.find_all("div", class_="contents-Div")
    for div in divs:
        h3 = div.find("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)
        h3.decompose()
        text = div.get_text(separator=" ", strip=True)[:MAX_SECTION_CHARS]
        if title and text:
            content[title] = f"{content[title]} {text}" if title in content else text
    return content


def parse_generic_sections(soup: BeautifulSoup) -> dict:
    """특별 페이지용: 본문 영역의 h2~h4 제목 기준으로 구간 나누기 (구조가 달라도 최대한 수집)"""
    root = (soup.find(id=re.compile(r"^(contents?|container|content-wrap|sub_content)$", re.I))
            or soup.find("main") or soup.body or soup)
    for t in root.find_all(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    content, title, buf = {}, "개요", []
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "dd", "td"]):
        if el.name in ("h2", "h3", "h4"):
            if buf:
                content[title] = (content.get(title, "") + " " + " ".join(buf)).strip()[:MAX_SECTION_CHARS]
            title, buf = el.get_text(" ", strip=True) or title, []
        else:
            txt = el.get_text(" ", strip=True)
            if txt and (not buf or txt != buf[-1]):
                buf.append(txt)
    if buf:
        content[title] = (content.get(title, "") + " " + " ".join(buf)).strip()[:MAX_SECTION_CHARS]
    nav = {"건강정보", "생애주기별 건강정보", "심뇌혈관질환정보", "뇌혈관질환", "심혈관질환", "정신건강정보"}
    return {k: v for k, v in content.items() if len(v) >= 20 and k.strip() not in nav}   # 메뉴 제목 제외


def get_disease_detail(driver: webdriver.Chrome, disease: dict, system_code: str) -> dict:
    """질환 상세 페이지 (fn_goView 실행 방식, 직접 URL 접근은 메인으로 돌아감)"""
    try:
        driver.set_page_load_timeout(20)
        sid, name = disease["brd_sid"], disease["name"].replace("'", "\\'")
        for _ in range(2):
            open_system(driver, system_code)
            driver.execute_script(f"fn_goView('{sid}','{name}');")
            time.sleep(2)
            if "gnrlzHealthInfoView" in driver.current_url:
                break
        else:
            return {"error": "상세 페이지 접근 실패"}

        content = parse_sections(BeautifulSoup(driver.page_source, "lxml"))
        if not content:
            return {"error": "내용 없음"}
        return {
            "name": disease["name"],
            "url": disease["url"],
            "brd_sid": sid,
            "content": content,
            "redflag_content": {},     # v4: 위험 신호는 etl_crawl.py가 문장 단위로 추출
            "has_redflag": False,
            "crawler_version": "v4",
            "crawled_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e).split("\n")[0][:100]}


def collect_system(driver: webdriver.Chrome, code: str, name: str, list_only: bool,
                   diseases: list | None = None, shown_total=None, pages: int = 0) -> dict:
    logger.info(f"\n▶ [{name}({code})] 시작")
    if diseases is None:
        diseases, shown_total, pages = get_disease_list(driver, code)
    stat = {"listed": len(diseases), "shown_total": shown_total, "pages": pages, "ok": 0, "skip": 0, "fail": 0}
    if list_only or not diseases:
        return stat

    for i, d in enumerate(diseases, 1):
        s3_key = bronze_key("crawl/health_portal", f"{code}_{d['brd_sid']}.json")
        if key_exists(s3_key):
            stat["skip"] += 1
            continue
        detail = get_disease_detail(driver, d, code)
        if "error" in detail:
            stat["fail"] += 1
            logger.warning(f"  [{i}/{len(diseases)}] {d['name']} ❌ {detail['error']}")
        else:
            detail.update(system_code=code, system_name=name)
            upload_json(detail, s3_key)
            stat["ok"] += 1
            logger.info(f"  [{i}/{len(diseases)}] {d['name']} ✅ 구간 {len(detail['content'])}개")
        time.sleep(1)
    return stat


def collect_special(driver: webdriver.Chrome, list_only: bool) -> dict:
    logger.info("\n▶ [심뇌혈관·응급 특별 페이지] 시작")
    stat = {"ok": 0, "skip": 0, "fail": 0}
    for i, (name, (code, url)) in enumerate(SPECIAL_PAGES.items(), 1):
        key_id = re.sub(r"\W+", "_", url.rsplit("/", 1)[-1].replace(".do", ""))
        s3_key = bronze_key("crawl/health_portal", f"{code}_special_{key_id}.json")
        if list_only:
            logger.info(f"  {name}: {url}")
            continue
        if key_exists(s3_key):
            stat["skip"] += 1
            continue
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "lxml")
            content = parse_sections(soup) or parse_generic_sections(soup)
        except Exception as e:
            content, err = {}, str(e)[:80]
        else:
            err = "내용 없음"
        if not content:
            stat["fail"] += 1
            logger.warning(f"  [{i}] {name} ❌ {err}")
            continue
        upload_json({
            "name": name, "url": url, "brd_sid": f"special_{key_id}",
            "content": content, "redflag_content": {}, "has_redflag": False,
            "system_code": code, "system_name": {"SO": "순환기", "NE": "뇌신경", "EM": "응급"}[code],
            "page_type": "special", "crawler_version": "v4", "crawled_at": datetime.now().isoformat(),
        }, s3_key)
        stat["ok"] += 1
        logger.info(f"  [{i}] {name} ✅ 구간 {len(content)}개: {list(content)[:6]}")
        time.sleep(1)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true", help="목록(질환 수·페이지 수)만 확인, 저장 안 함")
    ap.add_argument("--systems", default="", help="쉼표로 구분한 신체계통 코드만 (예: SO,NE)")
    ap.add_argument("--special-only", action="store_true", help="심뇌혈관·응급 특별 페이지만")
    ap.add_argument("--no-special", action="store_true", help="특별 페이지 제외")
    args = ap.parse_args()

    systems = {c: n for c, n in BDY_SYSTEMS.items()
               if not args.systems or c in {s.strip().upper() for s in args.systems.split(",")}}
    logger.info("=" * 50)
    logger.info(f"국가건강정보포털 크롤링 v4 {'(목록만)' if args.list_only else ''}")
    logger.info(f"저장 폴더: {bronze_key('crawl/health_portal', '')}")
    logger.info("=" * 50)

    driver = init_driver()
    stats = {}
    try:
        if not args.special_only:
            in_system = set()
            for code, name in systems.items():
                if args.list_only:
                    diseases, total, pages = get_disease_list(driver, code)
                    in_system.update(d["brd_sid"] for d in diseases)
                    stats[f"{name}({code})"] = {"listed": len(diseases), "shown_total": total, "pages": pages}
                else:
                    diseases, total, pages = get_disease_list(driver, code)
                    in_system.update(d["brd_sid"] for d in diseases)
                    stats[f"{name}({code})"] = collect_system(driver, code, name, False, diseases, total, pages)
            if not args.systems:
                # 신체계통 필터에 안 걸리는 나머지 (전체 목록 - 신체계통 목록) → system_code ETC
                logger.info("\n▶ [전체 목록] 신체계통 미지정 항목 확인")
                all_list, all_total, all_pages = get_all_list(driver)
                rest = [d for d in all_list if d["brd_sid"] not in in_system]
                logger.info(f"  전체 {len(all_list)}개 (화면 총 {all_total}건, {all_pages}페이지) → 신체계통 미지정 {len(rest)}개")
                stats["기타(ETC)"] = (collect_system(driver, "ETC", "기타", args.list_only, rest, all_total, all_pages)
                                     if not args.list_only else
                                     {"listed": len(rest), "all_listed": len(all_list), "shown_total": all_total, "pages": all_pages})
        if not args.no_special and not args.systems:
            stats["특별 페이지"] = collect_special(driver, args.list_only)
    finally:
        driver.quit()

    logger.info("=" * 50)
    total_listed = sum(s.get("listed", 0) for s in stats.values())
    for k, s in stats.items():
        logger.info(f"  {k:14} {s}")
    logger.info(f"  목록 합계 {total_listed}개 (예전 v3 수집: 214개, 신체계통 간 중복 포함)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

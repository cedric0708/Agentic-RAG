"""
[GRAPH 1] Silver → Neo4j Knowledge Graph 적재

입력 (각 prefix의 최신 날짜 폴더만 사용)
  silver/dur_ingredient/{type}/        DUR 성분정보 (금기·주의 엣지의 기준 데이터)
  silver/dur_ingredient/code_map/      원료코드(M/I) → DUR 성분코드(D) 공식 매핑
  silver/drugs/drug_permit/            제품 마스터 (허가정보)
  silver/drug_ingredients/             제품-주성분 (허가정보, M코드)
  silver/dur_interactions/{임부·연령·노인}/  DUR 품목정보 (제품 단위 지정)

그래프 스키마
  (:Drug {item_seq})-[:CONTAINS]->(:Ingredient {code})
  (:Ingredient)-[:CONTRAINDICATED_WITH]-(:Ingredient)        성분 쌍당 엣지 1개, code 사전순 lo→hi로 저장 (조회 시 방향 무시)
  (:Ingredient)-[:RESTRICTED_FOR]->(:PatientGroup {code})     PREGNANT / PEDIATRIC / ELDERLY
  (:Ingredient)-[:BELONGS_TO]->(:EffectGroup {name})          효능군중복 검사용
  (:Drug)-[:DUR_FLAGGED {ingr_codes}]->(:PatientGroup)        DUR 품목정보가 이 '제품'을 임부·연령·노인 대상으로 지정
                                                               (서방정 등 특정 제형에만 해당하는 성분 금기의 제품별 판정 근거)
  Ingredient 속성: dose_limits(1일 최대투여량), term_limits(최대투여기간)

Ingredient.code 규칙
  DUR 성분코드가 있으면 D코드 (예: D000762)
  DUR에 없는 성분은 'M:' + 원료코드 (금기 대상은 아니지만 동일 성분 중복 검사에 필요)

사용법
  python load_graph.py            # 증분 적재 (MERGE)
  python load_graph.py --reset    # 그래프 전체 삭제 후 재적재
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env", override=True)
from utils_s3 import list_keys, download_json

from neo4j import GraphDatabase

BATCH = 2000
DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/")

DUR_SOURCE = "식약처 DUR 성분정보"
DUR_SOURCE_URL = "https://www.data.go.kr/data/15056780/openapi.do"

PATIENT_GROUPS = [
    {"code": "PREGNANT",  "name": "임부"},
    {"code": "PEDIATRIC", "name": "소아·청소년"},
    {"code": "ELDERLY",   "name": "노인(65세 이상)"},
]
TYPE_TO_GROUP = {
    "pregnancy_ban": "PREGNANT",
    "age_specific_ban": "PEDIATRIC",
    "elderly_caution": "ELDERLY",
}


# ─────────────────────────────────────
# S3 로드
# ─────────────────────────────────────
def latest_items(prefix: str) -> list[dict]:
    """prefix 아래 날짜 폴더 중 가장 최신 것만 읽기 (ETL 재실행으로 날짜 폴더가 여러 개여도 안전)"""
    keys = [k for k in list_keys(prefix) if k.endswith(".json")]
    dates = sorted({m.group(1) for k in keys if (m := DATE_RE.search(k[len(prefix) - 1:]))})
    if not dates:
        logger.warning(f"  {prefix}: 데이터 없음")
        return []
    latest = dates[-1]
    items = []
    for k in keys:
        if f"/{latest}/" in k:
            items.extend((download_json(k) or {}).get("items", []))
    logger.info(f"  {prefix} ({latest}): {len(items)}건")
    return items


def meta(x: dict) -> dict:
    return x.get("metadata", x)


# ─────────────────────────────────────
# 그래프 데이터 구성
# ─────────────────────────────────────
class GraphData:
    def __init__(self):
        self.ingredients: dict[str, dict] = {}
        self.drugs: dict[str, dict] = {}
        self.contains: dict[tuple, dict] = {}
        self.pairs: dict[tuple, dict] = {}
        self.restrictions: dict[tuple, dict] = {}
        self.effects: dict[tuple, set] = defaultdict(set)
        self.dose_limits: dict[str, set] = defaultdict(set)
        self.term_limits: dict[str, set] = defaultdict(set)
        self.m2d: dict[str, set] = defaultdict(set)
        self.product_flags: dict[tuple, set] = defaultdict(set)   # (item_seq, group) → {성분코드}
        self.stats = defaultdict(int)

    def add_ingredient(self, code: str, kor: str = "", eng: str = "", in_dur: bool = False):
        e = self.ingredients.setdefault(code, {"code": code, "name_kor": "", "name_eng": "",
                                               "aliases": set(), "in_dur": False})
        if kor and not e["name_kor"]:
            e["name_kor"] = kor
        if eng and not e["name_eng"]:
            e["name_eng"] = eng
        e["aliases"].update(x for x in (kor, eng) if x)
        e["in_dur"] = e["in_dur"] or in_dur

    # ── DUR 성분정보 ──────────────────────
    def build_dur(self, by_type: dict[str, list[dict]]):
        for row in by_type.get("combination_ban", []):
            m = meta(row)
            a, b = m.get("ingr_code_a"), m.get("ingr_code_b")
            if not (a and b):
                self.stats["병용금기_코드없음_스킵"] += 1
                continue
            self.add_ingredient(a, m.get("ingredient_a_kor", ""), m.get("ingredient_a_eng", ""), True)
            self.add_ingredient(b, m.get("ingredient_b_kor", ""), m.get("ingredient_b_eng", ""), True)
            lo, hi = sorted([a, b])
            e = self.pairs.setdefault((lo, hi), {"reasons": {}, "conditions": set(), "variants": set(),
                                                 "dates": set(), "n_records": 0})
            date = m.get("notification_date", "")
            if m.get("reason"):
                e["reasons"][m["reason"]] = max(e["reasons"].get(m["reason"], ""), date)
            if m.get("remark"):
                e["conditions"].add(m["remark"])          # 예: '48시간 이내 병용금기'
            if m.get("mix_ingr_a") or m.get("mix_ingr_b"):
                # 복합제 형태로도 금기가 적용된다는 근거 (A/B 방향은 원본 기준)
                e["variants"].add(json.dumps({
                    "a": a, "mix_a": m.get("mix_ingr_a", ""),
                    "b": b, "mix_b": m.get("mix_ingr_b", ""),
                }, ensure_ascii=False, sort_keys=True))
            if date:
                e["dates"].add(date)
            e["n_records"] += 1

        for t, group in TYPE_TO_GROUP.items():
            for row in by_type.get(t, []):
                m = meta(row)
                code = m.get("ingr_code_a")
                if not code:
                    continue
                self.add_ingredient(code, m.get("ingredient_a_kor", ""), m.get("ingredient_a_eng", ""), True)
                e = self.restrictions.setdefault((code, group), {
                    "level": "caution", "grades": set(), "ages": set(), "age_unknown": False,
                    "age_bases": set(), "reasons": set(), "remarks": set(), "forms": set(), "dur_type": t,
                })
                if m.get("level") == "ban":
                    e["level"] = "ban"
                if m.get("pregnancy_grade") is not None:
                    e["grades"].add(m["pregnancy_grade"])
                if t == "age_specific_ban":
                    if m.get("max_age_months") is None:
                        e["age_unknown"] = True
                    else:
                        e["ages"].add(m["max_age_months"])
                    if m.get("age_base"):
                        e["age_bases"].add(m["age_base"])
                for src, dst in (("reason", "reasons"), ("remark", "remarks"), ("form_name", "forms")):
                    if m.get(src):
                        e[dst].add(m[src])

        for row in by_type.get("effect_duplicate", []):
            m = meta(row)
            code, eff = m.get("ingr_code_a"), m.get("effect_name")
            if code and eff:
                self.add_ingredient(code, m.get("ingredient_a_kor", ""), m.get("ingredient_a_eng", ""), True)
                if m.get("sers_name"):
                    self.effects[(code, eff)].add(m["sers_name"])
                else:
                    self.effects[(code, eff)]

        for t, store, field in (("dose_caution", self.dose_limits, "max_qty"),
                                ("dosing_period_caution", self.term_limits, "max_term")):
            for row in by_type.get(t, []):
                m = meta(row)
                code = m.get("ingr_code_a")
                if code and m.get(field):
                    self.add_ingredient(code, m.get("ingredient_a_kor", ""), m.get("ingredient_a_eng", ""), True)
                    form = m.get("form_name") or "전체"
                    store[code].add(f"{form}: {m[field]}")

    # ── 코드 매핑 ─────────────────────────
    def build_code_map(self, rows: list[dict]):
        for r in rows:
            r = meta(r)
            if r.get("src_code") and r.get("dur_code"):
                self.m2d[r["src_code"]].add(r["dur_code"])
                self.add_ingredient(r["dur_code"], r.get("dur_name_kor", ""), r.get("dur_name_eng", ""), True)
                if r.get("src_name"):
                    self.ingredients[r["dur_code"]]["aliases"].add(r["src_name"])

    # ── 허가정보 (제품 + CONTAINS) ─────────
    def build_product_flags(self, by_type: dict[str, list[dict]]):
        """DUR 품목정보: 식약처가 성분 규칙을 실제 제품에 적용한 목록 (제형 조건 반영됨)"""
        for t, group in TYPE_TO_GROUP.items():
            for row in by_type.get(t, []):
                m = meta(row)
                seq, code = m.get("item_seq"), m.get("ingr_code_a")
                if seq and code:
                    self.product_flags[(seq, group)].add(code)
        self.stats["DUR_FLAGGED_제품지정"] = len(self.product_flags)

    def build_permit(self, products: list[dict], ingr_rows: list[dict]):
        for p in products:
            m = meta(p)
            seq = m.get("item_seq")
            if not seq:
                continue
            self.drugs[seq] = {
                "item_seq": seq,
                "name": m.get("name", ""),
                "name_eng": m.get("name_eng", ""),
                "manufacturer": m.get("manufacturer", ""),
                "etc_otc": m.get("etc_otc", ""),
                "is_otc": bool(m.get("is_otc")),
                "is_active": bool(m.get("is_active", True)),
                "is_finished_drug": bool(m.get("is_finished_drug", True)),
                "atc_code": m.get("atc_code", ""),
                "bar_codes": list(m.get("bar_codes") or []),
                "official_url": m.get("official_url", ""),
            }

        for r in ingr_rows:
            m = meta(r)
            seq, src = m.get("item_seq"), m.get("ingr_code", "")
            if not seq:
                continue
            if seq not in self.drugs:              # mcpn에만 있는 제품 → 최소 정보로 생성
                self.drugs[seq] = {"item_seq": seq, "name": m.get("item_name", ""), "etc_otc": m.get("etc_otc", ""),
                                   "is_otc": bool(m.get("is_otc")), "is_active": bool(m.get("is_active", True)),
                                   "is_finished_drug": True, "name_eng": "", "manufacturer": "",
                                   "atc_code": "", "bar_codes": [], "official_url": ""}
            d_codes = sorted(self.m2d.get(src, set()))
            if d_codes:
                targets, mapped = d_codes, True
                self.stats["CONTAINS_D코드연결"] += 1
            else:
                key = f"M:{src}" if src else f"N:{m.get('ingr_name_kor', '')}"
                if key in ("M:", "N:"):
                    continue
                self.add_ingredient(key, m.get("ingr_name_kor", ""), m.get("ingr_name_eng", ""), False)
                targets, mapped = [key], False
                self.stats["CONTAINS_DUR외성분"] += 1
            for code in targets:
                self.contains[(seq, code)] = {
                    "item_seq": seq, "code": code, "src_code": src, "mapped_to_dur": mapped,
                    "amount": m.get("amount", ""), "unit": m.get("unit", ""),
                }

    # ── Neo4j 적재용 행 변환 ───────────────
    def rows(self) -> dict[str, list[dict]]:
        ingredients = [{**v, "aliases": sorted(v["aliases"])} for v in self.ingredients.values()]
        for r in ingredients:
            r["dose_limits"] = sorted(self.dose_limits.get(r["code"], []))
            r["term_limits"] = sorted(self.term_limits.get(r["code"], []))

        pairs = []
        for (lo, hi), e in self.pairs.items():
            reasons = [r for r, _ in sorted(e["reasons"].items(), key=lambda kv: kv[1], reverse=True)]
            dates = sorted(e["dates"])
            pairs.append({
                "lo": lo, "hi": hi,
                "reasons": reasons,
                "latest_reason": reasons[0] if reasons else "",
                "conditions": sorted(e["conditions"]),
                "variants": sorted(e["variants"]),
                "first_notified": dates[0] if dates else "",
                "last_notified": dates[-1] if dates else "",
                "n_records": e["n_records"],
            })

        restrictions = []
        for (code, group), e in self.restrictions.items():
            restrictions.append({
                "code": code, "group": group, "level": e["level"], "dur_type": e["dur_type"],
                "pregnancy_grade": min(e["grades"]) if e["grades"] else None,     # 1등급이 가장 엄격
                # 연령: 알려진 기준 중 가장 넓은 범위, 기준을 모르는 레코드가 있으면 age_unknown=True
                "max_age_months": max(e["ages"]) if e["ages"] else None,
                "age_unknown": e["age_unknown"],
                "age_bases": sorted(e["age_bases"]),
                "reasons": sorted(e["reasons"]),
                "remarks": sorted(e["remarks"]),
                "forms": sorted(e["forms"]),
            })

        effects = [{"code": c, "effect": eff, "series": sorted(s)} for (c, eff), s in self.effects.items()]
        return {
            "ingredients": ingredients,
            "drugs": list(self.drugs.values()),
            "contains": list(self.contains.values()),
            "pairs": pairs,
            "restrictions": restrictions,
            "effects": effects,
            "product_flags": [{"item_seq": seq, "group": grp, "ingr_codes": sorted(codes)}
                              for (seq, grp), codes in self.product_flags.items()],
        }


# ─────────────────────────────────────
# Neo4j 쿼리
# ─────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT ingredient_code IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.code IS UNIQUE",
    "CREATE CONSTRAINT drug_item_seq IF NOT EXISTS FOR (d:Drug) REQUIRE d.item_seq IS UNIQUE",
    "CREATE CONSTRAINT patient_group_code IF NOT EXISTS FOR (g:PatientGroup) REQUIRE g.code IS UNIQUE",
    "CREATE CONSTRAINT effect_group_name IF NOT EXISTS FOR (e:EffectGroup) REQUIRE e.name IS UNIQUE",
    "CREATE INDEX ingredient_name_kor IF NOT EXISTS FOR (i:Ingredient) ON (i.name_kor)",
    "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
]

Q = {
    "ingredients": """
        UNWIND $rows AS r
        MERGE (i:Ingredient {code: r.code})
        SET i.name_kor = r.name_kor, i.name_eng = r.name_eng, i.aliases = r.aliases,
            i.in_dur = r.in_dur, i.dose_limits = r.dose_limits, i.term_limits = r.term_limits""",
    "drugs": """
        UNWIND $rows AS r
        MERGE (d:Drug {item_seq: r.item_seq})
        SET d.name = r.name, d.name_eng = r.name_eng, d.manufacturer = r.manufacturer,
            d.etc_otc = r.etc_otc, d.is_otc = r.is_otc, d.is_active = r.is_active,
            d.is_finished_drug = r.is_finished_drug, d.atc_code = r.atc_code,
            d.bar_codes = r.bar_codes, d.official_url = r.official_url""",
    "contains": """
        UNWIND $rows AS r
        MATCH (d:Drug {item_seq: r.item_seq})
        MATCH (i:Ingredient {code: r.code})
        MERGE (d)-[c:CONTAINS]->(i)
        SET c.src_code = r.src_code, c.mapped_to_dur = r.mapped_to_dur,
            c.amount = r.amount, c.unit = r.unit""",
    "pairs": f"""
        UNWIND $rows AS r
        MATCH (a:Ingredient {{code: r.lo}})
        MATCH (b:Ingredient {{code: r.hi}})
        MERGE (a)-[e:CONTRAINDICATED_WITH]->(b)
        SET e.reasons = r.reasons, e.latest_reason = r.latest_reason, e.conditions = r.conditions,
            e.variants = r.variants, e.first_notified = r.first_notified, e.last_notified = r.last_notified,
            e.n_records = r.n_records, e.level = 'ban',
            e.source = '{DUR_SOURCE}', e.source_url = '{DUR_SOURCE_URL}'""",
    "restrictions": f"""
        UNWIND $rows AS r
        MATCH (i:Ingredient {{code: r.code}})
        MATCH (g:PatientGroup {{code: r.group}})
        MERGE (i)-[e:RESTRICTED_FOR]->(g)
        SET e.level = r.level, e.dur_type = r.dur_type, e.pregnancy_grade = r.pregnancy_grade,
            e.max_age_months = r.max_age_months, e.age_unknown = r.age_unknown, e.age_bases = r.age_bases,
            e.reasons = r.reasons, e.remarks = r.remarks, e.forms = r.forms,
            e.source = '{DUR_SOURCE}', e.source_url = '{DUR_SOURCE_URL}'""",
    "effects": f"""
        UNWIND $rows AS r
        MATCH (i:Ingredient {{code: r.code}})
        MERGE (g:EffectGroup {{name: r.effect}})
        MERGE (i)-[e:BELONGS_TO]->(g)
        SET e.series = r.series, e.source = '{DUR_SOURCE}', e.source_url = '{DUR_SOURCE_URL}'""",
}
# 노드를 먼저, 관계를 나중에 (MATCH 대상이 있어야 하므로 순서 중요)
Q["product_flags"] = """
        UNWIND $rows AS r
        MATCH (d:Drug {item_seq: r.item_seq})
        MATCH (g:PatientGroup {code: r.group})
        MERGE (d)-[f:DUR_FLAGGED]->(g)
        SET f.ingr_codes = r.ingr_codes, f.source = '식약처 DUR 품목정보',
            f.source_url = 'https://www.data.go.kr/data/15059486/openapi.do'"""

LOAD_ORDER = ["ingredients", "drugs", "contains", "pairs", "restrictions", "effects", "product_flags"]


def reset(session):
    total = 0
    while True:
        c = session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c").single()["c"]
        total += c
        if c == 0:
            break
    logger.info(f"  기존 노드 {total}개 삭제")


def write(session, name: str, rows: list[dict]):
    for i in range(0, len(rows), BATCH):
        session.run(Q[name], rows=rows[i:i + BATCH]).consume()
    logger.info(f"  ✅ {name:13} {len(rows):>7,}건")


def verify(session):
    logger.info("─" * 60)
    for r in session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
        logger.info(f"  노드 {r['label']:14} {r['n']:>8,}")
    for r in session.run("MATCH ()-[e]->() RETURN type(e) AS t, count(*) AS n ORDER BY n DESC"):
        logger.info(f"  관계 {r['t']:22} {r['n']:>8,}")

    r = session.run("""
        MATCH (d:Drug) WHERE d.is_otc AND d.is_active AND d.is_finished_drug
        OPTIONAL MATCH (d)-[c:CONTAINS]->()
        WITH d, count(c) AS n_all, sum(CASE WHEN c.mapped_to_dur THEN 1 ELSE 0 END) AS n_dur
        RETURN count(d) AS total,
               sum(CASE WHEN n_all > 0 THEN 1 ELSE 0 END) AS with_ingr,
               sum(CASE WHEN n_dur > 0 THEN 1 ELSE 0 END) AS with_dur""").single()
    t = max(r["total"], 1)
    logger.info(f"  유효 완제 일반의약품 {r['total']:,}개 | 성분 연결 {r['with_ingr']:,} ({r['with_ingr'] / t * 100:.1f}%)"
                f" | DUR 성분 포함 {r['with_dur']:,} ({r['with_dur'] / t * 100:.1f}%)")

    sample = session.run("""
        MATCH (a:Ingredient {name_kor:'이트라코나졸'})-[e:CONTRAINDICATED_WITH]-(b:Ingredient {name_kor:'심바스타틴'})
        RETURN e.latest_reason AS reason, e.conditions AS cond""").single()
    logger.info(f"  샘플 이트라코나졸–심바스타틴: {dict(sample) if sample else '없음 ⚠️'}")
    logger.info("─" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="그래프 전체 삭제 후 재적재")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("[GRAPH 1] Silver → Neo4j 적재 시작")
    logger.info("=" * 50)

    logger.info("① Silver 로드")
    dur_types = ["combination_ban", "pregnancy_ban", "age_specific_ban", "elderly_caution",
                 "dose_caution", "dosing_period_caution", "effect_duplicate"]
    by_type = {t: latest_items(f"silver/dur_ingredient/{t}/") for t in dur_types}
    code_map = latest_items("silver/dur_ingredient/code_map/")
    products = latest_items("silver/drugs/drug_permit/")
    ingr_rows = latest_items("silver/drug_ingredients/")
    product_dur = {t: latest_items(f"silver/dur_interactions/{t}/") for t in TYPE_TO_GROUP}

    logger.info("② 그래프 데이터 구성")
    g = GraphData()
    g.build_dur(by_type)
    g.build_code_map(code_map)
    g.build_permit(products, ingr_rows)
    g.build_product_flags(product_dur)
    rows = g.rows()
    for k, v in g.stats.items():
        logger.info(f"  {k}: {v:,}")
    amb = sum(1 for ds in g.m2d.values() if len(ds) > 1)
    if amb:
        logger.warning(f"  원료코드 {amb}종이 여러 D코드에 매핑됨 → 모두 연결 (안전 측 처리)")

    logger.info("③ Neo4j 적재")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password1234")),
    )
    try:
        with driver.session() as s:
            if args.reset:
                reset(s)
            for c in CONSTRAINTS:
                s.run(c).consume()
            s.run("UNWIND $rows AS r MERGE (g:PatientGroup {code: r.code}) SET g.name = r.name",
                  rows=PATIENT_GROUPS).consume()
            for name in LOAD_ORDER:
                write(s, name, rows[name])
            verify(s)
    finally:
        driver.close()
    logger.info("🎉 적재 완료 → Neo4j Browser: http://localhost:7474")


if __name__ == "__main__":
    main()
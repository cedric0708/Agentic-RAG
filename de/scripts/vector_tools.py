"""
[VECTOR] 벡터 DB 검색 도구 (AI 파트 연동용)

AI Agent가 호출하는 '근거 문서 검색' 함수 모음입니다. graph_tools.py와 짝으로 씁니다.
  - 그래프(graph_tools)는 "먹어도 되는가"를 판정하고   ← 판정은 항상 그래프
  - 벡터 DB(이 파일)는 "무슨 약인지 / 왜 안 되는지 / 어떤 질환인지"를 설명합니다

함수 목록
  search(query, doc_types, ...)          범용 의미 검색 (질문 여러 개를 한 번에 넣을 수 있음)
  search_drugs(query, ...)               증상 → 일반의약품 후보 (효능 + 허가정보, 일반의약품·허가 유효만)
  get_drug_info(item_seq|name, ...)      특정 약의 효능/용법/주의사항/부작용 원문 (임베딩 없이 조회)
  explain_findings(assess_result)        graph_tools.assess() 결과의 금기·주의마다 DUR 근거 문서를 붙임
  explain_ingredient(codes|names, ...)   성분 단위 금기·주의 문서 (임베딩 없이 조회)
  search_symptoms(query, ...)            질환·증상 설명 문서
  emergency_check(text)                  응급 증상 규칙 검사 ★ 검색보다 먼저 호출 (DB·모델 불필요)
  describe()                             테이블 상태 (doc_type별 건수, dur_type 값)

검색 규칙 (v1/v2 검색 품질 비교 결과 반영)
  1. 사용자 표현을 의학 용어로 바꾼 질문을 함께 넣기
     search_symptoms(["속이 쓰리고 신물이 올라와요", "위산 역류 속쓰림 가슴쓰림"])
     → 일상 표현만: 관련 0~20% / 의학 용어 추가: 100%
  2. 병용금기·임부금기 등은 dur_type으로 좁히기 (explain_* 함수는 자동 적용)
  3. 금기 '판정'은 graph_tools.assess(). 이 파일의 결과로 먹어도 된다/안 된다를 판단하지 않기
  4. 추천 후보는 일반의약품만 (search_drugs 기본값 otc_only=True)
  5. 응급 증상은 emergency_check()를 먼저. symptom_redflag 검색에만 의존하지 않기
     (심근경색·협심증·뇌졸중 문서가 아직 데이터에 없음)

사용법
  from vector_tools import DrugVectorDB
  v = DrugVectorDB()                    # .env의 PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / VECTOR_DB 사용
  v.emergency_check("가슴이 조이고 식은땀이 나요")
  v.search_drugs(["머리가 아프고 열이 나요", "두통 발열 해열진통제"])
  v.get_drug_info(name="타이레놀정500밀리그람")
  v.close()

  python vector_tools.py               # 동작 확인 (임베딩 모델 다운로드 약 2.2GB, CPU 가능)
  python vector_tools.py --no-model    # 모델 없이 되는 기능만 확인

필요 패키지: pip install psycopg2-binary sentence-transformers python-dotenv
"""
import json
import os
import re
from pathlib import Path

import psycopg2
import psycopg2.extras

try:  # 이 파일 위치 기준 config/.env (없으면 이미 설정된 환경변수 사용)
    from dotenv import load_dotenv
    for p in Path(__file__).resolve().parents:
        if (p / "config" / ".env").exists():
            load_dotenv(p / "config" / ".env")
            break
except ImportError:
    pass

TABLE = "drug_documents_v2"
MODEL_NAME = "intfloat/multilingual-e5-large"   # 임베딩할 때와 같은 모델이어야 함 (1024차원)
DISCLAIMER = "검색된 문서는 참고 정보이며, 복용 여부 판단은 그래프 판정(assess)과 약사·의사 상담을 따르세요."

DOC_TYPES = {
    "drug_efficacy":    "일반의약품 효능 (e약은요)",
    "drug_usage":       "일반의약품 용법·용량",
    "drug_caution":     "일반의약품 주의사항",
    "drug_side_effect": "일반의약품 부작용",
    "drug_permit":      "식약처 허가정보 (section=summary: 제품 요약 / caution: 사용상주의사항)",
    "dur_ingredient":   "성분 단위 금기·주의 (금기 근거 검색은 이것만)",
    "symptom_info":     "질환·증상 설명",
    "symptom_redflag":  "병원·응급실에 가야 하는 경우 (보조용, 응급 판단은 emergency_check)",
}

# dur_type 값 (DB에는 영문 키로 저장됨)
DUR_TYPES = {
    "combination_ban":       "병용금기",
    "pregnancy_ban":         "임부금기",
    "age_specific_ban":      "특정연령대금기",
    "elderly_caution":       "노인주의",
    "dose_caution":          "용량주의",
    "dosing_period_caution": "투여기간주의",
    "effect_duplicate":      "효능군중복",
}
_DUR_ALIAS = {**{k: k for k in DUR_TYPES}, **{v: k for k, v in DUR_TYPES.items()},
              "연령금기": "age_specific_ban", "소아금기": "age_specific_ban"}

# graph_tools.assess() findings → dur_type
FINDING_TO_DUR = {
    "contraindication": "combination_ban",
    "effect_duplicate": "effect_duplicate",
    "PREGNANT":         "pregnancy_ban",
    "PEDIATRIC":        "age_specific_ban",
    "ELDERLY":          "elderly_caution",
}

RESULT_COLUMNS = ["doc_id", "doc_type", "section", "chunk_index", "name", "item_seq", "text",
                  "dur_type", "level", "ingredient_a", "ingredient_b", "is_otc", "is_active",
                  "manufacturer", "source_name", "source_url", "item_url"]


# ─────────────────────────────────────
# 응급 증상 규칙 (초안 — AI 파트·팀과 함께 검토 후 확정)
#   groups: 모든 그룹에서 단어가 하나 이상 나오면 해당 (띄어쓰기 무시)
# ─────────────────────────────────────
EMERGENCY_RULES = [
    {"id": "chest_pain", "level": "EMERGENCY", "label": "심장 이상 의심 (가슴 조임·압박)",
     "groups": [["가슴", "흉통", "명치", "심장"], ["조이", "조여", "쥐어짜", "짓누르", "누르는", "압박", "뻐근"]],
     "action": "즉시 119에 연락하거나 가까운 응급실로 가세요. 약을 먹고 지켜보지 마세요."},
    {"id": "chest_pain_sweat", "level": "EMERGENCY", "label": "가슴 통증 + 식은땀·호흡곤란",
     "groups": [["가슴", "흉통"], ["식은땀", "숨이차", "숨쉬기", "호흡곤란", "왼팔", "턱"]],
     "action": "즉시 119에 연락하거나 가까운 응급실로 가세요."},
    {"id": "stroke_weakness", "level": "EMERGENCY", "label": "뇌졸중 의심 (한쪽 마비·감각 이상)",
     "groups": [["한쪽", "반쪽", "편측", "왼쪽", "오른쪽"], ["마비", "힘이빠", "힘이안", "감각이없", "저리"]],
     "action": "즉시 119에 연락하세요. 증상이 시작된 시각을 기억해 두세요."},
    {"id": "stroke_speech", "level": "EMERGENCY", "label": "뇌졸중 의심 (말·얼굴 이상)",
     "groups": [["말이어눌", "발음이이상", "말을못", "말이안나", "얼굴이비뚤", "입이돌아", "입꼬리가처"]],
     "action": "즉시 119에 연락하세요. 증상이 시작된 시각을 기억해 두세요."},
    {"id": "breathing", "level": "EMERGENCY", "label": "심한 호흡곤란",
     "groups": [["숨을못", "숨이막", "호흡곤란", "숨쉬기힘들", "숨쉬기어려", "숨이안쉬"]],
     "action": "즉시 119에 연락하거나 응급실로 가세요."},
    {"id": "anaphylaxis", "level": "URGENT", "label": "알레르기 반응 의심 (입술·혀·얼굴 부기)",
     "groups": [["입술", "혀", "목구멍", "얼굴", "눈두덩"], ["붓", "부어", "부었"]],
     "action": "숨이 차거나 목이 조이면 즉시 119에 연락하세요. 방금 먹은 약은 더 먹지 마세요."},
    {"id": "consciousness", "level": "EMERGENCY", "label": "의식 저하·경련",
     "groups": [["의식이없", "의식을잃", "정신을잃", "기절", "실신", "경련", "발작", "깨워도"]],
     "action": "즉시 119에 연락하세요."},
    {"id": "bleeding", "level": "EMERGENCY", "label": "출혈 (토혈·혈변·흑변·각혈)",
     "groups": [["피를토", "토혈", "검은변", "까만변", "흑변", "혈변", "피가섞인변", "각혈", "피가멈추지않"]],
     "action": "즉시 응급실로 가세요."},
    {"id": "thunderclap_headache", "level": "EMERGENCY", "label": "갑작스러운 극심한 두통",
     "groups": [["갑자기", "벼락", "태어나서처음", "가장심한", "망치로"], ["두통", "머리가아", "머리가깨질"]],
     "action": "즉시 응급실로 가세요."},
    {"id": "fever_stiff_neck", "level": "EMERGENCY", "label": "발열 + 목 경직",
     "groups": [["열이", "고열", "발열"], ["목이뻣뻣", "목이굳", "목을못숙"]],
     "action": "즉시 응급실로 가세요."},
    {"id": "overdose", "level": "EMERGENCY", "label": "약물 과다 복용",
     "groups": [["많이먹었", "과다복용", "한꺼번에먹", "통째로먹", "잘못먹었", "삼켰"]],
     "action": "즉시 119에 연락하세요. 먹은 약의 이름과 양을 확인해 두세요."},
    {"id": "self_harm", "level": "EMERGENCY", "label": "자해·자살 위험",
     "groups": [["죽고싶", "자살", "자해", "극단적선택", "살고싶지않"]],
     "action": "혼자 견디지 마세요. 자살예방상담전화 109(24시간) 또는 119에 바로 연락하세요."},
    {"id": "infant_fever", "level": "URGENT", "label": "영아 발열",
     "groups": [["개월", "신생아", "갓난", "아기"], ["열이", "고열", "발열", "38도", "39도"]],
     "action": "3개월 미만 아기의 발열은 바로 소아청소년과 진료를 받으세요. 해열제를 임의로 먹이지 마세요."},
    {"id": "pregnancy_bleeding", "level": "URGENT", "label": "임신 중 출혈·복통",
     "groups": [["임신", "임산부"], ["출혈", "피가나", "하혈", "배가심하게"]],
     "action": "바로 산부인과 또는 응급실 진료를 받으세요."},
]
_LEVEL_RANK = {"NONE": 0, "URGENT": 1, "EMERGENCY": 2}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


class DrugVectorDB:
    def __init__(self, dsn: dict | None = None, model=None, embed_fn=None, table: str = TABLE):
        """
        dsn      : psycopg2 접속 정보 (없으면 .env)
        model    : 이미 로드한 SentenceTransformer (없으면 처음 검색할 때 로드)
        embed_fn : 직접 만든 임베딩 함수 list[str] -> list[list[float]] ("query: " 접두어는 이 함수가 붙여야 함)
        """
        self.table = table
        self.conn = psycopg2.connect(**(dsn or {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": int(os.getenv("PG_PORT", "5432")),
            "dbname": os.getenv("VECTOR_DB", "drug_rag"),
            "user": os.getenv("PG_USER", "airflow"),
            "password": os.getenv("PG_PASSWORD", "airflow"),
        }))
        self.conn.autocommit = True
        self._model = model
        self._embed_fn = embed_fn
        with self.conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if not row:
                raise RuntimeError("이 DB에 pgvector가 없습니다. VECTOR_DB(기본 drug_rag) 이름을 확인하세요.")
            self.pgvector_version = row[0]
            cur.execute("SET hnsw.ef_search = 100")
            if tuple(int(x) for x in row[0].split(".")[:2]) >= (0, 8):
                cur.execute("SET hnsw.iterative_scan = 'relaxed_order'")   # 필터를 걸어도 결과가 덜 빠지게

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 내부 ─────────────────────────────
    def _embed(self, queries: list[str]) -> list[list[float]]:
        if self._embed_fn:
            return [list(map(float, v)) for v in self._embed_fn(queries)]
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            self._model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        vecs = self._model.encode([f"query: {q}" for q in queries], normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def _rows(self, sql: str, params) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def _queries(query) -> list[str]:
        qs = [query] if isinstance(query, str) else list(query or [])
        qs = [q.strip() for q in qs if q and q.strip()]
        if not qs:
            raise ValueError("query가 비어 있습니다.")
        return qs

    # ── 범용 검색 ────────────────────────
    def search(self, query: str | list[str], doc_types: str | list[str], top_k: int = 5,
               section: str | None = None, otc_only: bool = False, active_only: bool = False,
               dur_types: list[str] | None = None, item_seqs: list[str] | None = None,
               min_score: float = 0.0, per_type_k: int | None = None) -> list[dict]:
        """
        query    : 문자열 또는 문자열 목록 (원문 + 의학 용어로 바꾼 질문을 함께 넣기 권장)
        doc_types: 'drug_efficacy' 등 하나 또는 여러 개 (DOC_TYPES 참고)
        반환     : score 높은 순 dict 목록. 같은 문서가 여러 질문에 걸리면 가장 높은 점수 1개만
        """
        qs = self._queries(query)
        types = [doc_types] if isinstance(doc_types, str) else list(doc_types)
        bad = [t for t in types if t not in DOC_TYPES]
        if bad:
            raise ValueError(f"알 수 없는 doc_type: {bad} (가능: {list(DOC_TYPES)})")
        durs = [_DUR_ALIAS.get(d, d) for d in (dur_types or [])]
        k = per_type_k or top_k

        best: dict[str, dict] = {}
        for qi, vec in enumerate(self._embed(qs)):
            vec_txt = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            for t in types:   # doc_type마다 따로 검색 → 문서 종류별 HNSW 인덱스 사용
                where, params = ["doc_type = %s"], [t]
                if section:
                    where.append("section = %s"); params.append(section)
                if otc_only:
                    where.append("is_otc IS TRUE");
                if active_only:
                    where.append("is_active IS TRUE")
                if durs:
                    where.append("dur_type = ANY(%s)"); params.append(durs)
                if item_seqs:
                    where.append("item_seq = ANY(%s)"); params.append(list(item_seqs))
                sql = (f"SELECT {', '.join(RESULT_COLUMNS)}, 1 - (embedding <=> %s::vector) AS score "
                       f"FROM {self.table} WHERE {' AND '.join(where)} "
                       f"ORDER BY embedding <=> %s::vector LIMIT %s")
                for r in self._rows(sql, [vec_txt] + params + [vec_txt, k]):
                    r["score"] = round(float(r["score"]), 4)
                    r["matched_query"] = qs[qi]
                    if r["score"] >= min_score and (r["doc_id"] not in best or r["score"] > best[r["doc_id"]]["score"]):
                        best[r["doc_id"]] = r
        return sorted(best.values(), key=lambda r: -r["score"])[:top_k]

    # ── 증상 → 일반의약품 후보 ───────────
    def search_drugs(self, query: str | list[str], top_k: int = 5, otc_only: bool = True,
                     include_permit: bool = True) -> list[dict]:
        """
        증상에 맞는 약 후보. 같은 약은 1개로 묶고(가장 높은 점수), 후보마다 graph_tools.assess()로 판정할 것
          - drug_efficacy (e약은요: 일반의약품 효능)
          - drug_permit section=summary (식약처 허가 제품 요약, 일반의약품·허가 유효만)
        반환 항목의 item_seq가 비어 있으면 graph_tools.find_drug(name)으로 찾기
        """
        rows = self.search(query, "drug_efficacy", top_k=top_k * 3)
        if include_permit:
            rows += self.search(query, "drug_permit", top_k=top_k * 3, section="summary",
                                otc_only=otc_only, active_only=True)
        merged: dict[str, dict] = {}
        for r in sorted(rows, key=lambda r: -r["score"]):
            key = r["item_seq"] or _norm_name(r["name"])
            if key and key not in merged:
                merged[key] = r
        out = list(merged.values())[:top_k]
        self._fill_item_seq(out)
        return out

    def _fill_item_seq(self, rows: list[dict]) -> None:
        """e약은요 문서처럼 item_seq가 비어 있으면 허가정보의 같은 제품명으로 채움 (graph_tools.assess에 바로 쓰도록)"""
        need = {_norm_name(r["name"]) for r in rows if not r.get("item_seq") and r.get("name")}
        if not need:
            return
        found = self._rows(
            f"SELECT DISTINCT ON (k) k, item_seq, name, is_otc, is_active FROM ("
            f"  SELECT regexp_replace(lower(regexp_replace(name, '\\([^)]*\\)|\\s', '', 'g')), '밀리그[램람]', 'mg', 'g') AS k, "
            f"         item_seq, name, is_otc, is_active "
            f"  FROM {self.table} WHERE doc_type = 'drug_permit' AND section = 'summary' AND chunk_index = 0"
            f") t WHERE k = ANY(%s) ORDER BY k, is_active DESC NULLS LAST, is_otc DESC NULLS LAST", [list(need)])
        by_key = {f["k"]: f for f in found}
        for r in rows:
            if not r.get("item_seq") and (f := by_key.get(_norm_name(r.get("name")))):
                r["item_seq"] = f["item_seq"]
                r["item_seq_from"] = f"허가정보 제품명 일치: {f['name']}"

    # ── 특정 약 정보 (임베딩 없음) ────────
    def get_drug_info(self, item_seq: str | None = None, name: str | None = None,
                      sections: tuple = ("efficacy", "usage", "caution", "side_effect"),
                      include_permit: bool = True) -> dict:
        """
        품목기준코드 또는 정확한 제품명으로 원문 조회 → {'name', 'item_seq', 'sections': {section: text}}
        e약은요 문서가 없는 약은 허가정보(drug_permit) 요약·주의사항으로 대신함
        """
        if not (item_seq or name):
            raise ValueError("item_seq 또는 name 중 하나는 필요합니다.")
        types = ["drug_efficacy", "drug_usage", "drug_caution", "drug_side_effect"] + (
            ["drug_permit"] if include_permit else [])
        cond, params = ("item_seq = %s", [item_seq]) if item_seq else ("name = %s", [name])
        rows = self._rows(
            f"SELECT doc_type, section, chunk_index, name, item_seq, text, source_name, item_url "
            f"FROM {self.table} WHERE {cond} AND doc_type = ANY(%s) "
            f"ORDER BY doc_type, section, chunk_index", params + [types])
        if not rows and name:   # 정확히 같은 이름이 없으면 공백 무시 부분 일치
            rows = self._rows(
                f"SELECT doc_type, section, chunk_index, name, item_seq, text, source_name, item_url "
                f"FROM {self.table} WHERE replace(name, ' ', '') LIKE %s AND doc_type = ANY(%s) "
                f"ORDER BY length(name), doc_type, section, chunk_index",
                [f"%{name.replace(' ', '')}%", types])
            if rows:   # 가장 짧은(가장 가까운) 이름 하나만
                first = rows[0]["name"]
                rows = [r for r in rows if r["name"] == first]
        out = {"name": rows[0]["name"] if rows else name, "item_seq": item_seq or next(
            (r["item_seq"] for r in rows if r["item_seq"]), None), "sections": {}, "sources": set()}
        for r in rows:
            sec = r["section"] or ""
            if r["doc_type"] == "drug_permit":
                sec = "permit_summary" if sec == "summary" else "permit_caution"
            elif sec not in sections:
                continue
            out["sections"].setdefault(sec, []).append(_strip_header(r["text"]))
            if r["source_name"]:
                out["sources"].add(r["source_name"])
        out["sections"] = {k: _join_chunks(v) for k, v in out["sections"].items()}
        out["sources"] = sorted(out["sources"])
        out["found"] = bool(rows)
        return out

    # ── 금기 근거 (임베딩 없음) ───────────
    def explain_ingredient(self, codes: list[str] | None = None, names: list[str] | None = None,
                           dur_types: list[str] | None = None, pair_with: list[str] | None = None,
                           limit: int = 20) -> list[dict]:
        """
        성분 단위 DUR 문서(dur_ingredient) 조회
          codes     : 성분코드(D코드, 그래프 Ingredient.code)  ← 정확, 권장
          names     : 한글 성분명 (codes가 없을 때)
          dur_types : 'combination_ban' / '병용금기' 등 (없으면 전체)
          pair_with : 병용금기 상대 성분코드 → 두 성분 쌍만
        """
        if not (codes or names):
            raise ValueError("codes 또는 names가 필요합니다.")
        where, params = ["doc_type = 'dur_ingredient'"], []
        if codes:
            where.append("(metadata->>'ingr_code_a' = ANY(%s) OR metadata->>'ingr_code_b' = ANY(%s))")
            params += [list(codes), list(codes)]
        else:
            where.append("(ingredient_a = ANY(%s) OR ingredient_b = ANY(%s))")
            params += [list(names), list(names)]
        if dur_types:
            where.append("dur_type = ANY(%s)"); params.append([_DUR_ALIAS.get(d, d) for d in dur_types])
        if pair_with:
            where.append("(metadata->>'ingr_code_a' = ANY(%s) OR metadata->>'ingr_code_b' = ANY(%s))")
            params += [list(pair_with), list(pair_with)]
        rows = self._rows(
            f"SELECT doc_id, dur_type, level, name, ingredient_a, ingredient_b, text, source_name, source_url, "
            f"metadata->>'ingr_code_a' AS code_a, metadata->>'ingr_code_b' AS code_b "
            f"FROM {self.table} WHERE {' AND '.join(where)} ORDER BY dur_type, doc_id LIMIT %s", params + [limit])
        seen, out = set(), []
        for r in rows:   # 제형만 다른 같은 내용 문서는 하나로
            sig = (r["dur_type"], r["code_a"], r["code_b"], re.sub(r"\|\s*제형:[^|]*", "", r["text"]))
            if sig not in seen:
                seen.add(sig)
                r["dur_type_name"] = DUR_TYPES.get(r["dur_type"], r["dur_type"])
                out.append(r)
        return out

    def explain_findings(self, assess_result: dict, per_finding: int = 2) -> dict:
        """
        graph_tools.assess() 결과를 받아 금기·주의 항목마다 근거 문서(evidence)를 붙여 반환
        판정(verdict)은 바꾸지 않음
        """
        result = json.loads(json.dumps(assess_result, ensure_ascii=False, default=str))
        for f in result.get("findings", []):
            dur = FINDING_TO_DUR.get(f.get("type")) or FINDING_TO_DUR.get(f.get("group"))
            if not dur:
                f["evidence"] = []
                continue
            if f.get("type") in ("contraindication", "effect_duplicate"):
                ev = self.explain_ingredient(codes=[f.get("candidate_code")], dur_types=[dur],
                                             pair_with=[f.get("current_code")] if dur == "combination_ban" else None,
                                             limit=per_finding * 3)
            else:
                ev = self.explain_ingredient(codes=[f.get("code")], dur_types=[dur], limit=per_finding * 3)
            f["evidence"] = [{"text": e["text"], "dur_type": e["dur_type_name"],
                              "source_name": e["source_name"], "source_url": e["source_url"]}
                             for e in ev[:per_finding]]
        return result

    # ── 질환·증상 ────────────────────────
    def search_symptoms(self, query: str | list[str], top_k: int = 5, include_redflag: bool = True,
                        group_by_disease: bool = False) -> list[dict]:
        """
        질환·증상 설명 검색. 위험 신호 문서(symptom_redflag)는 is_redflag=True로 표시
        group_by_disease=True: 같은 질환은 가장 높은 점수 조각 1개만
        ※ 응급 판단은 emergency_check()로 먼저 할 것
        """
        types = ["symptom_info"] + (["symptom_redflag"] if include_redflag else [])
        rows = self.search(query, types, top_k=top_k * (3 if group_by_disease else 1))
        for r in rows:
            r["is_redflag"] = r["doc_type"] == "symptom_redflag"
        if group_by_disease:
            seen, grouped = set(), []
            for r in rows:
                if r["name"] not in seen:
                    seen.add(r["name"]); grouped.append(r)
            rows = grouped
        return rows[:top_k]

    # ── 응급 규칙 (DB·모델 불필요) ────────
    @staticmethod
    def emergency_check(text: str) -> dict:
        """
        사용자 입력에서 응급 증상 규칙을 확인 → {'level': EMERGENCY|URGENT|NONE, 'matched': [...], 'message'}
        EMERGENCY면 약 추천을 하지 말고 action 안내를 먼저 보여줄 것
        """
        t = _norm(text)
        matched = []
        for rule in EMERGENCY_RULES:
            if all(any(_norm(w) in t for w in group) for group in rule["groups"]):
                matched.append({"id": rule["id"], "level": rule["level"], "label": rule["label"],
                                "action": rule["action"]})
        level = max((m["level"] for m in matched), key=_LEVEL_RANK.get, default="NONE")
        message = {
            "EMERGENCY": "응급 상황일 수 있습니다. 약을 먹고 지켜보지 말고 바로 아래 안내를 따르세요.",
            "URGENT": "일반의약품보다 먼저 진료가 필요한 상황일 수 있습니다.",
            "NONE": "",
        }[level]
        return {"level": level, "matched": matched, "message": message,
                "note": "규칙 기반 1차 확인입니다. 해당하지 않아도 증상이 심하거나 지속되면 진료를 받으세요."}

    # ── 상태 확인 ────────────────────────
    def describe(self) -> dict:
        counts = self._rows(f"SELECT doc_type, COALESCE(section, '') AS section, COUNT(*) AS n "
                            f"FROM {self.table} GROUP BY 1, 2 ORDER BY 1, 2", [])
        durs = self._rows(f"SELECT dur_type, COUNT(*) AS n FROM {self.table} "
                          f"WHERE doc_type = 'dur_ingredient' GROUP BY 1 ORDER BY 1", [])
        return {"pgvector": self.pgvector_version, "table": self.table,
                "doc_types": counts, "dur_types": durs}


def _norm_name(name: str | None) -> str:
    """'타이레놀정500밀리그람(아세트아미노펜)' → '타이레놀정500mg' (괄호·공백 제거, 단위 통일)"""
    n = re.sub(r"\([^)]*\)|\s", "", name or "").lower()
    return re.sub(r"밀리그[램람]", "mg", n)   # '325mg' = '325밀리그램' = '325밀리그람'


def _join_chunks(chunks: list[str]) -> str:
    """잘린 조각을 다시 이어붙임: 2번째 조각부터 반복되는 '주의사항:' 같은 라벨과 앞 조각과 겹치는 부분(overlap) 제거"""
    if not chunks:
        return ""
    label = re.match(r"^([^\s:|]{1,10}):\s*", chunks[0])
    out = chunks[0]
    for c in chunks[1:]:
        if label and c.startswith(label.group(0)):
            c = c[len(label.group(0)):]
        overlap = next((n for n in range(min(len(out), len(c), 120), 9, -1) if out.endswith(c[:n])), 0)
        out = f"{out}{c[overlap:]}" if overlap else f"{out} {c}"
    return out


def _strip_header(text: str) -> str:
    """'약품명: OO | 효능: …' → '효능: …' (조각마다 붙인 머리말 제거)"""
    return re.sub(r"^(약품명|질환명):[^|]*\|\s*", "", text or "").strip()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="임베딩 모델 없이 되는 기능만 확인")
    args = ap.parse_args()

    def show(title, obj):
        print(f"\n=== {title} ===")
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str)[:1500])

    show("emergency_check('가슴이 조이고 식은땀이 나요')", DrugVectorDB.emergency_check("가슴이 조이고 식은땀이 나요"))
    show("emergency_check('콧물이 나고 재채기가 나요')", DrugVectorDB.emergency_check("콧물이 나고 재채기가 나요"))

    with DrugVectorDB() as v:
        show("describe()", v.describe())
        show("get_drug_info(name='타이레놀정500밀리그람')", v.get_drug_info(name="타이레놀정500밀리그람"))
        show("explain_ingredient(names=['이부프로펜'], dur_types=['병용금기'])",
             v.explain_ingredient(names=["이부프로펜"], dur_types=["병용금기"], limit=5))

        try:   # 그래프가 연결돼 있으면 assess → explain_findings 연동 확인
            from graph_tools import DrugGraph
            with DrugGraph() as g:
                cur = g.find_drug("코니트라", limit=1)
                cand = g.find_drug("리피스탄", limit=1)
                if cur and cand:
                    res = g.assess(current=[cur[0]["item_seq"]], candidate=cand[0]["item_seq"], age=35)
                    show("explain_findings(assess(코니트라 + 리피스탄))", v.explain_findings(res))
        except Exception as e:
            print(f"\n(그래프 연동 확인 건너뜀: {str(e)[:80]})")

        if not args.no_model:
            show("search_drugs(['머리가 아프고 열이 나요', '두통 발열 해열진통제'])",
                 [{k: r[k] for k in ("name", "item_seq", "doc_type", "score")}
                  for r in v.search_drugs(["머리가 아프고 열이 나요", "두통 발열 해열진통제"])])
            show("search_symptoms(['속이 쓰리고 신물이 올라와요', '위산 역류 속쓰림 가슴쓰림'])",
                 [{k: r[k] for k in ("name", "doc_type", "score", "is_redflag")}
                  for r in v.search_symptoms(["속이 쓰리고 신물이 올라와요", "위산 역류 속쓰림 가슴쓰림"],
                                             group_by_disease=True)])
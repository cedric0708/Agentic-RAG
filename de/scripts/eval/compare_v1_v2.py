"""
v1(drug_documents) vs v2(drug_documents_v2) 검색 품질 비교

같은 질문을 같은 임베딩 모델(multilingual-e5-large)로 바꿔서 두 테이블을 검색하고,
상위 결과를 나란히 비교합니다. GPU 없이 CPU로 실행됩니다. 처음 실행할 때 모델(약 2.2GB)을 내려받습니다.

위치: de/scripts/eval/compare_v1_v2.py
설치: pip install sentence-transformers psycopg2-binary python-dotenv
실행:
  python compare_v1_v2.py              # 실제 서비스처럼 인덱스를 사용해 검색
  python compare_v1_v2.py --exact      # 인덱스를 끄고 전수 비교 (데이터 자체 품질만 비교)
  python compare_v1_v2.py --top-k 10

결과:
  de/eval_results/compare_v1_v2_YYYYMMDD_HHMM.md   ← 사람이 읽는 보고서 (요약표 + 상위 결과)
  de/eval_results/compare_v1_v2_YYYYMMDD_HHMM.csv  ← relevant 열에 1/0을 채워 정확도(P@k) 계산용

접속 정보 (config/.env, 없으면 기본값)
  PG_HOST=localhost  PG_PORT=5432  PG_USER=airflow  PG_PASSWORD=...
  VECTOR_DB=drug_rag   ← 벡터 테이블이 있는 DB 이름 (PG_DB와 다를 수 있어서 따로 둠)
"""
import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

DE_DIR = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(DE_DIR / "config" / ".env", override=True)
except ImportError:
    pass

import psycopg2
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"
V1_TABLE, V2_TABLE = "drug_documents", "drug_documents_v2"

# ─────────────────────────────────────────────────────────────
# 질문 세트
#   v1 / v2: (doc_type, section)  ← 각 버전에서 이 질문에 쓸 문서 종류. None이면 그 버전은 비교 안 함
#   keywords: 결과의 약품명이나 본문에 이 중 하나라도 있으면 "관련 있음"으로 자동 판정 (대략적인 지표)
# ─────────────────────────────────────────────────────────────
QUERIES = [
    dict(id="Q01", cat="증상→약", q="머리가 아프고 열이 나요",
         v1=("drug_efficacy", None), v2=("drug_efficacy", None),
         keywords=["두통", "해열", "발열", "진통"]),
    dict(id="Q02", cat="증상→약", q="콧물이 나고 재채기가 계속 나요",
         v1=("drug_efficacy", None), v2=("drug_efficacy", None),
         keywords=["콧물", "재채기", "비염", "코감기", "알레르기"]),
    dict(id="Q03", cat="증상→약", q="어린이가 먹을 수 있는 해열제 시럽",
         v1=("drug_efficacy", None), v2=("drug_efficacy", None),
         keywords=["어린이", "소아", "시럽", "현탁"]),
    dict(id="Q04", cat="용법", q="타이레놀은 하루에 몇 번 먹어야 하나요",
         v1=("drug_usage", None), v2=("drug_usage", None),
         keywords=["타이레놀", "아세트아미노펜"]),
    dict(id="Q05", cat="주의사항", q="임산부가 먹으면 안 되는 감기약",
         v1=("drug_caution", None), v2=("drug_caution", None),
         keywords=["임부", "임신", "임산부"]),
    dict(id="Q06", cat="부작용", q="이부프로펜 부작용",
         v1=("drug_side_effect", None), v2=("drug_side_effect", None),
         keywords=["이부프로펜"]),
    dict(id="Q07", cat="금기", q="이부프로펜과 같이 먹으면 안 되는 성분",
         v1=("dur_interaction", None), v2=("dur_ingredient", None),
         keywords=["이부프로펜"]),
    dict(id="Q08", cat="금기", q="노인이 복용할 때 주의해야 하는 수면유도 성분",
         v1=("dur_interaction", None), v2=("dur_ingredient", None),
         keywords=["노인", "고령", "독실아민", "디펜히드라민"]),
    dict(id="Q09", cat="금기", q="임신 중 사용하면 안 되는 진통제 성분",
         v1=("dur_interaction", None), v2=("dur_ingredient", None),
         keywords=["임부", "임신"]),
    dict(id="Q10", cat="증상정보", q="속이 쓰리고 신물이 올라와요",
         v1=("symptom_info", None), v2=("symptom_info", None),
         keywords=["위식도역류", "역류", "속쓰림", "위염", "위궤양"]),
    dict(id="Q11", cat="위험증상", q="가슴이 조이고 식은땀이 나요",
         v1=("symptom_redflag", None), v2=("symptom_redflag", None),
         keywords=["흉통", "심근경색", "협심증", "가슴"]),
    dict(id="Q12", cat="허가정보", q="졸음이 덜한 알레르기 비염약",
         v1=("drug_efficacy", None), v2=("drug_permit", "summary"),
         keywords=["알레르기", "비염", "두드러기", "가려움"]),
    dict(id="Q10b", cat="증상정보(용어변환)", q="위산 역류로 인한 속쓰림과 가슴쓰림",
         v1=("symptom_info", None), v2=("symptom_info", None),
         keywords=["위식도역류", "역류", "속쓰림", "위염", "위궤양"]),
]


def connect():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("VECTOR_DB", "drug_rag"),
        user=os.getenv("PG_USER", "airflow"),
        password=os.getenv("PG_PASSWORD", "airflow"),
    )


def setup_session(conn, exact):
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
        if not row:
            raise SystemExit("❌ 이 DB에 pgvector 확장이 없습니다 (VECTOR_DB 이름 확인)")
        ver = row[0]
        cur.execute("SET ivfflat.probes = 10")          # v1이 IVFFlat 인덱스일 때 정확도 보정
        cur.execute("SET hnsw.ef_search = 100")
        major, minor = (int(x) for x in ver.split(".")[:2])
        if (major, minor) >= (0, 8):
            cur.execute("SET hnsw.iterative_scan = 'relaxed_order'")   # 필터 걸어도 결과가 덜 빠지게
        if exact:
            cur.execute("SET enable_indexscan = off")   # 인덱스 끄기 → 전수 비교
        cur.execute("""SELECT table_name FROM information_schema.tables
                       WHERE table_name IN (%s, %s)""", (V1_TABLE, V2_TABLE))
        tables = {r[0] for r in cur.fetchall()}
    conn.commit()
    return ver, tables


def doc_types(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT doc_type FROM {table}")
        return {r[0] for r in cur.fetchall()}


def search(conn, table, qvec, doc_type, section, k):
    vec = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    where = "doc_type = %s" + (" AND section = %s" if section else "")
    params = [vec, doc_type] + ([section] if section else []) + [vec, k]
    sql = (f"SELECT name, doc_type, text, 1 - (embedding <=> %s::vector) AS score "
           f"FROM {table} WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s")
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.commit()
    return rows, (time.perf_counter() - t0) * 1000


def hit(row, keywords):
    blob = f"{row[0] or ''} {row[2] or ''}"
    return any(k in blob for k in keywords)


def summarize(rows, keywords):
    if not rows:
        return dict(hit=None, distinct=None, score=None)
    return dict(
        hit=sum(hit(r, keywords) for r in rows) / len(rows),
        distinct=len({(r[0] or "").strip() for r in rows}),
        score=statistics.mean(r[3] for r in rows),
    )


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.0%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def clip(text, n=120):
    t = " ".join((text or "").split())
    return (t[:n] + "…") if len(t) > n else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--exact", action="store_true", help="인덱스를 끄고 전수 검색")
    args = ap.parse_args()
    k = args.top_k

    conn = connect()
    ver, tables = setup_session(conn, args.exact)
    print(f"pgvector {ver} | 테이블: {sorted(tables)} | 모드: {'전수(--exact)' if args.exact else '인덱스 사용'}")
    for t in (V1_TABLE, V2_TABLE):
        if t not in tables:
            raise SystemExit(f"❌ 테이블 {t} 이 없습니다")
    types = {V1_TABLE: doc_types(conn, V1_TABLE), V2_TABLE: doc_types(conn, V2_TABLE)}

    print(f"모델 로드 중: {MODEL_NAME} (CPU, 처음엔 다운로드로 몇 분 걸림)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # 워밍업 (첫 쿼리는 캐시 때문에 느려서 속도 비교에서 제외)
    warm = model.encode("query: 워밍업", normalize_embeddings=True)
    search(conn, V2_TABLE, warm, "drug_efficacy", None, 1)
    search(conn, V1_TABLE, warm, "drug_efficacy", None, 1)

    results = []
    for q in QUERIES:
        qvec = model.encode(f"query: {q['q']}", normalize_embeddings=True)
        entry = dict(q=q)
        for ver_key, table in (("v1", V1_TABLE), ("v2", V2_TABLE)):
            spec = q[ver_key]
            if spec is None or spec[0] not in types[table]:
                entry[ver_key] = dict(rows=[], ms=None, spec=spec, missing=True)
                continue
            rows, ms = search(conn, table, qvec, spec[0], spec[1], k)
            entry[ver_key] = dict(rows=rows, ms=ms, spec=spec, missing=False)
        results.append(entry)
        s1, s2 = summarize(entry["v1"]["rows"], q["keywords"]), summarize(entry["v2"]["rows"], q["keywords"])
        print(f"{q['id']} {q['q'][:24]:<26} 관련 v1 {fmt(s1['hit'], True):>5} / v2 {fmt(s2['hit'], True):>5}"
              f"   서로 다른 약 v1 {fmt(s1['distinct']):>2} / v2 {fmt(s2['distinct']):>2}")
    conn.close()

    # ── 보고서 ──
    out_dir = DE_DIR / "eval_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_path, csv_path = out_dir / f"compare_v1_v2_{stamp}.md", out_dir / f"compare_v1_v2_{stamp}.csv"

    lines = [f"# v1 vs v2 검색 비교 ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"- 모델 `{MODEL_NAME}` / top-{k} / pgvector {ver} / 모드: {'전수 검색' if args.exact else '인덱스 사용'}",
             "- **관련**: 상위 결과 중 질문 키워드가 들어간 비율 (자동 판정, 대략적인 지표)",
             "- **서로 다른 약**: 상위 결과에 나온 서로 다른 이름 수 (같은 약 조각이 반복되면 낮아짐)",
             "- 유사도 점수는 v2가 머리말을 붙여서 v1과 기준이 조금 다릅니다 → 점수보다 **결과 내용**으로 판단하세요", "",
             "## 요약", "",
             "| ID | 분류 | 질문 | v1 문서 | v2 문서 | 관련 v1 | 관련 v2 | 서로 다른 약 v1 | 서로 다른 약 v2 | 속도 v1(ms) | 속도 v2(ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"v1": [], "v2": []}
    for e in results:
        q = e["q"]
        s = {v: summarize(e[v]["rows"], q["keywords"]) for v in ("v1", "v2")}
        for v in ("v1", "v2"):
            if s[v]["hit"] is not None:
                tot[v].append(s[v]["hit"])
        spec = {v: ("/".join(x for x in e[v]["spec"] if x) if e[v]["spec"] else "-") + (" (없음)" if e[v]["missing"] else "")
                for v in ("v1", "v2")}
        ms = {v: (f"{e[v]['ms']:.0f}" if e[v]["ms"] is not None else "-") for v in ("v1", "v2")}
        lines.append(f"| {q['id']} | {q['cat']} | {q['q']} | {spec['v1']} | {spec['v2']} | "
                     f"{fmt(s['v1']['hit'], True)} | {fmt(s['v2']['hit'], True)} | "
                     f"{fmt(s['v1']['distinct'])} | {fmt(s['v2']['distinct'])} | {ms['v1']} | {ms['v2']} |")
    avg = {v: (f"{statistics.mean(tot[v]):.0%}" if tot[v] else "-") for v in tot}
    lines += ["", f"**평균 관련 비율: v1 {avg['v1']} / v2 {avg['v2']}**", "", "## 질문별 상위 결과", ""]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "version", "rank", "doc_type", "name", "score", "keyword_hit", "relevant(1/0)", "text"])
        for e in results:
            q = e["q"]
            lines += [f"### {q['id']} · {q['q']}", ""]
            for v in ("v1", "v2"):
                lines.append(f"**{v}** ({'/'.join(x for x in e[v]['spec'] if x) if e[v]['spec'] else '-'})")
                if not e[v]["rows"]:
                    lines += ["", "- (해당 문서 종류 없음)", ""]
                    continue
                lines.append("")
                for i, r in enumerate(e[v]["rows"], 1):
                    mark = "✅" if hit(r, q["keywords"]) else "▫️"
                    lines.append(f"{i}. {mark} [{r[3]:.3f}] **{r[0] or '(이름 없음)'}** — {clip(r[2])}")
                    w.writerow([q["id"], q["q"], v, i, r[1], r[0], f"{r[3]:.4f}",
                                int(hit(r, q["keywords"])), "", " ".join((r[2] or "").split())])
                lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n평균 관련 비율: v1 {avg['v1']} / v2 {avg['v2']}")
    print(f"보고서: {md_path}")
    print(f"채점용 CSV: {csv_path}  (relevant 열에 1/0 입력 → 질문별 정확도 계산)")


if __name__ == "__main__":
    main()

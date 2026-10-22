# Drug RAG DE — 데이터 명세서

AI 파트와 WEB 파트가 DE 데이터를 쓰기 위한 데이터 구조, 종류, 태그, 접속 정보를 정리한 문서입니다.

- **데이터 기준일: 2026-10-22** (전체 재수집본, S3 날짜 폴더 `2026-10-22`)
- 비밀번호·API 키는 이 문서에 적지 않습니다. `config/.env`로 DE 파트가 개별 전달합니다.
- 사용법·Agent 흐름·검색 규칙의 자세한 설명은 노션 **「팀원 공유용 정리본」** 참고

---

## 1. 아키텍처 개요

```
공공 API · 크롤링
    ↓ ingestion/ (수집)
S3 Bronze (원본)
    ↓ etl/ (정제)
S3 Silver (정제 JSON)
    ├─ load_graph.py ──────────────→ Neo4j AuraDB (그래프: 금기 판정)
    └─ Colab GPU 임베딩 (multilingual-e5-large)
           ↓ pg_dump → S3 Gold (gold/pgvector_dump/)
       PostgreSQL + pgvector (공용 EC2 서버: 검색·근거)
                    ↓
AI 파트 (Agentic RAG)  ←  graph_tools.py (판정) + vector_tools.py (검색·설명·응급 규칙)
```

| 저장소 | 역할 | 접근 방법 |
|---|---|---|
| Neo4j AuraDB | "이 약을 먹어도 되는가" 판정 (병용금기, 임부·소아·노인 제한, 성분·효능군 중복) | `graph_tools.DrugGraph` |
| PostgreSQL + pgvector | 증상 → 약 후보 검색, 약 정보, 금기 근거 문서, 질환 설명 | `vector_tools.DrugVectorDB` |
| 코드 내 규칙 | 응급 증상 1차 확인 (DB 불필요) | `DrugVectorDB.emergency_check()` |
| S3 | 원본·정제 데이터, 벡터 DB 덤프 보관 | 필요할 때만 (8장) |

---

## 2. 접속 정보

### 2-1. 벡터 DB (공용 EC2 서버)

```
host:      43.200.194.39        (탄력적 IP, 고정)
port:      5433
dbname:    drug_rag
user:      drug_reader          (조회 전용)
password:  .env 참고 (DE 파트 개별 전달)
table:     drug_documents_v2    (이름의 v2는 고정된 이름일 뿐, 데이터 기준과 무관)

임베딩 모델: intfloat/multilingual-e5-large
임베딩 차원: 1024 (정규화, 코사인 유사도)
질문 접두사: "query: {질문}"     ← 필수 (문서는 "passage: "로 임베딩됨)
```

- 서버는 **등록된 IP에서만** 접속을 받습니다. 접속이 시간 초과되면 내 IP를 DE 파트에 알려 주세요.

### 2-2. 그래프 DB (Neo4j AuraDB)

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=.env 참고
```

- 브라우저로 볼 때: https://browser.neo4j.io 에 위 값으로 로그인
- 쓰기 권한이 있는 계정이므로 **조회 쿼리(`MATCH … RETURN`)만** 실행

### 2-3. `.env` 형식

`graph_tools.py`, `vector_tools.py`는 자기 위치에서 위쪽 폴더로 올라가며 `config/.env`를 찾아 자동으로 읽습니다.

```bash
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<DE 파트에 문의>

PG_HOST=43.200.194.39
PG_PORT=5433
PG_USER=drug_reader
PG_PASSWORD=<DE 파트에 문의>
VECTOR_DB=drug_rag
```

---

## 3. 사용 방법

### 3-1. 권장: 제공 함수 사용

```python
from graph_tools import DrugGraph
from vector_tools import DrugVectorDB

g, v = DrugGraph(), DrugVectorDB()

v.emergency_check("가슴이 조이고 식은땀이 나요")          # 응급 규칙 (가장 먼저)
v.search_drugs(["머리가 아프고 열이 나요", "두통 발열 해열진통제"], top_k=5)   # 일반의약품 후보
v.search_symptoms(["속이 쓰리고 신물", "위산 역류 속쓰림 가슴쓰림"])          # 질환 설명
res = g.assess(current=["200402374"], candidate="200000913", age=35, pregnant=False)  # 판정
res = v.explain_findings(res)                             # 판정 근거 문서 첨부
v.get_drug_info(item_seq="202106092")                     # 약 정보 원문

g.close(); v.close()
```

| 함수 | 용도 |
|---|---|
| `DrugGraph.assess(current, candidate, age, pregnant)` | 최종 판정 `BAN / CAUTION / OK / UNKNOWN / NOT_FOUND` + 근거 + WEB 그래프 JSON |
| `DrugGraph.find_drug(query)` | 제품명·바코드·품목기준코드 → 제품 후보 |
| `DrugVectorDB.emergency_check(text)` | 응급 규칙 → `EMERGENCY / URGENT / NONE` |
| `DrugVectorDB.search_drugs(query)` | 증상 → 일반의약품 후보 (같은 약 1개로, `item_seq` 채움) |
| `DrugVectorDB.search_symptoms(query)` | 질환·위험 신호 설명 |
| `DrugVectorDB.explain_findings(assess_result)` | 판정 결과의 금기·주의마다 식약처 근거 문서 추가 |
| `DrugVectorDB.explain_ingredient(names=, dur_types=)` | 성분 단위 금기·주의 문서 |
| `DrugVectorDB.get_drug_info(item_seq= / name=)` | 효능·용법·주의·부작용·허가 원문 |
| `DrugVectorDB.search(query, doc_types, …)` | 범용 검색 (`section`, `otc_only`, `dur_types`, `item_seqs` 필터) |
| `DrugVectorDB.describe()` | doc_type별 건수 확인 |

### 3-2. 직접 SQL로 검색할 때

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large")
q = model.encode("query: 두통에 먹는 약", normalize_embeddings=True).tolist()
```

```sql
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = relaxed_order;      -- 추가 필터가 있어도 결과 수 보장 (pgvector 0.8+)

SELECT name, item_seq, section, text, source_name, source_url,
       1 - (embedding <=> %(q)s::vector) AS score
FROM drug_documents_v2
WHERE doc_type = 'drug_efficacy'              -- 필수: doc_type별 인덱스를 사용
ORDER BY embedding <=> %(q)s::vector
LIMIT 5;
```

- **`doc_type` 조건은 항상 넣습니다.** 벡터 인덱스가 doc_type별 부분 인덱스라, 빠지면 전체를 읽어 느려집니다.
- 일반의약품만: `drug_permit`에서 `AND is_otc AND is_active` (e약은요 문서는 원래 일반의약품이라 `is_otc`가 비어 있음)
- 금기 종류 좁히기: `dur_ingredient`에서 `AND dur_type = 'combination_ban'`
- 같은 약의 조각이 여러 개 나올 수 있으므로 `item_seq`(없으면 `name`)로 중복 제거

---

## 4. `drug_documents_v2` 테이블 스키마

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | SERIAL | PK |
| doc_id | TEXT | 조각 고유 ID (UNIQUE) |
| doc_type | TEXT | 문서 종류 (5장) |
| chunk_index | INTEGER | 같은 문서 안의 조각 번호 |
| section | TEXT | 세부 구분 (`efficacy` / `usage` / `caution` / `side_effect`, `drug_permit`은 `summary` / `caution`) |
| text | TEXT | 임베딩 대상 텍스트. 모든 조각 앞에 `약품명: …` / `질환명: …` 머리말 |
| embedding | vector(1024) | 임베딩 벡터 |
| name | TEXT | 약품명 / 질환명 / 금기 성분명(`A + B`) |
| item_seq | TEXT | 품목기준코드 (그래프 `Drug.item_seq`와 같은 코드). 약학정보원 문서는 비어 있음 |
| source_name | TEXT | 출처명 |
| source_url | TEXT | 출처 URL |
| license | TEXT | 라이선스 |
| provider | TEXT | 제공 기관 |
| item_url | TEXT | 상세 페이지 URL (식약처 의약품안전나라, 국가건강정보포털 등) |
| manufacturer | TEXT | 제조사 (약품) |
| image_url | TEXT | 약품 이미지 URL (e약은요) |
| dur_type | TEXT | 금기 유형 (`dur_ingredient`만, 6장) |
| level | TEXT | `ban` / `caution` (`dur_ingredient`만) |
| ingredient_a | TEXT | 금기 성분 A 한글명 (`dur_ingredient`만) |
| ingredient_b | TEXT | 금기 성분 B 한글명 (병용금기만) |
| is_otc | BOOLEAN | 일반의약품 여부 (`drug_permit`만) |
| is_active | BOOLEAN | 허가 유효 여부 (`drug_permit`만) |
| data_version | TEXT | 적재 회차 (내부 관리용) |
| metadata | JSONB | 원천별 전체 메타데이터 (5장) |
| created_at | TIMESTAMP | 적재 일시 |

**인덱스**: `doc_id` UNIQUE, `doc_type` / `dur_type` / `item_seq` B-tree, doc_type별 부분 HNSW(`vector_cosine_ops`, m=16, ef_construction=64)

---

## 5. doc_type 태그와 메타데이터

| doc_type | 설명 | 건수(조각) | 출처 | 주로 쓰는 곳 |
|---|---|---|---|---|
| `drug_efficacy` | 효능 (어디에 쓰는 약인가) | 11,831 | e약은요, 약학정보원 | 약품 추천 |
| `drug_usage` | 용법·용량 + 복용 스케줄 | 12,366 | e약은요, 약학정보원 | 복용 안내 |
| `drug_caution` | 주의사항·경고 | 18,494 | e약은요, 약학정보원 | 주의 안내 |
| `drug_side_effect` | 부작용 | 5,578 | e약은요, 약학정보원 | 부작용 안내 |
| `drug_permit` | 식약처 허가 기준 제품 정보 (`summary` / `caution`) | 103,132 / 60,004 | 의약품 허가정보 | 약품 추천, 복용 안내 |
| `dur_ingredient` | 성분 단위 금기·주의 7종 | 4,639 | DUR 성분정보 | **금기 근거 설명** |
| `symptom_info` | 질환 설명 | 13,228 | 국가건강정보포털, 약학정보원 | 증상 분석 |
| `symptom_redflag` | 병원·응급실에 가야 하는 경우 (문장 추출) | 324 | 국가건강정보포털 | 위험 안내 (보조용) |
| **합계** | | **229,596** | | |

- 제품 쌍 단위 DUR(약 82만 건)은 같은 금기의 반복이라 벡터 DB에 넣지 않고, 그래프의 `DUR_FLAGGED`로만 사용합니다.
- `symptom_redflag`는 보조 자료입니다. 응급 판단은 `emergency_check()`를 먼저 사용합니다.

### 5-1. drug_efficacy / drug_usage / drug_caution / drug_side_effect

```json
{
  "name": "타이레놀정500밀리그람(아세트아미노펜)",
  "item_seq": "202106092",
  "manufacturer": "한국존슨앤드존슨판매(유)",
  "image_url": "https://...",
  "source_name": "식품의약품안전처 e약은요",
  "source_url": "https://www.data.go.kr/data/15075057/openapi.do",
  "license": "공공누리 1유형",
  "provider": "식품의약품안전처",
  "doc_type": "drug_efficacy",
  "section": "efficacy",
  "parent_doc_id": "public_202106092",
  "etl_at": "2026-10-22T…"
}
```

- 약학정보원 문서는 `source_name: "약학정보원 의약품정보"`, `license: "비상업적 참조 목적"`이며 `item_seq`가 없습니다.

**drug_usage 전용 — `structured_usage`**

```json
{
  "structured_usage": {
    "age_groups": ["만 15세 이상", "성인"],
    "dose": {"amount": "1~2", "unit": "정", "raw": "1회 1~2정"},
    "frequency": {"times_per_day": 3, "raw": "1일 3회"},
    "timing": [{"type": "식후", "minutes": 30}],
    "interval": {"min_hours": 4},
    "max_days": {"max_days": 3},
    "max_daily_dose": {"amount": "8", "unit": "정"},
    "raw": "원본 텍스트"
  }
}
```

`timing.type` 값: `식후` / `식전` / `식간` / `취침전` / `공복` / `식사와함께` / `배변후` / `승차전` / `기상후`

### 5-2. drug_permit

```json
{
  "name": "알레틴정(로라타딘)",
  "name_eng": "…",
  "item_seq": "…",
  "manufacturer": "…",
  "etc_otc": "일반의약품",
  "is_otc": true,
  "is_active": true,
  "is_finished_drug": true,
  "main_ingr_raw": "…",
  "additives_raw": "…",
  "atc_code": "…",
  "bar_codes": ["880…"],
  "edi_codes": ["…"],
  "permit_date": "…",
  "cancel_name": "",
  "official_url": "https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq=…",
  "source_name": "식품의약품안전처 의약품 제품 허가정보",
  "source_url": "https://www.data.go.kr/data/15095677/openapi.do",
  "license": "공공누리 1유형",
  "provider": "식품의약품안전처"
}
```

- `section = summary`: 허가 유효한 완제의약품의 제품 요약 (구분·주성분·효능·용법)
- `section = caution`: 허가 유효한 완제 **일반의약품**의 사용상 주의사항 원문
- `bar_codes`로 약 상자 바코드 → 제품을 찾을 수 있습니다 (일반의약품의 약 95%).

### 5-3. dur_ingredient

```json
{
  "dur_type": "combination_ban",
  "dur_desc": "병용금기",
  "level": "ban",
  "ingr_code_a": "D000762",
  "ingredient_a_kor": "이트라코나졸",
  "ingredient_a_eng": "Itraconazole",
  "ingr_code_b": "D000027",
  "ingredient_b_kor": "심바스타틴",
  "ingredient_b_eng": "Simvastatin",
  "pair_key": "D000027|D000762",
  "form_name": "",
  "reason": "근병증의 위험 증가 가능",
  "remark": "",
  "notification_date": "…",
  "pregnancy_grade": null,
  "age_base": "",
  "max_age_months": null,
  "max_qty": "",
  "max_term": "",
  "effect_name": "",
  "source_name": "식품의약품안전처 DUR(의약품 안전사용 서비스) 성분정보",
  "source_url": "https://www.data.go.kr/data/15056780/openapi.do",
  "license": "공공누리 1유형",
  "provider": "식품의약품안전처"
}
```

텍스트 예시:

```
[병용금기] 성분: 이트라코나졸 + 심바스타틴 | 사유: 근병증의 위험 증가 가능
[임부금기] 성분: 프로게스테론 | 제형: 연질캡슐제 | 등급: 1등급 | 사유: 임부에 대한 안전성 미확립.
[특정연령대금기] 성분: 아카보즈 | 제형: 정제 | 연령기준: 18세 이하 | 사유: 안전성 및 유효성 미확립
```

### 5-4. symptom_info / symptom_redflag

```json
{
  "name": "심근경색증",
  "system_code": "SO",
  "system_name": "순환기",
  "item_url": "https://health.kdca.go.kr/...",
  "brd_sid": "…",
  "has_redflag": true,
  "is_redflag": true,
  "redflag_method": "sentence_rule_v4",
  "doc_type": "symptom_redflag",
  "parent_doc_id": "portal_SO_심근경색증_redflag",
  "source_name": "국가건강정보포털",
  "source_url": "https://health.kdca.go.kr",
  "license": "공공누리",
  "provider": "질병관리청"
}
```

- `symptom_redflag`: 질환 설명에서 "병원·응급실에 가야 하는 경우" 문장만 규칙으로 뽑아 모은 문서 (`질환명: … | [위험 신호] …`)
- `symptom_info`: 질환 설명 전체 (개요·증상·진단·치료·예방 등)

**system_code 값**

| system_code | system_name | | system_code | system_name |
|---|---|---|---|---|
| `NE` | 뇌신경 | | `HH` | 호흡기 |
| `JU` | 정신건강 | | `SO` | 순환기 |
| `NU` | 눈 | | `SW` | 소화기 |
| `KO` | 귀코목 | | `MH` | 면역 |
| `KU` | 구강 | | `SA` | 비뇨기 |
| `BB` | 뼈근육 | | `SS` | 생식기 |
| `PB` | 피부 | | `ETC` | 기타 (신체계통 미지정 질환) |
| `NB` | 내분비 | | `EM` | 응급 (응급상황 대처, 심폐소생술) |

- 수집 범위: 질환 약 670개 + 심근경색증·뇌졸중·뇌졸중 이후의 삶·응급상황 대처·심폐소생술 페이지

---

## 6. dur_type 값

| dur_type | 설명 | level |
|---|---|---|
| `combination_ban` | 병용금기 | ban |
| `pregnancy_ban` | 임부금기 (`pregnancy_grade` 1등급이 가장 엄격) | ban |
| `age_specific_ban` | 특정연령대금기 (`age_base`, `max_age_months`) | ban |
| `elderly_caution` | 노인주의 (약 77%는 사유 없음) | caution |
| `dose_caution` | 용량주의 (`max_qty`) | caution |
| `dosing_period_caution` | 투여기간주의 (`max_term`) | caution |
| `effect_duplicate` | 효능군중복 (`effect_name`) | caution |

| dur_type | 건수 |
|---|---|
| combination_ban | 1,769 |
| pregnancy_ban | 1,428 |
| dose_caution | 637 |
| effect_duplicate | 375 |
| age_specific_ban | 229 |
| elderly_caution | 106 |
| dosing_period_caution | 95 |

---

## 7. 그래프 스키마 (Neo4j)

```
           (임부 / 소아·청소년 / 노인)
                  PatientGroup
                       ▲
                       │ RESTRICTED_FOR (성분 기준 제한)
                       │ DUR_FLAGGED   (식약처가 제품별로 지정)
                       │
 Drug ──CONTAINS──▶ Ingredient ◀──CONTRAINDICATED_WITH──▶ Ingredient ◀──CONTAINS── Drug
(제품)              (성분)         (병용금기)                (성분)                 (제품)
                       │
                       └──BELONGS_TO──▶ EffectGroup (효능군)
```

| 구분 | 이름 | 개수 | 주요 속성 |
|---|---|---|---|
| 노드 | `Drug` | 42,746 | `item_seq`, `name`, `etc_otc`, `is_otc`, `is_active`, 바코드, 식약처 링크 |
| 노드 | `Ingredient` | 5,626 | `code`(D코드), `name_kor`, `name_eng`, 1일 최대투여량, 최대투여기간 |
| 노드 | `PatientGroup` | 3 | 임부, 소아·청소년, 노인 |
| 노드 | `EffectGroup` | 9 | 효능군 |
| 관계 | `CONTAINS` | 89,912 | 제품 → 성분 |
| 관계 | `CONTRAINDICATED_WITH` | 1,250 | 성분 ↔ 성분 병용금기 (`reasons`, 시간 조건, 출처) |
| 관계 | `RESTRICTED_FOR` | 1,483 | 성분 → 환자군 (`level`, 등급, 기준 나이) |
| 관계 | `DUR_FLAGGED` | 19,923 | 제품 → 환자군 (서방형 전용 금기 등 제품별 지정) |
| 관계 | `BELONGS_TO` | 374 | 성분 → 효능군 |

- 유효 완제 일반의약품 8,566개 중 8,510개(99.3%)가 성분까지 연결되어 금기 검사가 가능합니다. 나머지는 `UNKNOWN`으로 판정됩니다.
- 금기 판정은 반드시 `assess()`로 합니다. 벡터 검색 결과로 판정하지 않습니다.

---

## 8. 검색 규칙

| 규칙 | 이유 |
|---|---|
| 1. 질문을 의학 용어로 바꿔 **원문과 함께** 검색 (필수) | "속이 쓰리고 신물이 올라와요" 단독 → 관련 0%, "위산 역류 속쓰림 가슴쓰림" → 100% |
| 2. 금기 검색은 `dur_type`으로 좁히기 | 필터 없이 병용금기를 물으면 임부금기 문서가 위에 나옴 |
| 3. 금기 판정은 `assess()`만 | 의미 검색은 약 분류를 정확히 구분하지 못함 |
| 4. 추천은 일반의약품만 (`otc_only=True`) | 필터 없으면 전문의약품이 섞임 |
| 5. 응급은 `emergency_check()` 먼저 | 검색은 놓칠 수 있음 |

- 유사도 점수는 대부분 0.83~0.91에 몰려 있습니다. 고정 기준점("0.85 이상")보다 **순위**로 사용하세요.

---

## 9. 출처 및 라이선스

| 출처 | 라이선스 | 상업 이용 | 쓰는 곳 |
|---|---|---|---|
| 식품의약품안전처 e약은요 | 공공누리 1유형 | ✅ 가능 | 약 문서 |
| 식품의약품안전처 의약품 제품 허가정보 | 공공누리 1유형 | ✅ 가능 | 그래프 제품·성분, `drug_permit` |
| 식품의약품안전처 DUR 성분정보 | 공공누리 1유형 | ✅ 가능 | 그래프 금기, `dur_ingredient` |
| 식품의약품안전처 DUR 품목정보 | 공공누리 1유형 | ✅ 가능 | 그래프 `DUR_FLAGGED` |
| 국가건강정보포털 (질병관리청) | 공공누리 4유형 | ⚠️ 출처표시 + 비상업 + 변경금지 | `symptom_*` |
| 약학정보원 | 비상업적 참조 목적 | ⚠️ 비상업 한정 | 약 문서, `symptom_info` 일부 |

- 답변에는 `source_name`과 `source_url`(또는 `item_url`)을 함께 표시합니다.

---

## 10. S3 데이터 레이크 구조

```
s3://drug-rag-datalake-331145994962/
├── bronze/                                   ← 원본 (API 응답·크롤링 결과)
│   ├── api/
│   │   ├── public_data/YYYY-MM-DD/           ← e약은요
│   │   ├── drug_permit/{op}/YYYY-MM-DD/      ← 의약품 허가정보 (제품·주성분)
│   │   ├── dur/YYYY-MM-DD/                   ← DUR 품목정보
│   │   └── dur_ingredient/{dur_type}/YYYY-MM-DD/  ← DUR 성분정보
│   └── crawl/
│       ├── health_portal/YYYY-MM-DD/         ← 국가건강정보포털 ({system_code}_{brd_sid}.json)
│       └── health_kr/YYYY-MM-DD/             ← 약학정보원
├── silver/                                   ← 정제 JSON ({"count", "items": [...]})
│   ├── drugs/
│   │   ├── public_drug/                      ← e약은요 (+ structured_usage)
│   │   ├── health_kr_drug/                   ← 약학정보원
│   │   └── drug_permit/                      ← 허가정보 제품
│   ├── drug_ingredients/                     ← 제품별 주성분
│   ├── dur_ingredient/{dur_type}/            ← DUR 성분정보 7종
│   ├── dur_ingredient/code_map/              ← 원료코드 → DUR 성분코드 매핑
│   ├── dur_interactions/{dur_type}/          ← DUR 품목정보 7종 (제품 쌍)
│   ├── symptoms/
│   │   ├── health_portal/
│   │   └── health_kr_disease/
│   └── refined/                              ← doc_type별로 나눈 문서 (임베딩 입력)
│       ├── drugs/                            ← {source}_batch_NNNN.json
│       └── symptoms/
└── gold/
    ├── vector_{RUN_ID}/parts/                ← 임베딩 체크포인트 (parquet)
    └── pgvector_dump/
        └── 2026-10-22/
            └── drug_rag_v3.dump              ← 현재 벡터 DB 덤프 (PostgreSQL 15 형식)
```

- 각 경로 아래에 `YYYY-MM-DD` 날짜 폴더가 있습니다. 현재 기준은 `2026-10-22`이며, ETL은 실행 날짜(`RUN_DATE`) 폴더만 읽습니다.
- 약학정보원(`health_kr`)은 2026-09-01 수집본을 그대로 사용합니다.
- S3 데이터는 서비스 개발에 필요하지 않습니다. 필요하면 DE 파트에 다운로드 링크를 요청하세요.
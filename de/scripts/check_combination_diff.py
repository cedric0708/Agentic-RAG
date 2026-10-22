# check_combination_diff.py — 같은 성분 쌍 안에서 어떤 필드가 다른지 집계
import sys
from collections import defaultdict, Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env", override=True)  # ① 먼저 로드
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import list_keys, download_json                                                  # ② 그 다음 import

FIELDS = ["reason", "remark", "mix_ingr_a", "mix_ingr_b", "mix_type_a", "mix_type_b", "notification_date"]

groups = defaultdict(list)
for k in list_keys("silver/dur_ingredient/combination_ban/"):
    if k.endswith(".json"):
        for d in download_json(k)["items"]:
            m = d["metadata"]
            groups[(m["ingr_code_a"], m["ingr_code_b"])].append(m)

diff_cnt, examples = Counter(), {}
multi = {k: v for k, v in groups.items() if len(v) > 1}
for pair, rows in multi.items():
    for f in FIELDS:
        if len({r.get(f, "") for r in rows}) > 1:
            diff_cnt[f] += 1
            examples.setdefault(f, (pair, [r.get(f, "") for r in rows][:3]))

print(f"레코드가 2건 이상인 성분쌍: {len(multi)}개")
print("서로 다른 필드별 성분쌍 수:", dict(diff_cnt))
for f, (pair, vals) in examples.items():
    print(f"\n[{f}] 예시 {pair}")
    for v in vals:
        print("   -", v)
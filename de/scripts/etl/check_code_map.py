# check_code_map.py
import sys
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / "config" / ".env", override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import list_keys, download_json

m2d = defaultdict(set); d_name = {}
for k in list_keys("silver/dur_ingredient/code_map/"):
    if k.endswith(".json"):
        for r in download_json(k)["items"]:
            m2d[r["src_code"]].add(r["dur_code"])
            d_name[r["dur_code"]] = r.get("dur_name_kor") or r.get("dur_name_eng")

amb = {m: ds for m, ds in m2d.items() if len(ds) > 1}
print(f"원료코드 {len(m2d)}종 중 다중 매핑 {len(amb)}종")
print("D코드 개수 분포:", sorted({len(ds) for ds in amb.values()}))
for m, ds in list(amb.items())[:10]:
    print(f"  {m} → " + " / ".join(f"{d}({d_name.get(d)})" for d in sorted(ds)))
    

# check_code_map.py 아래에 이어서 실행
from collections import Counter
src_name = {}
for k in list_keys("silver/dur_ingredient/code_map/"):
    if k.endswith(".json"):
        for r in download_json(k)["items"]:
            src_name[r["src_code"]] = r.get("src_name", "")

print("\n코드 종류별 다중 매핑:", dict(Counter(m[0] for m in amb)))
print("코드 종류별 전체:     ", dict(Counter(m[0] for m in m2d)))
for prefix in ("M", "I"):
    print(f"\n[{prefix} 코드 예시]")
    for m, ds in [(m, ds) for m, ds in amb.items() if m.startswith(prefix)][:10]:
        print(f"  {m}({src_name.get(m)}) → " + " / ".join(f"{d}({d_name.get(d)})" for d in sorted(ds)))
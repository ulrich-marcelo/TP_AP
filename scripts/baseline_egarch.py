import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from arch import arch_model
from src.data_loader import load_returns_reference


DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
OUT_PATH = ROOT / "results" / "egarch_ged_params_by_fold.csv"

df_ref = load_returns_reference(DATA_PATH, phase="phase_a")
r_pct = df_ref["BTC_r_cc"] * 100.0

rows = []
for f in range(14):
    r_tr = r_pct.iloc[f * 182 : f * 182 + 1095]
    res = arch_model(r_tr, mean="Constant", vol="EGARCH", p=1, o=1, q=3, dist="ged").fit(disp="off")
    rows.append({**res.params.to_dict(), "fold": f})

df_p = pd.DataFrame(rows)
df_p.to_csv("results/egarch_ged_params_by_fold.csv", index=False)
print(df_p.agg(["mean", "std", "median"]).T.round(4))
print("\nFold 0:\n", df_p.iloc[0].round(4))
print("\nFold 13:\n", df_p.iloc[13].round(4))
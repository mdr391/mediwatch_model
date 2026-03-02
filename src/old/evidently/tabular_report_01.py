import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset  # optional extra context

URL = "https://zvibo.com/data/diabetic_data.csv"
df = pd.read_csv(URL, na_values=["?"])

# --- 1) Sort by time proxy and create two windows ---
df["encounter_id"] = pd.to_numeric(df["encounter_id"], errors="coerce")
df = df.dropna(subset=["encounter_id"]).sort_values("encounter_id")

# Example: first 70% reference, last 30% current
cut = int(len(df) * 0.70)
ref_raw = df.iloc[:cut].copy()
cur_raw = df.iloc[cut:].copy()

# --- 2) Optional: remove patient leakage between windows ---
# Assign each patient to the window of their earliest encounter, drop their rows from the other window.
first_enc = df.groupby("patient_nbr", dropna=False)["encounter_id"].min()
ref_patients = set(first_enc[first_enc <= ref_raw["encounter_id"].max()].index)
cur_patients = set(first_enc[first_enc > ref_raw["encounter_id"].max()].index)

ref = ref_raw[ref_raw["patient_nbr"].isin(ref_patients)].copy()
cur = cur_raw[cur_raw["patient_nbr"].isin(cur_patients)].copy()

# --- 3) Choose monitored input columns ---
# Keep this to "model inputs" you care about. Exclude identifiers and target-like fields.
excluded = {"patient_nbr", "readmitted"}  # add more if you want
all_cols = [c for c in df.columns if c not in excluded]

# Identify numerics you want treated as numeric (many are counts/ids stored as ints/strings in some pipelines)
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
for c in numeric_cols:
    if c in ref.columns:
        ref[c] = pd.to_numeric(ref[c], errors="coerce")
        cur[c] = pd.to_numeric(cur[c], errors="coerce")

# Everything else (strings, coded categories, diag_1/2/3, etc.) will be categorical by explicit mapping.
categorical_cols = [c for c in all_cols if c not in set(numeric_cols) and c != "encounter_id"]

definition = DataDefinition(
    id_column="encounter_id",                 # ignored in drift calculations :contentReference[oaicite:3]{index=3}
    numerical_columns=[c for c in numeric_cols if c in ref.columns],
    categorical_columns=[c for c in categorical_cols if c in ref.columns],
)

ref_ds = Dataset.from_pandas(ref, data_definition=definition)
cur_ds = Dataset.from_pandas(cur, data_definition=definition)

# --- 4) Run drift ---
report = Report([
    DataDriftPreset(),                        # default drift logic per column type :contentReference[oaicite:4]{index=4}
    DataSummaryPreset(),                    # optional: side-by-side stats context :contentReference[oaicite:5]{index=5}
], include_tests=True)

snap = report.run(cur_ds, ref_ds)

# Notebook: render by evaluating `snap` in a cell (Jupyter/Colab). :contentReference[oaicite:6]{index=6}
snap.save_html("diabetes_input_drift.html")   # save/shareable HTML :contentReference[oaicite:7]{index=7}
# snap.json(), snap.dict(), snap.save_json("diabetes_input_drift.json") also available :contentReference[oaicite:8]{index=8}

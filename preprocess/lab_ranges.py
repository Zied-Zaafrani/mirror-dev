"""
Physiologically valid ranges for 18 core clinical labs used for outlier clipping and binning.

NOTE: MIRROR trains on up to 200 labs (controlled by --num_labs). This file covers only the
18 core labs that have validated clinical reference ranges. Non-core labs fall back to
bin=2 (normal) since no clinical threshold is defined.

Two-tier outlier handling following MIMIC-Extract (Wang et al., 2020):
  - Tier 1: value outside [OUTLIER_LOW, OUTLIER_HIGH] → NaN (data entry error)
  - Tier 2: value outside [VALID_LOW, VALID_HIGH] → clip (extreme but real)

Sources:
  - 16/18 from MIMIC-Extract variable_ranges.csv
  - INR: clinical derivation (not tracked by MIMIC-Extract)
  - Calcium: clinical derivation (blank in MIMIC-Extract)
"""

# Each entry: (ITEMID, name, OUTLIER_LOW, VALID_LOW, VALID_HIGH, OUTLIER_HIGH, unit)
LAB_RANGES = {
    # Kidney
    50912: ("Creatinine",       0,   0.1,    60,   66,   "mg/dL"),
    51006: ("BUN",              0,   0,     250,  275,   "mg/dL"),
    # Liver
    50861: ("ALT",              0,   2,   10000, 11000,  "IU/L"),
    50878: ("AST",              0,   6,   20000, 22000,  "IU/L"),
    50885: ("Bilirubin",        0,   0.1,    60,   66,   "mg/dL"),
    50863: ("Alk Phos",         0,  20,    3625,  4000,  "IU/L"),
    # Coagulation
    51237: ("INR",              0,   0.5,    20,   50,   "ratio"),
    51274: ("PT",               0,   9.9,  97.1,  150,   "seconds"),
    51275: ("PTT",              0,  18.8,   150,  150,   "seconds"),
    # Electrolytes
    50983: ("Sodium",           0,  50,     225,  250,   "mEq/L"),
    50971: ("Potassium",        0,   0,      12,   15,   "mEq/L"),
    50960: ("Magnesium",        0,   0,      20,   22,   "mg/dL"),
    50893: ("Calcium",          0,   4.0,    20,   40,   "mg/dL"),
    # Metabolic
    50931: ("Glucose",          0,  33,    2000, 2200,   "mg/dL"),
    50862: ("Albumin",          0,   0.6,     6,   60,   "g/dL"),
    50813: ("Lactate",          0,   0.4,    30,   33,   "mmol/L"),
    51301: ("WBC",              0,   0,    1000, 1100,   "K/uL"),
    51222: ("Hemoglobin",       0,   0,      25,   30,   "g/dL"),
}

# Ordered list of ITEMIDs (determines position in the 18-dim vector)
LAB_ITEMIDS = [
    50912, 51006,                        # Kidney
    50861, 50878, 50885, 50863,          # Liver
    51237, 51274, 51275,                 # Coagulation
    50983, 50971, 50960, 50893,          # Electrolytes
    50931, 50862, 50813, 51301, 51222,   # Metabolic
]

# Name lookup for logging / display
ITEMID_TO_NAME = {iid: LAB_RANGES[iid][0] for iid in LAB_ITEMIDS}

NUM_LABS = len(LAB_ITEMIDS)  # 18
LAB_DIM_STATIC = NUM_LABS * 2  # 36  (value + flag)
LAB_DIM_TRENDS = NUM_LABS * 4  # 72  (value + flag + slope + variance)


def clip_lab_value(itemid: int, value: float) -> float | None:
    """Apply two-tier outlier handling to a single lab value.

    Returns:
        float: Clipped value (Tier 2) or original value if within valid range.
        None:  If value is outside outlier bounds (Tier 1 → treat as missing).
    """
    _, outlier_lo, valid_lo, valid_hi, outlier_hi, _ = LAB_RANGES[itemid]

    # Tier 1: physiologically impossible → remove
    if value < outlier_lo or value > outlier_hi:
        return None

    # Tier 2: extreme but possible → clip
    if value < valid_lo:
        return valid_lo
    if value > valid_hi:
        return valid_hi

    return value

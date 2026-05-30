import pandas as pd
import argparse
from pathlib import Path

def analyze_lab_coverage(mimic_dir: Path):
    print("Loading LABEVENTS...")
    labevents = pd.read_csv(
        mimic_dir / "LABEVENTS.csv.gz",
        usecols=["HADM_ID", "ITEMID", "VALUENUM"],
        compression="gzip"
    ).dropna(subset=["HADM_ID", "VALUENUM"])
    
    print("Loading D_LABITEMS...")
    d_labitems = pd.read_csv(
        mimic_dir / "D_LABITEMS.csv.gz",
        usecols=["ITEMID", "LABEL", "FLUID", "CATEGORY"],
        compression="gzip"
    )
    
    # Total distinct admissions
    total_admissions = labevents["HADM_ID"].nunique()
    print(f"Total admissions with labs: {total_admissions}")
    
    # Coverage per itemid: number of unique admissions with at least one reading
    coverage = labevents.groupby("ITEMID")["HADM_ID"].nunique().reset_index()
    coverage.columns = ["ITEMID", "COVERAGE_COUNT"]
    coverage["COVERAGE_PCT"] = (coverage["COVERAGE_COUNT"] / total_admissions) * 100
    
    # Mean and std per itemid
    stats = labevents.groupby("ITEMID")["VALUENUM"].agg(["mean", "std"]).reset_index()
    
    # Merge all
    df = coverage.merge(stats, on="ITEMID").merge(d_labitems, on="ITEMID", how="left")
    df = df.sort_values("COVERAGE_PCT", ascending=False)
    
    print("\n--- Top 60 Labs by Coverage ---")
    for idx, row in df.head(60).iterrows():
        print(f"[{row['ITEMID']}] {row['LABEL'][:30]:<30} | {row['FLUID'][:10]:<10} | {row['COVERAGE_PCT']:>5.1f}% | mean: {row['mean']:>8.2f} ± {row['std']:>8.2f}")
    
    current_18 = {50868, 50882, 50893, 50902, 50912, 50931, 50960, 50970, 50971, 50983, 51006, 51144, 51221, 51222, 51248, 51265, 51277, 51279}
    
    print("\n--- Missing from Current 18 (>30% coverage) ---")
    missing = df[~df["ITEMID"].isin(current_18) & (df["COVERAGE_PCT"] > 30.0)]
    for idx, row in missing.iterrows():
        print(f"[{row['ITEMID']}] {row['LABEL'][:30]:<30} ({row['COVERAGE_PCT']:>5.1f}%)")
        
    print("\n--- Lab Sets ---")
    print("top_5 =", df["ITEMID"].head(5).tolist())
    print("top_10 =", df["ITEMID"].head(10).tolist())
    print("top_18 =", df["ITEMID"].head(18).tolist())
    print("top_30 =", df["ITEMID"].head(30).tolist())
    print("top_50 =", df["ITEMID"].head(50).tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic3_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_lab_coverage(Path(args.mimic3_dir))

"""
MIRROR Lab Configuration Switcher
===================================
Switch the active lab configuration by selecting a pre-built lab_vectors pkl.

Usage:
    python src/scripts/switch_labs.py --num_labs 200
    python src/scripts/switch_labs.py --list
    python src/scripts/switch_labs.py --status

The script:
  1. Validates that lab_vectors_{N}labs.pkl exists in processed/
  2. Updates config.yaml:  lab_dim = N*2,  lab_dim_trends = N*4
  3. Prints the exact --lab_key argument to pass to train.py

It does NOT copy or symlink files — train.py reads the pkl by name directly.

NO DEFAULT lab count — you must pass --num_labs.

MIMIC-IV note:
    MIMIC-IV lab pkls follow the same naming convention.
    If your processed/ dir contains both MIMIC-III and MIMIC-IV pkls,
    use --pkl_dir to point at the right one.
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("switch_labs")

# ── Path resolution ────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
MIRROR_ROOT   = SCRIPT_DIR.parents[2]
PROCESSED_DIR = MIRROR_ROOT / "processed"
CONFIG_PATH   = MIRROR_ROOT / "src" / "config.yaml"
LABS_DIR      = PROCESSED_DIR / "labs"

logger.info("MIRROR_ROOT   = %s", MIRROR_ROOT)
logger.info("PROCESSED_DIR = %s", PROCESSED_DIR)
logger.info("CONFIG_PATH   = %s", CONFIG_PATH)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_lab_configs(pkl_dir: Path) -> dict:
    """
    Scan pkl_dir for lab_vectors_*.pkl files.

    Returns:
        dict mapping lab_count (int) → {pkl_path, lab_key, folder_name, is_max, num_labs}
    """
    configs = {}
    for p in sorted(pkl_dir.glob("lab_vectors_*.pkl")):
        name = p.stem
        body = name.replace("lab_vectors_", "")
        is_max = "_MAX" in body.upper()
        num_str = body.replace("labs", "").replace("_MAX", "").replace("MAX", "")
        try:
            n = int(num_str)
        except ValueError:
            logger.debug("Skipping unrecognised pkl: %s", p.name)
            continue
        configs[n] = {
            "pkl_path":    p,
            "pkl_name":    p.name,
            "folder_name": f"max_{n}" if is_max else f"top_{n}",
            "is_max":      is_max,
            "num_labs":    n,
            "lab_dim":     n * 2,   # standard: z-scores + flags
            "lab_dim_trends": n * 4,  # trend variant: + slopes + variances
        }
    return configs


def check_embeddings_exist(n: int, labs_dir: Path) -> dict:
    """Check which embedding files exist for lab count n."""
    for folder_name in [f"top_{n}", f"max_{n}"]:
        folder = labs_dir / folder_name
        has_desc = (folder / "lab_description_embeddings.npy").exists()
        has_text = (folder / "lab_text_embeddings.pt").exists()
        has_manifest = (folder / "manifest.json").exists()
        if has_manifest or has_desc or has_text:
            return {
                "folder": folder,
                "has_description_embeddings": has_desc,
                "has_text_embeddings":        has_text,
                "has_manifest":               has_manifest,
                "all_present":                has_desc and has_text and has_manifest,
            }
    return {
        "folder": None,
        "has_description_embeddings": False,
        "has_text_embeddings":        False,
        "has_manifest":               False,
        "all_present":                False,
    }


# ── Config.yaml update ────────────────────────────────────────────────────────

def update_config_yaml(config_path: Path, lab_dim: int, lab_dim_trends: int, num_labs: int):
    """Update lab_dim and lab_dim_trends in config.yaml."""
    if not config_path.exists():
        logger.warning("config.yaml not found at %s — skipping config update.", config_path)
        return

    with open(config_path, "r") as f:
        lines = f.readlines()

    updated = []
    changed = {"lab_dim": False, "lab_dim_trends": False}
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("lab_dim:") and "trends" not in line:
            updated.append(f"  lab_dim: {lab_dim}              # {num_labs} labs × 2 (z-scores + flags)\n")
            changed["lab_dim"] = True
            logger.info("  config.yaml: lab_dim=%d", lab_dim)
        elif stripped.startswith("lab_dim_trends:"):
            updated.append(f"  lab_dim_trends: {lab_dim_trends}         # {num_labs} labs × 4 (+ slope + variance)\n")
            changed["lab_dim_trends"] = True
            logger.info("  config.yaml: lab_dim_trends=%d", lab_dim_trends)
        else:
            updated.append(line)

    if not changed["lab_dim"]:
        logger.warning("  lab_dim key not found in config.yaml — may need manual update.")
    if not changed["lab_dim_trends"]:
        logger.warning("  lab_dim_trends key not found in config.yaml — may need manual update.")

    with open(config_path, "w") as f:
        f.writelines(updated)

    logger.info("  config.yaml updated: %s", config_path)


# ── Main switch logic ──────────────────────────────────────────────────────────

def switch_labs(num_labs: int, pkl_dir: Path, config_path: Path, labs_dir: Path):
    """Switch to a specific lab configuration."""
    configs = discover_lab_configs(pkl_dir)

    if num_labs not in configs:
        available = sorted(configs.keys())
        logger.critical(
            "--num_labs %d not found in %s\n"
            "Available lab counts: %s\n\n"
            "To generate missing pkl files, run:\n"
            "  python src/scripts/generate_phase9_lab_pkls.py\n\n"
            "To see current status:\n"
            "  python src/scripts/switch_labs.py --list",
            num_labs, pkl_dir, available,
        )
        sys.exit(1)

    cfg = configs[num_labs]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Switching to %d labs", num_labs)
    logger.info("=" * 60)
    logger.info("  pkl file       : %s  (%.1f MB)",
                cfg["pkl_name"], cfg["pkl_path"].stat().st_size / 1e6)
    logger.info("  lab_dim        : %d  (num_labs × 2)", cfg["lab_dim"])
    logger.info("  lab_dim_trends : %d  (num_labs × 4)", cfg["lab_dim_trends"])

    # Check embeddings
    emb_status = check_embeddings_exist(num_labs, labs_dir)
    if emb_status["all_present"]:
        logger.info("  embedding folder: %s  ✓ complete", emb_status["folder"])
    else:
        logger.warning(
            "  ⚠  Embedding files NOT found for %d labs in %s\n"
            "  Missing: %s\n"
            "  These are needed for per_lab_attn and lab_as_text encoders.\n"
            "  To generate them:\n"
            "    python src/scripts/precompute_lab_embeddings.py --num_labs %d",
            num_labs, labs_dir,
            [
                k for k, v in emb_status.items()
                if k.startswith("has_") and not v
            ],
            num_labs,
        )

    # Update config.yaml
    update_config_yaml(config_path, cfg["lab_dim"], cfg["lab_dim_trends"], num_labs)

    # Print the exact train.py command
    logger.info("")
    logger.info("─" * 60)
    logger.info("✓  Configuration set. Run training with:")
    logger.info("")
    logger.info("   python src/train.py \\")
    logger.info("     --num_labs %d \\", num_labs)
    logger.info("     --lab_key lab_vectors \\")
    logger.info("     --lab_file processed/lab_vectors_%dlabs.pkl",
                num_labs if not cfg["is_max"] else f"{num_labs}_MAX")
    logger.info("")
    logger.info("─" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_labs", type=int,
        help="Number of labs to switch to (must match an existing pkl).",
    )
    group.add_argument(
        "--list", action="store_true",
        help="List all available lab configurations and their embedding status.",
    )
    group.add_argument(
        "--status", action="store_true",
        help="Show status of all discovered configs (pkl ✓/✗, embeddings ✓/✗).",
    )
    p.add_argument(
        "--pkl_dir", type=Path, default=PROCESSED_DIR,
        help=f"Directory containing lab_vectors_*.pkl files. Default: {PROCESSED_DIR}",
    )
    p.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help=f"Path to config.yaml. Default: {CONFIG_PATH}",
    )
    p.add_argument(
        "--labs_dir", type=Path, default=LABS_DIR,
        help=f"Root directory for labs/ embedding folders. Default: {LABS_DIR}",
    )
    return p.parse_args()


def main():
    args = parse_args()

    configs = discover_lab_configs(args.pkl_dir)

    if not configs:
        logger.critical(
            "No lab_vectors_*.pkl files found in %s\n"
            "Expected files like: lab_vectors_200labs.pkl, lab_vectors_446_MAXlabs.pkl",
            args.pkl_dir,
        )
        sys.exit(1)

    if args.list or args.status:
        print(f"\nLab configurations available in {args.pkl_dir}:")
        print(f"{'N':>6}  {'pkl':35}  {'pkl MB':>7}  {'embeddings':12}  folder")
        print("─" * 80)
        for n in sorted(configs):
            cfg = configs[n]
            size_mb = cfg["pkl_path"].stat().st_size / 1e6
            emb = check_embeddings_exist(n, args.labs_dir)
            emb_str = "✓ complete" if emb["all_present"] else "✗ missing"
            print(f"  {n:>4}  {cfg['pkl_name']:35}  {size_mb:6.1f}  {emb_str:12}  {cfg['folder_name']}")

        print()
        missing_emb = [n for n in sorted(configs) if not check_embeddings_exist(n, args.labs_dir)["all_present"]]
        if missing_emb:
            print("⚠  Missing embeddings for:", missing_emb)
            print("   Generate them with:")
            print("   python src/scripts/precompute_lab_embeddings.py --all")
        sys.exit(0)

    switch_labs(args.num_labs, args.pkl_dir, args.config, args.labs_dir)


if __name__ == "__main__":
    main()

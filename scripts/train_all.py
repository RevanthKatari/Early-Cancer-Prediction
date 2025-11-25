"""
Master training script to retrain all models with proper preprocessing.
Run this script to regenerate all model artifacts.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

scripts = [
    "train_cervical.py",
    "train_brain.py",
    "train_oral.py",
]

print("=" * 60)
print("Training All Cancer Prediction Models")
print("=" * 60)

for script in scripts:
    print(f"\n{'=' * 60}")
    print(f"Training: {script}")
    print(f"{'=' * 60}\n")
    
    script_path = SCRIPTS_DIR / script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error training {script}")
        sys.exit(1)
    else:
        print(f"\n✅ Successfully trained {script}")

print("\n" + "=" * 60)
print("All models trained successfully!")
print("=" * 60)


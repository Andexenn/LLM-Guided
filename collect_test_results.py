#!/usr/bin/env python3
"""Collect all TEST lines from final_test_evaluation.log across all ablation combinations."""

import os
import glob

LOG_ROOT = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/logs/mmist/v1.2"
OUTPUT_FILE = os.path.join(LOG_ROOT, "all_test_results.log")

# Find all final_test_evaluation.log files
log_files = sorted(glob.glob(os.path.join(LOG_ROOT, "*/final_test_evaluation.log")))

test_lines = []
missing = []

for log_file in log_files:
    combo_name = os.path.basename(os.path.dirname(log_file))
    with open(log_file, "r") as f:
        found = False
        for line in f:
            if "TEST |" in line:
                test_lines.append(line.strip())
                found = True
        if not found:
            missing.append(combo_name)

# Write collected results
with open(OUTPUT_FILE, "w") as f:
    f.write(f"# Synapse Ablation Study — Test Results (collected from {len(log_files)} combinations)\n")
    f.write(f"# Format: [timestamp] TEST | <modality>_<method> | AUC | BACC | F1 | Acc\n\n")
    for line in test_lines:
        f.write(line + "\n")
    if missing:
        f.write(f"\n# WARNING: No TEST line found in: {', '.join(missing)}\n")

print(f"Collected {len(test_lines)} TEST results from {len(log_files)} log files → {OUTPUT_FILE}")
if missing:
    print(f"WARNING: Missing TEST results for: {', '.join(missing)}")

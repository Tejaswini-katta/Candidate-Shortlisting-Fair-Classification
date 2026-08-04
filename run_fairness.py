"""
Step 5: Fairness & Bias Analysis Runner.

Loads trained model and processed validation dataset, computes baseline subgroup
fairness metrics across Gender, applies threshold optimization bias mitigation,
saves fairness reports, and prints comparison summary.
"""

import os
from src.fairness import run_fairness_audit


def main():
    results = run_fairness_audit()

    print("\n" + "=" * 65)
    print("FAIRNESS & BIAS MITIGATION SUMMARY REPORT")
    print("=" * 65)

    raw_s = results["raw_summary"]
    mit_s = results["mitigated_summary"]

    print(f"{'Fairness Metric':<32} | {'Raw Model':<12} | {'Fair Model':<12} | {'Improvement':<10}")
    print("-" * 72)

    for metric in ["Demographic Parity Difference", "Equal Opportunity Difference", "Equalized Odds Difference"]:
        raw_val = raw_s[metric]
        mit_val = mit_s[metric]
        improvement = ((raw_val - mit_val) / (raw_val + 1e-7)) * 100
        print(f"{metric:<32} | {raw_val:<12.4f} | {mit_val:<12.4f} | {improvement:<9.1f}%")

    print("=" * 65)
    print("Group-Specific Optimized Thresholds:")
    for group, thresh in results["fair_thresholds"].items():
        print(f"   - Group '{group}': Threshold = {thresh:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

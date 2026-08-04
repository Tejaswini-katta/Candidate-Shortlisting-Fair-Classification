"""
Runner script to execute Step 7 Candidate Probability Ranking Engine.
"""

import os
from src.ranking import run_ranking_pipeline


def main():
    results = run_ranking_pipeline()

    print("\n" + "=" * 65)
    print("CANDIDATE SHORTLISTING & PRIORITY TIER REPORT")
    print("=" * 65)
    print(f"Total Test Candidates Evaluated: {results['total_candidates']}")
    print("-" * 65)
    print(f"{'Priority Tier':<25} | {'Candidate Count':<18} | {'Percentage':<12}")
    print("-" * 65)

    tier_order = ["High Priority", "Qualified", "Extended", "Reserve"]
    total = results['total_candidates']
    for tier in tier_order:
        count = results['tier_counts'].get(tier, 0)
        pct = (count / total) * 100
        print(f"{tier:<25} | {count:<18} | {pct:<11.2f}%")

    print("=" * 65)
    print("\nTop 5 Highest Probability Candidates:")
    sample = results['ranking_df'].head(5)
    print(sample.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()

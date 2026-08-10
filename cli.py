#!/usr/bin/env python3
"""
CLI for the AI-powered resume screening system.

Usage:
    python cli.py --job sample_data/job_description.txt --resumes sample_data/resumes --threshold 0.15
"""
import argparse
import os
import sys
import time

from screener import ResumeScreener, Candidate


def load_candidates(resumes_dir: str) -> list[Candidate]:
    candidates = []
    for filename in sorted(os.listdir(resumes_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(resumes_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        name = os.path.splitext(filename)[0].replace("candidate_", "").replace("_", " ").title()
        candidates.append(Candidate(name=name, resume_text=text))
    return candidates


def main():
    # Windows consoles often default to a legacy codepage (e.g. cp1252) that
    # can't encode the checkmark below; force UTF-8 so output doesn't crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI-powered resume screening system")
    parser.add_argument("--job", required=True, help="Path to job description .txt file")
    parser.add_argument("--resumes", required=True, help="Directory of candidate resume .txt files")
    parser.add_argument("--threshold", type=float, default=0.15, help="Minimum similarity score to shortlist (0..1)")
    parser.add_argument("--top-k", type=int, default=None, help="Cap the shortlist to top K candidates")
    args = parser.parse_args()

    with open(args.job, "r", encoding="utf-8") as f:
        job_description = f.read()

    candidates = load_candidates(args.resumes)
    if not candidates:
        print(f"No .txt resumes found in {args.resumes}", file=sys.stderr)
        sys.exit(1)

    screener = ResumeScreener(job_description)

    start = time.time()
    ranked = screener.rank(candidates)
    elapsed_ms = (time.time() - start) * 1000

    print(f"\nScreened {len(candidates)} resumes in {elapsed_ms:.1f}ms")
    print(f"Auto-derived required skills from JD: {', '.join(screener.required_keywords)}\n")
    print(f"{'Rank':<5}{'Candidate':<20}{'Score':<10}{'Matched Skills'}")
    print("-" * 90)
    for i, r in enumerate(ranked, start=1):
        matched = ", ".join(r.matched_keywords) if r.matched_keywords else "-"
        print(f"{i:<5}{r.name:<20}{r.score:<10}{matched}")

    shortlisted = [r for r in ranked if r.score >= args.threshold]
    if args.top_k:
        shortlisted = shortlisted[: args.top_k]

    print(f"\nShortlisted ({len(shortlisted)}/{len(candidates)}) at threshold >= {args.threshold}:")
    for r in shortlisted:
        print(f"  ✓ {r.name} (score={r.score})")
        if r.missing_keywords:
            print(f"      missing: {', '.join(r.missing_keywords)}")


if __name__ == "__main__":
    main()

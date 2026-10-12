"""
Evaluates the screener against a public, human-labeled resume/job-description
dataset.

Dataset: cnamuangtoun/resume-job-description-fit on Hugging Face
(6,241 train / 1,759 test pairs, each labeled "Good Fit", "Potential Fit" or
"No Fit"). It is downloaded on first run into ./eval_data (git-ignored).

Task: shortlisting. A candidate is a true positive if labeled Good Fit or
Potential Fit, negative if No Fit.

Method:
  * Resumes are grouped by job description and ranked with ResumeScreener,
    exactly as the CLI would rank a batch of applicants for one posting.
  * The shortlisting threshold is chosen on the TRAIN split only, then
    applied unchanged to the held-out TEST split.
  * Compared against a majority-class baseline, with ROC AUC reported as a
    threshold-free measure of ranking quality.

Run with: python evaluate.py
"""
import csv
import os
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from screener import Candidate, ResumeScreener

DATA_DIR = os.path.join(os.path.dirname(__file__), "eval_data")
BASE_URL = "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main"
POSITIVE = {"Good Fit", "Potential Fit"}


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Downloading {name}.csv ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv", path)
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_split(rows):
    """Ranks each job description's applicant batch; returns scores, labels, and per-batch timings."""
    batches = defaultdict(list)
    for row in rows:
        batches[row["job_description_text"]].append(row)

    scores, labels, timings = [], [], []
    for jd, batch in batches.items():
        candidates = [Candidate(name=str(i), resume_text=r["resume_text"]) for i, r in enumerate(batch)]
        t0 = time.perf_counter()
        ranked = ResumeScreener(jd).rank(candidates)
        timings.append((time.perf_counter() - t0) * 1000)
        by_name = {r.name: r.score for r in ranked}
        for i, r in enumerate(batch):
            scores.append(by_name[str(i)])
            labels.append(int(r["label"] in POSITIVE))
    return np.array(scores), np.array(labels), np.array(timings), len(batches)


def best_threshold(scores, labels):
    candidates = np.unique(scores)
    accs = [((scores >= t).astype(int) == labels).mean() for t in candidates]
    return float(candidates[int(np.argmax(accs))])


def main():
    train_s, train_y, _, train_batches = score_split(load_split("train"))
    test_s, test_y, timings, test_batches = score_split(load_split("test"))

    threshold = best_threshold(train_s, train_y)
    pred = (test_s >= threshold).astype(int)
    majority = int(train_y.mean() >= 0.5)

    tp = int(((pred == 1) & (test_y == 1)).sum())
    precision = tp / max(1, pred.sum())
    recall = tp / max(1, test_y.sum())

    print(f"Train: {len(train_y)} pairs across {train_batches} job descriptions")
    print(f"Test:  {len(test_y)} pairs across {test_batches} job descriptions (held out)\n")
    print(f"Threshold (tuned on train): {threshold:.4f}")
    print(f"Held-out shortlisting accuracy : {(pred == test_y).mean() * 100:.1f}%")
    print(f"Majority-class baseline        : {(test_y == majority).mean() * 100:.1f}%")
    print(f"Precision / recall             : {precision * 100:.1f}% / {recall * 100:.1f}%")
    print(f"ROC AUC (ranking quality)      : {roc_auc_score(test_y, test_s):.3f}  (0.5 = random)")
    print(f"Batch ranking time             : median {np.median(timings):.1f}ms, "
          f"p95 {np.percentile(timings, 95):.1f}ms (avg batch {len(test_y) / test_batches:.0f} resumes)")


if __name__ == "__main__":
    main()

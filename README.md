# AI-Powered Resume Screening System

An NLP-driven candidate ranking engine that scores resumes against a job description using TF-IDF semantic similarity, extracts distinctive keywords, and automatically shortlists the strongest matches.

## How it works

1. The job description and every resume are vectorized into a shared **TF-IDF** space (unigrams + bigrams).
2. Each resume is scored by **cosine similarity** against the job description vector — this is the semantic-similarity step.
3. Required skills are auto-derived from the job description itself (or supplied explicitly), then diffed against each resume to report matched vs. missing skills.
4. Candidates are ranked and can be shortlisted by a minimum score threshold and/or capped to the top K.

Built on TF-IDF + cosine similarity (not a large transformer model) so it runs instantly, fully offline, with no GPU or model download — and the `ResumeScreener` interface is written so it can be swapped later for sentence-transformer embeddings without changing calling code.

## Tech Stack

Python · scikit-learn (TF-IDF, cosine similarity) · pytest

## Getting Started

```bash
pip install -r requirements.txt

python cli.py --job sample_data/job_description.txt \
              --resumes sample_data/resumes \
              --threshold 0.15
```

Sample output:

```
Screened 4 resumes in 10.0ms
Auto-derived required skills from JD: design, aws, data, docker, ...

Rank Candidate           Score     Matched Skills
------------------------------------------------------------------------------------------
1    Alice               0.2297    design, aws, docker, experience, postgresql, python, rest...
2    Carla               0.2206    design, aws, data, docker, experience, postgresql, python...
3    Bob                 0.0482    design, experience, rest
4    Dev                 0.0335    design, experience, using

Shortlisted (2/4) at threshold >= 0.15:
  ✓ Alice (score=0.2297)
  ✓ Carla (score=0.2206)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Usage as a library

```python
from screener import ResumeScreener, Candidate

screener = ResumeScreener(job_description_text)
ranked = screener.rank([
    Candidate(name="Alice", resume_text=alice_resume_text),
    Candidate(name="Bob", resume_text=bob_resume_text),
])

for c in ranked:
    print(c.name, c.score, c.matched_keywords)
```

## Project Structure

```
resume-screening-ai/
├── cli.py                     # command-line entrypoint
├── screener/
│   ├── engine.py               # ResumeScreener core ranking logic
│   └── keyword_extractor.py    # TF-IDF keyword extraction & overlap diffing
├── sample_data/
│   ├── job_description.txt
│   └── resumes/*.txt
└── tests/
    └── test_screener.py
```

## License

MIT — see [LICENSE](LICENSE).

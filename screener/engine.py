"""
Core NLP-driven candidate ranking engine.

Approach:
  1. Represent the job description and every resume as TF-IDF vectors over
     a shared vocabulary (unigrams + bigrams).
  2. Score each resume by cosine similarity against the job description --
     this is the "semantic similarity scoring" step.
  3. Extract distinctive keywords per resume and diff them against the
     required skills list pulled from the job description.
  4. Rank and optionally filter candidates above a similarity threshold.

This is intentionally built on TF-IDF + cosine similarity rather than a
heavyweight transformer model so it runs instantly, fully offline, and is
trivial to explain to a recruiter/interviewer -- the same interface can be
swapped later for sentence-transformer embeddings for deeper semantic
matching without changing the calling code.
"""
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .keyword_extractor import clean_text, extract_keywords, keyword_overlap


@dataclass
class Candidate:
    name: str
    resume_text: str


@dataclass
class ScoredCandidate:
    name: str
    score: float  # 0..1 semantic similarity to the job description
    matched_keywords: list = field(default_factory=list)
    missing_keywords: list = field(default_factory=list)
    extracted_keywords: list = field(default_factory=list)


class ResumeScreener:
    def __init__(self, job_description: str, required_keywords: list[str] | None = None):
        self.job_description = job_description
        # If the caller doesn't supply an explicit required-skills list,
        # auto-derive one from the job description itself.
        self.required_keywords = required_keywords or extract_keywords(job_description, top_n=12)

    def rank(self, candidates: list[Candidate]) -> list[ScoredCandidate]:
        if not candidates:
            return []

        corpus = [clean_text(self.job_description)] + [clean_text(c.resume_text) for c in candidates]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=2000)
        matrix = vectorizer.fit_transform(corpus)

        job_vector = matrix[0:1]
        resume_vectors = matrix[1:]
        similarities = cosine_similarity(job_vector, resume_vectors)[0]

        results = []
        for candidate, score in zip(candidates, similarities):
            overlap = keyword_overlap(self.required_keywords, candidate.resume_text)
            results.append(
                ScoredCandidate(
                    name=candidate.name,
                    score=round(float(score), 4),
                    matched_keywords=overlap["present"],
                    missing_keywords=overlap["missing"],
                    extracted_keywords=extract_keywords(candidate.resume_text, top_n=10),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def shortlist(self, candidates: list[Candidate], threshold: float = 0.15, top_k: int | None = None) -> list[ScoredCandidate]:
        """Filter candidates by a minimum similarity threshold, capped at top_k."""
        ranked = self.rank(candidates)
        filtered = [r for r in ranked if r.score >= threshold]
        return filtered[:top_k] if top_k else filtered

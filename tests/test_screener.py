import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screener import ResumeScreener, Candidate, extract_keywords


JOB_DESCRIPTION = """
Backend Software Engineer needed. Must know Python, REST APIs, PostgreSQL,
Docker, and AWS. Experience with microservices architecture is a plus.
"""

STRONG_RESUME = """
Software engineer with 4 years of Python experience building REST APIs
with Flask and FastAPI. Deployed microservices to AWS using Docker.
Worked extensively with PostgreSQL for relational data storage.
"""

WEAK_RESUME = """
Marketing coordinator specializing in social media campaigns, SEO,
and content writing. Experience with Google Analytics and Canva.
"""


def test_relevant_resume_scores_higher_than_irrelevant():
    screener = ResumeScreener(JOB_DESCRIPTION)
    candidates = [
        Candidate(name="Strong", resume_text=STRONG_RESUME),
        Candidate(name="Weak", resume_text=WEAK_RESUME),
    ]
    ranked = screener.rank(candidates)
    assert ranked[0].name == "Strong"
    assert ranked[0].score > ranked[1].score


def test_shortlist_filters_by_threshold():
    screener = ResumeScreener(JOB_DESCRIPTION)
    candidates = [
        Candidate(name="Strong", resume_text=STRONG_RESUME),
        Candidate(name="Weak", resume_text=WEAK_RESUME),
    ]
    shortlisted = screener.shortlist(candidates, threshold=0.15)
    names = [c.name for c in shortlisted]
    assert "Strong" in names
    assert "Weak" not in names


def test_shortlist_respects_top_k():
    screener = ResumeScreener(JOB_DESCRIPTION)
    candidates = [
        Candidate(name=f"Candidate{i}", resume_text=STRONG_RESUME) for i in range(5)
    ]
    shortlisted = screener.shortlist(candidates, threshold=0.0, top_k=2)
    assert len(shortlisted) == 2


def test_extract_keywords_returns_nonempty_for_real_text():
    keywords = extract_keywords(STRONG_RESUME, top_n=5)
    assert len(keywords) > 0
    assert all(isinstance(k, str) for k in keywords)


def test_empty_candidate_list_returns_empty():
    screener = ResumeScreener(JOB_DESCRIPTION)
    assert screener.rank([]) == []


def test_matched_keywords_present_for_relevant_resume():
    screener = ResumeScreener(JOB_DESCRIPTION, required_keywords=["python", "docker", "postgresql"])
    ranked = screener.rank([Candidate(name="Strong", resume_text=STRONG_RESUME)])
    assert "python" in ranked[0].matched_keywords
    assert "docker" in ranked[0].matched_keywords


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

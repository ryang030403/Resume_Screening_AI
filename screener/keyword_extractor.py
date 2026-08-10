"""
Lightweight keyword extraction using TF-IDF term weighting.

No heavyweight NLP model download required (works fully offline), which
keeps the system easy to deploy -- swap in spaCy/transformers embeddings
later behind the same extract_keywords() interface if richer semantics
are needed.
"""
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = set(
    """a an the and or but if while is are was were be been being to of in on
    for with as by at from this that these those it its it's i you he she
    they we our your their his her them us not no so than then also into
    over under out up down""".split()
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\.\s]", " ", text)  # keep tokens like c++, c#, node.js
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Extract the top_n most distinctive keywords/phrases from a single document
    using TF-IDF term weighting over unigrams and bigrams."""
    cleaned = clean_text(text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(STOPWORDS),
        max_features=200,
    )
    try:
        matrix = vectorizer.fit_transform([cleaned])
    except ValueError:
        return []
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, score in ranked[:top_n] if score > 0]


def keyword_overlap(job_keywords: list[str], resume_text: str) -> dict:
    """Return which required job keywords are present/missing in a resume."""
    cleaned_resume = clean_text(resume_text)
    present, missing = [], []
    for kw in job_keywords:
        if kw in cleaned_resume:
            present.append(kw)
        else:
            missing.append(kw)
    return {"present": present, "missing": missing}

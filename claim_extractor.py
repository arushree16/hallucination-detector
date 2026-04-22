"""
claim_extractor.py  —  Person 1's module
==========================================
Extracts factual, verifiable claims from raw text using spaCy.
Filters out opinions, questions, commands, and subjective fluff.

Install:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import spacy
from typing import List

# Load once at module level
nlp = spacy.load("en_core_web_sm")

# Subjective / opinion lemmas to reject
SUBJECTIVE_LEMMAS = {
    "beautiful", "amazing", "awesome", "great", "terrible", "ugly",
    "bad", "good", "best", "worst", "think", "believe", "feel",
    "opinion", "suggest", "lovely", "horrible", "incredible",
    "stunning", "gorgeous", "should", "totally",
}

# Scientific / factual keywords that indicate verifiable claims
SCIENTIFIC_KEYWORDS = {
    "dna", "helix", "structure", "molecule", "atom", "cell",
    "visible", "structure", "composed", "contains", "consists",
    "mountain", "river", "desert", "ocean", "planet", "galaxy",
    "species", "organism", "chemical", "element", "compound",
    # Animals and biology
    "bat", "bats", "bird", "birds", "mammal", "animal", "animals",
    "fish", "insect", "reptile", "amphibian",
    # Health and body
    "coffee", "growth", "health", "brain", "heart", "blood",
    # Common fact verbs
    "stunt", "cause", "prevent", "cure", "heal", "affect",
}


def extract_claims(text: str) -> List[str]:
    """
    Extracts sentence-level factual claims from raw text.

    Filters applied (in order):
    1. Skip empty / too-short sentences (< 4 real tokens)
    2. Skip questions  (ends with '?')
    3. Skip commands   (first real token is base-form verb)
    4. Skip opinions   (contains a known subjective lemma)
    5. Skip adjective-heavy sentences (ADJ ratio > 20 %)
    6. Keep only sentences with a Named Entity OR a number

    Parameters
    ----------
    text : Raw input paragraph or article.

    Returns
    -------
    List of factual claim strings.
    """
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()

        # 0. Skip empty
        if not sentence_text:
            continue

        # 1. Skip questions
        if sentence_text.endswith("?"):
            continue

        # Real tokens (no punctuation / whitespace)
        tokens = [t for t in sent if not t.is_punct and not t.is_space]

        # 2. Skip very short sentences
        if len(tokens) < 4:
            continue

        # 3. Skip commands (first word is base-form verb: VB tag, but not gerund/VBG)
        # Allow "Drinking coffee..." (VBG) as it's a factual claim, not a command
        if tokens[0].tag_ == "VB" and tokens[0].tag_ != "VBG":
            continue

        # 4. Skip opinions
        if any(t.lemma_.lower() in SUBJECTIVE_LEMMAS for t in tokens):
            continue

        # 5. Skip adjective-heavy sentences
        adj_ratio = sum(1 for t in tokens if t.pos_ == "ADJ") / len(tokens)
        if adj_ratio > 0.20:
            continue

        # 6. Must contain a Named Entity OR a number OR scientific keyword
        has_entity = len(sent.ents) > 0
        has_number = any(t.like_num for t in tokens)
        has_scientific = any(t.lemma_.lower() in SCIENTIFIC_KEYWORDS for t in tokens)

        if has_entity or has_number or has_scientific:
            claims.append(sentence_text)

    return claims

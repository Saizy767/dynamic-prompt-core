"""
Thesis normalization for the Stage 1 baseline runner.

Lemmatize + lower-case + collapse whitespace, preserving negation particles
(не, нет, без, никогда) unchanged.  pymorphy3 is used for Russian, spaCy for
English.  Both are imported lazily so this module imports without them.
"""
from __future__ import annotations

import re
from typing import Optional

NEGATION_PARTICLES = {"не", "нет", "без", "никогда"}

_ru_morph: Optional[object] = None
_en_nlp: Optional[object] = None


def _get_ru_morph():
    global _ru_morph
    if _ru_morph is None:
        try:
            import pymorphy3  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Russian normalization requires 'pymorphy3'; "
                "install with: pip install pymorphy3"
            ) from exc
        _ru_morph = pymorphy3.MorphAnalyzer()
    return _ru_morph


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "English normalization requires 'spacy'; "
                "install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            ) from exc
        try:
            _en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            _en_nlp = spacy.blank("en")
    return _en_nlp


def _lemmatize_word(word: str, lang: str) -> str:
    if lang == "ru":
        morph = _get_ru_morph()
        parses = morph.parse(word)
        if parses:
            return parses[0].normal_form
        return word
    elif lang == "en":
        nlp = _get_en_nlp()
        doc = nlp(word)
        if doc and doc[0].lemma_:
            return doc[0].lemma_
        return word
    return word


def normalize_thesis(text: str, lang: str = "ru") -> str:
    """Normalize a single thesis: lower-case, lemmatize, collapse whitespace.

    Negation particles (не, нет, без, никогда) are passed through lower-cased
    without lemmatization so they are never altered.  Punctuation at word
    boundaries is stripped before lemmatization.
    """
    words = text.split()
    out: list[str] = []
    for w in words:
        lw = w.lower()
        lw = re.sub(r"^[^\w]+|[^\w]+$", "", lw)
        if not lw:
            continue
        if lw in NEGATION_PARTICLES:
            out.append(lw)
        else:
            out.append(_lemmatize_word(lw, lang))
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def normalize_theses(theses: list[str], lang: str = "ru") -> list[str]:
    """Normalize a list of theses, returning a new list."""
    return [normalize_thesis(t, lang) for t in theses]

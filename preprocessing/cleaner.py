from __future__ import annotations

import logging
import re
from typing import Iterable

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SIGNATURE_PATTERN = re.compile(r"(?m)^--\s*$.*", re.DOTALL)
DISCLAIMER_PATTERN = re.compile(r"(?ims)(this email.*?confidential|please consider the environment before printing).*")
REPLY_PATTERN = re.compile(r"(?ms)^>.*$")
FORWARD_HEADER_PATTERN = re.compile(r"(?ims)^(-{5,}|_{5,}).*?original message.*$", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_HEADER_PATTERN = re.compile(r"(?m)^(from|sent|to|cc|subject|date):.*$")
PUNCTUATION_SPACING = re.compile(r"\s+([,.!?])")


def strip_html(text: str) -> str:
    """
    Convert HTML to plain text.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def remove_noise(text: str, extra_boilerplate: Iterable[str] | None = None) -> str:
    """
    Remove signatures, disclaimers, reply chains, URLs, email headers, and custom boilerplate.
    """
    text = strip_html(text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_HEADER_PATTERN.sub(" ", text)
    text = SIGNATURE_PATTERN.sub(" ", text)
    text = DISCLAIMER_PATTERN.sub(" ", text)
    text = REPLY_PATTERN.sub(" ", text)
    text = FORWARD_HEADER_PATTERN.sub(" ", text)
    if extra_boilerplate:
        for phrase in extra_boilerplate:
            text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_punctuation(text: str) -> str:
    text = PUNCTUATION_SPACING.sub(r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

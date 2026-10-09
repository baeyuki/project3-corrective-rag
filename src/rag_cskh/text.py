"""Vietnamese tokenisation for BM25.

Vietnamese words are space-separated syllables, most words have two ("bảo hành"), and customers often type
without diacritics ("bao hanh"). Tokens = syllables + syllable bigrams, each also in an accent-folded form
("~bao", "~bao_hanh") on BOTH the document and the query side, so a query typed without diacritics still
matches. Numbers are normalised: "15.000.000" -> "15000000".
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = set("""và là của có cho được không thì mà với các những một này đó khi để từ trong trên ra vào đã đang sẽ
rất cũng như nào gì ạ ơi vậy nhé nha shop bên mình tôi tớ em anh chị à ừ thế""".split())

_TOKEN = re.compile(r"[0-9a-zà-ỹđ]+(?:[.,][0-9]+)*", re.IGNORECASE)


def fold(text: str) -> str:
    """Remove Vietnamese diacritics: "bảo hành" -> "bao hanh"."""
    s = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def syllables(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(unicodedata.normalize("NFC", text.lower())):
        out.append(tok.replace(".", "").replace(",", "") if tok[0].isdigit() else tok)
    return out


def tokens(text: str, bigrams: bool = True, folded: bool = True) -> list[str]:
    syl = syllables(text)
    out = [s for s in syl if s not in STOPWORDS]
    if bigrams:
        out += [f"{a}_{b}" for a, b in zip(syl, syl[1:]) if not (a in STOPWORDS and b in STOPWORDS)]
    if folded:
        out += ["~" + fold(t) for t in list(out)]
    return out

"""Locate verbatim quotations in the saved source's Unicode coordinates."""


def quote_spans(text: str, quote: str) -> list[tuple[int, int]]:
    """Return every exact occurrence, including overlaps; never guess a match.

    Do not normalize Greek or whitespace: positions index the unchanged source.
    """
    if not quote or not quote.strip():
        return []
    spans = []
    start = text.find(quote)
    while start >= 0:
        spans.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return spans


def occurrence_label(text: str, span: tuple[int, int], index: int) -> str:
    a, b = span
    before = text[max(0, a - 45):a]
    after = text[b:b + 45]
    quote = text[a:b]
    if len(quote) > 90:
        quote = quote[:87] + "…"
    return f"{index + 1}番目：…{before}【{quote}】{after}…"

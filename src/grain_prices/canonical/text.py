from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    text: str
    original_indexes: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not self.original_indexes:
            return (0, 0)
        start = min(max(start, 0), len(self.original_indexes) - 1)
        if end <= start:
            return (self.original_indexes[start], self.original_indexes[start])
        end = min(end - 1, len(self.original_indexes) - 1)
        return (self.original_indexes[start], self.original_indexes[end] + 1)


def normalize_with_map(surface: str) -> NormalizedText:
    """Normalize text while retaining an index back to the original surface."""
    chars: list[str] = []
    indexes: list[int] = []
    pending_space: int | None = None
    for original_index, char in enumerate(surface or ""):
        expanded = unicodedata.normalize("NFKC", char)
        expanded = unicodedata.normalize("NFKD", expanded).casefold()
        expanded = "".join(c for c in expanded if not unicodedata.combining(c))
        for normalized in expanded:
            if normalized.isspace():
                if chars:
                    pending_space = original_index
                continue
            if pending_space is not None:
                chars.append(" ")
                indexes.append(pending_space)
                pending_space = None
            chars.append(normalized)
            indexes.append(original_index)
    return NormalizedText("".join(chars), tuple(indexes))


def stable_window_id(doc_id: str, grain_surface: str, neighborhood: str) -> str:
    identity_text = normalize_with_map(neighborhood).text
    identity_grain = normalize_with_map(grain_surface).text
    payload = "|".join((doc_id, identity_grain, identity_text)).encode("utf-8")
    return f"{doc_id}#{hashlib.sha1(payload).hexdigest()[:16]}"

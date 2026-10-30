#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window generation module.

Creates analysis windows (narrow text contexts) around grain mentions
for classification and extraction.

Refactored from Stage 2B (2b_llm_extract_prices.py::find_grain_windows).

Usage:
    from grain_prices.preprocessing.build_windows import find_grain_windows, GrainWindow
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from grain_prices.pipeline_shared import (
    GRAIN_RE,
    MONEY_RE,
    PRICEWORD_RE,
    UNIT_RE,
    normalize_for_search,
)


@dataclass
class GrainWindow:
    """A narrow text window centered on a grain mention."""
    grain_form: str
    window_text: str
    norm_window: str
    char_offset: int
    has_money: bool
    has_unit: bool
    has_number: bool
    has_priceword: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grain_form": self.grain_form,
            "window_text": self.window_text,
            "norm_window": self.norm_window,
            "char_offset": self.char_offset,
            "has_money": self.has_money,
            "has_unit": self.has_unit,
            "has_number": self.has_number,
            "has_priceword": self.has_priceword,
        }


def find_grain_windows(
    blocks: List[Tuple[str, str]],
    window_chars: int = 200,
    require_number: bool = True,
    dedup_bucket_size: int = 0,
) -> List[GrainWindow]:
    """
    Find grain mentions and return narrow text windows.

    Each window is centered on a grain mention with `window_chars` total context.

    Args:
        blocks: list of (tag, text) from iter_text_blocks
        window_chars: total window width in characters
        require_number: skip windows without numeric content
        dedup_bucket_size: legacy grid-based deduplication. 0 (default) keeps
            every grain mention.

            The legacy behaviour is NOT "merge mentions closer than N chars".
            It slices the text into fixed N-char cells from offset 0 and keeps
            only the first mention per cell, so whether two nearby mentions
            collapse depends on where the grid boundary happens to fall, not on
            the distance between them. Corpus-wide at N=50 that discarded 6,157
            of 42,293 mentions (14.6%), of which 2,225 carried a *different*
            grain word than the mention that survived — i.e. a κριθή a few
            characters after a πυρός vanished from the dataset entirely.
            It is also unstable: any upstream change to text length reshuffles
            every offset and re-rolls which mentions survive.

            Kept only so old runs can be reproduced; do not enable it for new work.

    Returns:
        List of GrainWindow objects
    """
    full_text = " ||| ".join(text for _, text in blocks)
    norm_text = normalize_for_search(full_text)

    windows = []
    seen_positions: Set[int] = set()

    for m in GRAIN_RE.finditer(norm_text):
        grain_start = m.start()
        grain_end = m.end()
        grain_form = m.group(0)

        # Legacy grid deduplication (disabled by default — see docstring)
        if dedup_bucket_size:
            bucket = grain_start // dedup_bucket_size
            if bucket in seen_positions:
                continue
            seen_positions.add(bucket)

        # Extract window from ORIGINAL text
        half = window_chars // 2
        win_start = max(0, grain_start - half)
        win_end = min(len(full_text), grain_end + half)
        window_text = full_text[win_start:win_end].strip()

        # Normalized window for pattern checks
        norm_window = norm_text[win_start:win_end].strip()

        # Pattern checks
        has_money = bool(MONEY_RE.search(norm_window))
        has_unit = bool(UNIT_RE.search(norm_window))
        has_number = bool(re.search(r"\d+", window_text))
        has_priceword = bool(PRICEWORD_RE.search(norm_window))

        # Pre-filter
        if require_number and not has_number:
            continue

        windows.append(GrainWindow(
            grain_form=grain_form,
            window_text=window_text,
            norm_window=norm_window,
            char_offset=grain_start,
            has_money=has_money,
            has_unit=has_unit,
            has_number=has_number,
            has_priceword=has_priceword,
        ))

    return windows


def find_grain_windows_from_text(
    full_text: str,
    window_chars: int = 200,
    require_number: bool = True,
) -> List[GrainWindow]:
    """
    Convenience: find grain windows directly from concatenated text
    (without requiring block tuples).
    """
    # Wrap as single block
    return find_grain_windows(
        blocks=[("text", full_text)],
        window_chars=window_chars,
        require_number=require_number,
    )

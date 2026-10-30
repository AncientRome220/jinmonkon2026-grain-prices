from __future__ import annotations

from grain_prices.pipeline_shared import EXPLICIT_LINK_CUES


BOUNDARY_TOKENS = {"zzlbzz", "zzcbzz", "zzpbzz"}
ACCOUNTING_BOUNDARIES = ("γινονται", "λοιπ", "ομοιως")


def relation_linkage(
    grain_token: int,
    money_token: int,
    quantity_token: int | None,
    explicit_cues: list[dict],
    crosses_boundary: bool,
    proximate_max_tokens: int = 12,
) -> tuple[str, list[str]]:
    """Assign linkage to one selected grain/quantity/money relation."""
    lo, hi = sorted((grain_token, money_token))
    local = sorted({
        cue["label"] for cue in explicit_cues
        if cue.get("label") in EXPLICIT_LINK_CUES
        and lo - 3 <= int(cue.get("token_index", -999)) <= hi + 3
    })
    if local and not crosses_boundary:
        return "explicit", local
    if (
        quantity_token is not None
        and abs(grain_token - money_token) <= proximate_max_tokens
        and abs(grain_token - quantity_token) <= proximate_max_tokens
        and not crosses_boundary
    ):
        return "proximate", []
    return "weak", []

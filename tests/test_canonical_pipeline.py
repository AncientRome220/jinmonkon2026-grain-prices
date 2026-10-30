from pathlib import Path
from importlib.resources import files

PROMPT_PATH = Path(str(files("grain_prices.prompts").joinpath(
    "price_adjudication_canonical_v13_1.md"
)))

def sample_candidate():
    return {
        "window_id": "w1", "ddb_id": "test;1;1", "grain_form": "πυροῦ",
        "grain_token_index": 1, "explicit_cue_positions": [{"label": "priceword_ext", "token_index": 0}],
        "evidence_text": "τιμῆς πυροῦ ἀρτάβης 2 δραχμαὶ 8",
        "qty_options": [{"option_id": "q0", "value": "2", "unit": "αρταβης", "token_index": 2}],
        "money_options": [{"option_id": "m0", "value": "8", "currency": "δραχμαι", "token_index": 4,
                           "crosses_relation_boundary": False, "intrinsic_value": None}],
        "place": "", "date_not_before": "", "date_not_after": "",
    }

def sample_answer():
    return {"any_price": True, "observations": [{
        "quantity_option": "q0", "money_option": "m0", "commodity": "wheat",
        "commodity_class": "raw_grain", "linkage_strength": "weak",
        "context_tags": ["tax_commutation"], "rate_basis": "derived_from_total",
        "evidence_span": "πυροῦ ἀρτάβης 2 δραχμαὶ 8", "confidence": "high",
        "reasoning": "The τιμή construction values the wheat in money.",
    }]}

"""Candidate content identity, independent of local file names."""
import hashlib
import json


def candidate_fingerprint(candidate: dict) -> str:
    payload = {k: v for k, v in candidate.items() if k != "candidate_fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_candidate_binding(prediction: dict, candidate: dict) -> None:
    expected = candidate_fingerprint(candidate)
    recorded = candidate.get("candidate_fingerprint")
    if recorded and recorded != expected:
        raise ValueError(f"candidate fingerprint invalid: {candidate['window_id']}")
    actual = prediction.get("candidate_fingerprint")
    if (recorded or actual) and actual != expected:
        raise ValueError(f"prediction/candidate fingerprint mismatch: {candidate['window_id']}")

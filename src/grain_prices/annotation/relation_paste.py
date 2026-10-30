"""Parse explicit, human-reviewable relation text; never infer readings or save."""
from grain_prices.annotation.span_selection import quote_spans
from grain_prices.canonical.relations import validate_relation


def parse_relation_paste(paste: str, row: dict) -> dict:
    fields = {}
    allowed = {"対象", "穀物", "穀物の引用", "数量", "金額", "価格種別", "根拠"}
    for line in paste.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        key, sep, value = line.replace("：", ":", 1).partition(":")
        key, value = key.strip(), value.strip()
        if not sep or key not in allowed or not value:
            raise ValueError("各行を「項目: 内容」の形式にしてください。対応項目：" + "、".join(sorted(allowed)))
        if key in fields and key not in {"数量", "金額"}:
            raise ValueError(f"{key} が重複しています。")
        fields.setdefault(key, []).append(value)
    if set(fields) != allowed:
        raise ValueError("不足している項目：" + "、".join(sorted(allowed - set(fields))))
    if fields["対象"][0] not in row["origin_sample_ids"] + [row["target_id"]]:
        raise ValueError("対象が現在の画面と異なります。対象を切り替えてください。")
    text = row["source_block_text"]
    ga, gb = row["grain_span"]
    if fields["穀物の引用"][0] != text[ga:gb]:
        raise ValueError("穀物の引用が現在選択されている穀物と一致しません。")
    evidence = [s for s in quote_spans(text, fields["根拠"][0]) if s[0] <= ga < gb <= s[1]]
    if len(evidence) != 1:
        raise ValueError("根拠が対象の穀物を含む原文の一箇所に一致しません。原文からコピーし直してください。")
    lo, hi = evidence[0]
    relation = {"grain_span": list(row["grain_span"]), "commodity": fields["穀物"][0],
                "evidence_span": [lo, hi], "rate_basis": fields["価格種別"][0]}
    for label, side in (("数量", "quantity"), ("金額", "money")):
        entries = fields[label]
        if entries == ["unstated"]:
            relation[side] = {"status": "unstated", "components": []}
            continue
        components = []
        statuses = set()
        for entry in entries:
            parts = [p.strip() for p in entry.split("|")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"{label} は「数値 | 単位 | 原文の引用」または unstated としてください。")
            value, denomination, quote = parts
            spans = [s for s in quote_spans(text, quote) if lo <= s[0] < s[1] <= hi]
            if len(spans) != 1:
                raise ValueError(f"{label} の引用が根拠内の一箇所に一致しません。引用を長くするか個別入力を使ってください。")
            statuses.add("unreadable" if value == "unreadable" else "present")
            components.append({"span": list(spans[0]), "value": None if value in {"null", "unreadable"} else value,
                               "denomination": None if denomination == "null" else denomination})
        if len(statuses) != 1:
            raise ValueError(f"{label} の判読状態が混在しています。個別入力で確認してください。")
        relation[side] = {"status": statuses.pop(), "components": components}
    errors = validate_relation(relation, row)
    if errors:
        raise ValueError("入力を確認してください：" + "; ".join(errors))
    return relation

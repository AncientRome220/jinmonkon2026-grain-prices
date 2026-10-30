"""Separate Streamlit target-grain review UI. No model predictions in gold mode."""
import argparse
from copy import deepcopy
import html
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from grain_prices.annotation.relation_review import grain_choices, reanchor, save_review
from grain_prices.canonical.io import read_jsonl, sha256_file
from grain_prices.canonical.relation_windows import window_candidate
from grain_prices.canonical.relations import validate_relation
from grain_prices.canonical.contracts import CONTEXT_VALUES, LINKAGE_VALUES
from grain_prices.annotation.span_selection import quote_spans, occurrence_label
from grain_prices.annotation.relation_paste import parse_relation_paste
from grain_prices.annotation.relation_autofill import suggest_relation
from grain_prices.canonical.relations import linkage_key


def marked(text, span, offset=0):
    a, b = span[0]-offset, span[1]-offset
    return '<div style="font-family: Gentium Plus, Palatino Linotype, serif; font-size:20px; line-height:1.8">'+html.escape(text[:a])+"<mark>"+html.escape(text[a:b])+"</mark>"+html.escape(text[b:])+"</div>"


def blank_relation(row):
    return {"grain_span": row["grain_span"], "commodity": "grain_unspecified",
            "quantity": {"status": "unresolved", "components": []},
            "money": {"status": "unresolved", "components": []},
            "rate_basis": "unknown", "evidence_span": row["grain_span"]}


def evidence_picker(text, evidence, key):
    """Form widgets: paste a quotation, disambiguate repeats, preview the span."""
    a, b = evidence
    valid = isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(text)
    st.markdown("**根拠範囲**")
    quote = st.text_area(
        "根拠の引用（原文からコピー）", value=text[a:b] if valid else "",
        key=key+"-quote", height=110,
        help="上の原文から連続した範囲をコピーしてください。下の「入力を反映・プレビュー」で位置とハイライトを確認できます。",
    )
    matches = quote_spans(text, quote)
    chosen = matches[0] if len(matches) == 1 else None
    if len(matches) > 1:
        st.info(f"同じ引用が{len(matches)}箇所あります。前後の文脈で選んでください。")
        options = [None] + matches
        suffix = hashlib.sha256(quote.encode("utf-8")).hexdigest()[:16]
        chosen = st.selectbox(
            "一致箇所を選択", options,
            index=options.index(tuple(evidence)) if tuple(evidence) in matches else 0,
            format_func=lambda s: "選択してください" if s is None else occurrence_label(text, s, matches.index(s)),
            key=key+"-occurrence-"+suffix,
        )
    with st.expander("詳細：文字位置を直接指定する"):
        manual = st.checkbox("引用検索の代わりに数値で指定", key=key+"-manual")
        st.caption("開始位置を含み、終了位置は含みません。通常は引用の貼り付けだけで指定できます。")
        start = st.number_input("Evidence start", min_value=0, max_value=len(text),
                                value=max(0, min(a, len(text))), step=1, key=key+"-a")
        end = st.number_input("Evidence end", min_value=0, max_value=len(text),
                              value=max(0, min(b, len(text))), step=1, key=key+"-b")
    if manual:
        chosen = (start, end) if start < end else None
        error = "終了位置は開始位置より後にしてください。" if chosen is None else None
    else:
        error = ("引用が原文と一致しません。原文からコピーし直してください（空白・アクセントも照合します）。"
                 if not matches else "同じ引用のどの箇所かを選択してください。" if chosen is None else None)
    if chosen is not None:
        a, b = chosen
        st.caption(f"選択範囲：開始 {a} ／ 終了 {b} · {b-a}文字")
        lo, hi = max(0, a-100), min(len(text), b+100)
        st.markdown(marked(text[lo:hi], chosen, lo), unsafe_allow_html=True)
    else:
        st.warning(error)
    return list(chosen) if chosen is not None else evidence, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="Target-grain review", layout="wide")
    st.title("Target-grain relation review")
    st.caption("Proposals are unconfirmed. Review the target and every valid relation, including explicit negatives. Save a draft before changing targets, adding/removing relations, or reanchoring.")
    with st.expander("判定基準：穀物言及への価格関係の帰属（2026-10-28）"):
        st.markdown(
            "1. **価格を述べる句に穀物名がある場合**：数量・金額は、その言及に対応づけます。"
            "文の構造と意味から判断し、同じ穀物が繰り返し現れる場合も各言及の役割を区別します。\n"
            "2. **穀物名が省略されている場合**：文脈から照応先が明確に定まる場合に、"
            "その言及へ対応づけます。照応の根拠を含めて範囲を指定し、理由をレビュー欄に残します。\n"
            "3. **選択中の対象を固定して全文を確認**：対象に属する全関係を記録します。"
            "関係がないと確認できた場合はRelationを0件として確定します。曖昧な場合は理由を記して下書き保存します。\n"
            "4. **±25／±80 tokensはモデルの入力条件**：人手確認は全文ブロックを使います。\n"
            "5. **起点変更は誤同定の訂正**：元の言及・変更後の言及・理由を記録し、再確認します。"
            "別の価格表現を分析する場合は、その言及を別対象として扱います。"
        )
        st.caption("詳細とa003の例：docs/TARGET_RELATION_EVALUATION.md。反復言及・照応を含む既確認の対象も、正式goldの凍結前にこの基準で点検します。")
    if "review_rows" not in st.session_state:
        st.session_state.review_rows = list(read_jsonl(args.review))
        st.session_state.review_hash = sha256_file(args.review)
        resume_id = st.session_state.pop("resume_target_id", None)
        st.session_state.target_index = next((i for i, r in enumerate(st.session_state.review_rows)
                                              if r["target_id"] == resume_id), 0)
    rows = st.session_state.review_rows
    st.sidebar.caption(f"保存先：{args.review.resolve()}")
    pilot = any(r.get("ai_draft") for r in rows)
    if pilot:
        st.warning("AI補完版・人手未確認：提案と保留理由を確認してください。人手で確定するまで正解データにはなりません。")
    if "target_index" not in st.session_state:
        st.session_state.target_index = next((i for i, r in enumerate(rows)
                                              if r["target_id"] == st.session_state.get("active_target")), 0)
    st.sidebar.write(f"Confirmed: {sum(r.get('review_status') == 'confirmed' for r in rows)} / {len(rows)} targets")
    draft_labels = {"relation_proposal": "AI提案：関係あり", "negative_proposal": "AI提案：関係なし", "needs_review": "AI提案：判断保留"}
    index = st.sidebar.selectbox("Target", range(len(rows)), format_func=lambda i:
        f"{','.join(rows[i]['origin_sample_ids'])} · {rows[i]['doc_id']} · {rows[i].get('review_status') or draft_labels.get(rows[i].get('ai_draft', {}).get('disposition'), 'pending')}",
        key="target_index")
    original = rows[index]
    if st.sidebar.button("Reload saved review"):
        st.session_state.clear()
        st.session_state.resume_target_id = original["target_id"]
        st.rerun()
    notice = st.session_state.pop("save_notice", None)
    if notice:
        st.success(notice)
    key = original["target_id"]
    if st.session_state.get("active_target") != key:
        st.session_state.active_target = key
        st.session_state.draft = deepcopy(original)
        st.session_state.elapsed = 0.0
        st.session_state.running_since = None
    row = st.session_state.draft
    if row.get("ai_draft"):
        st.info(row["ai_draft"]["reason"])
        st.caption("上記は作成時のAI提案です。現在の判定基準に沿って確認してください。下の入力欄を編集しても提案メモは履歴として残ります。")
    if (row["doc_id"] == "bgu;6;1277" and row["grain_span"] == [349, 355]
            and row["source_block_text"][601:607] == "ὀλυρῶν"):
        st.warning(
            "a003の確認事項（2026-10-28）：選択中のὄλυραν [349,355)は返却する穀物の品質を述べる言及です。"
            "後続の『1アルタバにつき4ドラクマ』は換価条項に明示されたὀλυρῶν [601,607)に帰属します。"
            "現在の起点を保ち、他に対象への価格関係があるか全文で確認してください。"
            "この案内によるRelationの変更・人手確定は行っていません。"
        )
    if pilot and any(s in {"c002", "c013", "c014"} for s in row["origin_sample_ids"]):
        notices = {
            "c002": "再確認事項：χαλκοῦが独立したchalkous成分として残っています。銅貨建ての指定との区別を確認してください。",
            "c013": "再確認事項：欠損で判断不能というメモと、関係なしの確定が併存しています。",
            "c014": "再確認事項：金額にτριώβολονの成分が反映されていません。",
        }
        for sample in row["origin_sample_ids"]:
            if sample in notices:
                st.warning(notices[sample])
    if row.get("review_phase") == "post_llm_check":
        st.warning("Post-LLM correction sample: model predictions are visible. These sessions measure checking time, not independent gold annotation.")
    widget_key = key + str(st.session_state.get("edit_epoch", 0))
    st.subheader(f"{row['doc_id']} · {row['source_block_id']} · {', '.join(row['origin_sample_ids'])}")
    st.write("Review flags:", row.get("preparation_flags", []))
    if row.get("ai_review_recommendation") and not row.get("ai_draft"):
        st.info(row["ai_review_recommendation"])
    spans = grain_choices(row)
    chosen = st.selectbox("Target grain mention (source character positions)", spans,
                          index=spans.index(row["grain_span"]) if row["grain_span"] in spans else 0,
                          format_func=lambda s: f"{s}: {row['source_block_text'][max(0,s[0]-25):s[1]+40]}", key=widget_key+"anchor")
    if st.button("Apply changed target; require fresh confirmation"):
        st.session_state.draft = reanchor(row, chosen)
        st.session_state.edit_epoch = st.session_state.get("edit_epoch", 0)+1
        st.rerun()
    st.markdown(marked(row["source_block_text"], row["grain_span"]), unsafe_allow_html=True)
    for column, radius in zip(st.columns(2), (25, 80)):
        c = window_candidate(row, radius)
        with column:
            st.write(f"±{radius} tokens")
            st.markdown(marked(c["evidence_text"], row["grain_span"], c["evidence_char_start"]), unsafe_allow_html=True)
            with st.expander("Visible options and token positions"):
                st.json({"quantity": c["qty_options"], "money": c["money_options"], "tokens": c["token_inventory"]})
    with st.expander("Original human annotations and unconfirmed proposals"):
        st.json({"original": row.get("origin_annotations", []), "proposals": row.get("relation_proposals", [])})
    st.write("Timer records this review session only; pause before leaving. It does not replace original annotation time.")
    left, right = st.columns(2)
    if left.button("Start / resume timer") and st.session_state.running_since is None:
        st.session_state.running_since = time.monotonic()
    if right.button("Pause timer") and st.session_state.running_since is not None:
        st.session_state.elapsed += time.monotonic()-st.session_state.running_since
        st.session_state.running_since = None
    if st.button("Add another relation"):
        row["relations"].append(blank_relation(row))
        st.session_state.edit_epoch = st.session_state.get("edit_epoch", 0)+1
        st.rerun()
    if row["relations"]:
        remove = st.selectbox("Relation to remove", range(len(row["relations"])), key=widget_key+"remove")
        if st.button("Remove selected relation"):
            row["relations"].pop(remove)
            st.session_state.edit_epoch = st.session_state.get("edit_epoch", 0)+1
            st.rerun()
    with st.form(widget_key+"edit", enter_to_submit=False):
        edited = deepcopy(row)
        reviewed_contexts = []
        evidence_errors = []
        st.markdown("**原文だけで自動入力**")
        st.caption("上の原文で範囲を選択してコピーし、下に貼り付けてください。穀物・数量・金額・文字位置を自動で提案します。内容を確認してから保存してください。")
        auto_quote = st.text_area("自動入力する原文", key=widget_key+"-auto-quote", height=100,
                                  placeholder="τιμὴν τῆς ἀρτάβης ἑκάστης τῶν πυρῶν zzlbzz δραχμὰς δέκα 10")
        auto_destination = st.selectbox("自動入力の反映先", list(range(len(row["relations"]))) + [-1],
                                       format_func=lambda i: "新しいRelationを追加" if i == -1 else f"Relation {i+1}を置き換える",
                                       key=widget_key+"-auto-destination")
        apply_auto = st.form_submit_button("原文から自動入力（保存しない）")
        with st.expander("手動補正：数量・金額のstart/endを調べる"):
            st.caption("数量や金額の表現だけを原文からコピーしてください。文字位置を表示します。値や単位は下の表で補正できます。")
            locate_quote = st.text_area("位置を調べる原文の引用", key=widget_key+"-locate-quote", height=80)
            locate = st.form_submit_button("引用の位置を調べる（保存しない）")
            if locate:
                found = quote_spans(row["source_block_text"], locate_quote)
                if not found:
                    st.warning("引用が原文に一致しません。空白や改行記号を含めてコピーしてください。")
                else:
                    if len(found) > 1:
                        st.warning("複数の位置に一致します。前後の文脈で対象の箇所を確認してください。")
                    for n, (a, b) in enumerate(found):
                        st.write(f"start = {a} ／ end = {b}")
                        st.caption(occurrence_label(row["source_block_text"], (a, b), n))
        with st.expander("まとめて入力（コピー＆ペースト）", expanded=False):
            st.caption("指定形式の文章を貼り付けると、数量・金額・根拠の文字位置も自動設定します。反映後に内容を確認し、Save draft で保存してください。")
            paste = st.text_area("Relation一括入力", key=widget_key+"-paste", height=210,
                                 placeholder="対象: c001\n穀物: wheat\n穀物の引用: 原文の穀物名\n数量: 1 | artaba | 原文の数量表現\n金額: 10 | drachma | 原文の金額表現\n価格種別: stated_rate\n根拠: 原文の連続した引用")
            destination = st.selectbox("反映先", list(range(len(row["relations"]))) + [-1],
                                       format_func=lambda i: "新しいRelationを追加" if i == -1 else f"Relation {i+1}を置き換える",
                                       key=widget_key+"-paste-destination")
            apply_paste = st.form_submit_button("貼り付け内容を反映（保存しない）")
        st.caption("引用を貼り付けたら「入力を反映・プレビュー」。位置を数える必要はありません。プレビューはファイルに保存しません。")
        for i, relation in enumerate(edited["relations"]):
            st.markdown(f"**Relation {i+1}**")
            relation["grain_span"] = row["grain_span"]
            relation["commodity"] = st.text_input("Commodity", relation.get("commodity", ""), key=f"{widget_key}-{i}-commodity")
            for side in ("quantity", "money"):
                statuses = ["unresolved", "present", "unreadable", "unstated"]
                status = st.selectbox(side+" status", statuses, index=statuses.index(relation[side]["status"]), key=f"{widget_key}-{i}-{side}-status")
                components = [{"start": c["span"][0], "end": c["span"][1], "value": c["value"] or "",
                               "denomination": c["denomination"] or ""} for c in relation[side]["components"]]
                table = st.data_editor(pd.DataFrame(components, columns=["start", "end", "value", "denomination"]),
                                       num_rows="dynamic", key=f"{widget_key}-{i}-{side}-components")
                parsed = []
                for x in table.to_dict("records"):
                    if pd.isna(x["start"]) or pd.isna(x["end"]):
                        continue
                    parsed.append({"span": [int(x["start"]), int(x["end"])],
                                   "value": str(x["value"]).strip() if pd.notna(x["value"]) and str(x["value"]).strip() else None,
                                   "denomination": str(x["denomination"]).strip() if pd.notna(x["denomination"]) and str(x["denomination"]).strip() else None})
                relation[side] = {"status": status, "components": parsed}
                for component in parsed:
                    ca, cb = component["span"]
                    if 0 <= ca < cb <= len(row["source_block_text"]):
                        st.caption(f"{side} の範囲 [{ca}, {cb})：{row['source_block_text'][ca:cb]}")
            basis = ["unknown", "stated_rate", "derived_from_total"]
            relation["rate_basis"] = st.selectbox("Rate basis", basis, index=basis.index(relation.get("rate_basis") or "unknown"), key=f"{widget_key}-{i}-basis")
            evidence = relation.get("evidence_span") or row["grain_span"]
            relation["evidence_span"], evidence_error = evidence_picker(row["source_block_text"], evidence, f"{widget_key}-{i}")
            if evidence_error:
                evidence_errors.append(f"Relation {i+1}: {evidence_error}")
            if row.get("review_phase") == "post_llm_check":
                old_context = next((c for c in row.get('reviewed_contexts', row['prediction_before_review'].get('contexts', []))
                                    if c['relation_index'] == i), {})
                tags = st.multiselect("Context (separate from relation correctness)", sorted(CONTEXT_VALUES),
                                      default=old_context.get('context_tags', ['unknown']), key=f"{widget_key}-{i}-tags")
                strengths = sorted(LINKAGE_VALUES)
                strength = st.selectbox("Linkage strength", strengths,
                                        index=strengths.index(old_context.get('linkage_strength', 'weak')), key=f"{widget_key}-{i}-linkage")
                reviewed_contexts.append({'relation_index': i, 'context_tags': tags or ['unknown'], 'linkage_strength': strength})
        if row.get("review_phase") == "post_llm_check":
            edited['reviewed_contexts'] = reviewed_contexts
        edited["review_note"] = st.text_area("Review notes / reasons", row.get("review_note", ""))
        edited["reviewed_by"] = st.text_input("Reviewer", row.get("reviewed_by", ""))
        exhaustive = st.checkbox("I checked the target and all its relations in the full block; an empty list is an explicit negative judgment.", value=row.get("review_status") == "confirmed")
        preview = st.form_submit_button("入力を反映・プレビュー")
        save = st.form_submit_button("Save draft")
        confirm = st.form_submit_button("Confirm reviewed target")
    if apply_paste or apply_auto:
        try:
            auto_notes = []
            if apply_auto:
                imported, auto_notes = suggest_relation(auto_quote, row)
                destination = auto_destination
            else:
                imported = parse_relation_paste(paste, row)
            others = [r for i, r in enumerate(edited["relations"]) if i != destination]
            if any(linkage_key(r) == linkage_key(imported) for r in others):
                raise ValueError("同じ関係が既にあります。反映先で該当するRelationを選んでください。")
            if destination == -1:
                edited["relations"].append(imported)
            else:
                edited["relations"][destination] = imported
            edited.update(review_status="", coverage="unreviewed")
            st.session_state.draft = deepcopy(edited)
            st.session_state.edit_epoch = st.session_state.get("edit_epoch", 0)+1
            st.session_state.paste_notice = True
            st.session_state.auto_notes = auto_notes
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.pop("paste_notice", False):
        st.info("貼り付け内容を反映しました。まだ保存していません。各項目と引用範囲を確認して Save draft を押してください。")
    for note in st.session_state.pop("auto_notes", []):
        st.info(note)
    if preview or locate:
        # Retain every submitted edit without writing a review or resetting time.
        st.session_state.draft = deepcopy(edited)
        if not evidence_errors:
            st.info("プレビューを更新しました。保存するには Save draft または Confirm reviewed target を押してください。")
    if save or confirm:
        edited.update(review_status="confirmed" if confirm else "", coverage="exhaustive" if confirm and exhaustive else "unreviewed")
        try:
            if evidence_errors:
                raise ValueError("\n".join(evidence_errors))
            if confirm and not exhaustive:
                raise ValueError("exhaustive review must be explicitly confirmed")
            seconds = st.session_state.elapsed
            if st.session_state.running_since is not None:
                seconds += time.monotonic()-st.session_state.running_since
            result = save_review(args.review, st.session_state.review_hash, key, edited, seconds)
            st.session_state.clear()
            # Saving can reorder rows or merge targets; restore by saved identity,
            # not the old list index. Fresh form state/timing still comes from disk.
            st.session_state.resume_target_id = result["target_id"]
            st.session_state.save_notice = ("保存しました。統合された対象の関係を再確認してください。" if result["merged"]
                                            else "確定して保存しました。" if confirm else "下書きを保存しました。")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    for i, r in enumerate(edited["relations"]):
        errors = validate_relation(r, row)
        if errors:
            st.warning(f"Relation {i+1}: "+"; ".join(errors))


if __name__ == "__main__":
    main()

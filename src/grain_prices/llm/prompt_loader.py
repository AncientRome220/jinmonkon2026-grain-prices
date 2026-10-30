#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM prompt loader.

Reads a versioned prompt resource and provides formatted system/user prompt
strings for any LLM backend.

Usage:
    from grain_prices.llm.prompt_loader import load_prompts, build_user_prompt

    system_prompt, user_template = load_prompts()
    user_prompt = build_user_prompt(user_template, doc_id="p.oxy;1;123",
                                    grain_form="πυρου", window_text="...")
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_DEFAULT_PROMPT = _PROMPT_DIR / "price_adjudication_canonical_v13_1.md"


def load_prompts(
    prompt_path: str | Path | None = None,
) -> Tuple[str, str]:
    """
    Load system prompt and user prompt template from the markdown file.

    Returns:
        (system_prompt, user_prompt_template)
    """
    path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Extract system prompt (between "## System Prompt" and next "##")
    system_match = re.search(
        r"## System Prompt\s*\n(.*?)(?=\n## |\Z)",
        text,
        re.DOTALL,
    )
    system_prompt = system_match.group(1).strip() if system_match else ""

    # Extract user prompt template (between "## User Prompt Template" and next "##")
    user_match = re.search(
        r"## User Prompt Template\s*\n(.*?)(?=\n## |\Z)",
        text,
        re.DOTALL,
    )
    user_template = user_match.group(1).strip() if user_match else ""

    return system_prompt, user_template


def build_user_prompt(
    template: str,
    doc_id: str,
    grain_form: str,
    window_text: str,
) -> str:
    """
    Format the user prompt template with document-specific values.

    Uses Python str.format() with named placeholders:
      {doc_id}, {grain_form}, {window_text}
    Escaped braces {{ }} in the template are preserved for JSON examples.
    """
    return template.format(
        doc_id=doc_id,
        grain_form=grain_form,
        window_text=window_text,
    )


if __name__ == "__main__":
    # Quick test
    sys_prompt, user_tmpl = load_prompts()
    print("=== System Prompt ===")
    print(sys_prompt[:200], "...")
    print()
    print("=== User Template (first 300 chars) ===")
    print(user_tmpl[:300], "...")
    print()
    # Test formatting
    user = build_user_prompt(user_tmpl, "test;1;1", "πυρου", "test window text")
    print("=== Formatted (first 300 chars) ===")
    print(user[:300], "...")

"""
Tests for RESPONSE FORMAT RULES in MRPL Sovereign AI Workbench.

Verifies:
1. Prompt templates embed the Zero Markdown Asterisks Rules and clean structured text rules.
2. Prompt forbids `*` and `**` for bold, italics, or bullets.
3. Plain labels (Purpose:, Function:, Role:) without asterisks.
4. Deterministic response sanitization in `strip_raw_ui_markers`.
"""

from backend.agent.prompts import REASONING_PROMPT, GENERAL_CHAT_PROMPT, SYNTHESIS_PROMPT
from backend.agent.graph import strip_raw_ui_markers


def test_reasoning_prompt_contains_zero_asterisk_rules():
    """Verify REASONING_PROMPT embeds zero asterisk format rules."""
    rendered = REASONING_PROMPT.format(
        current_query="what are the main units in mrpl",
        retrieved_context="Refining units: CDU, HCU, PFCCU",
        conversation_history="No previous conversation.",
    )

    # Zero Asterisks & Clean Structured Text Rules
    assert "## RESPONSE FORMAT RULES (ZERO MARKDOWN ASTERISKS — CLEAN STRUCTURED TEXT):" in rendered
    assert "1. ZERO ASTERISKS: NEVER use '*' or '**' anywhere in your response." in rendered
    assert "NEVER output bold markdown syntax" in rendered
    assert "NEVER output italic markdown syntax" in rendered
    assert "NEVER output asterisk bullets" in rendered
    assert "2. NO WALL OF TEXT: Keep paragraphs short" in rendered
    assert "3. NUMBERED SECTIONS: For lists of multiple technical items/units, use clear numbered sections" in rendered
    assert "4. PLAIN LABELS: Use plain labels without asterisks on their own lines" in rendered
    assert "Purpose: [concise description from evidence]" in rendered
    assert "Function: [key operational function from evidence]" in rendered
    assert "Role: [role in refinery flows from evidence]" in rendered
    assert "7. PRESERVE TERMINOLOGY: Preserve technical names, acronyms" in rendered
    assert "10. CITATIONS: Cite the relevant source at the end of the section or paragraph" in rendered

    # Adaptive format
    assert "ADAPTIVE FORMAT:" in rendered
    assert "Simple factual answer: 1 short paragraph" in rendered
    assert "List of items: Numbered list or clean plain-line items." in rendered
    assert '"Explain each" / "in detail" / follow-ups: Numbered section for each item with plain Purpose:, Function:, Role: labels.' in rendered


def test_synthesis_prompt_contains_zero_asterisk_rules():
    """Verify SYNTHESIS_PROMPT embeds zero asterisk readability rules."""
    rendered = SYNTHESIS_PROMPT.format(
        current_query="compare process units",
        tool_results="[{'tool': 'rag_search'}]",
    )
    assert "## RESPONSE FORMAT RULES (ZERO MARKDOWN ASTERISKS — CLEAN STRUCTURED TEXT):" in rendered
    assert "1. ZERO ASTERISKS: NEVER use '*' or '**' for bold, italics, or bullets." in rendered


def test_general_chat_prompt_contains_zero_asterisk_rules():
    """Verify GENERAL_CHAT_PROMPT embeds zero asterisk readability rules."""
    rendered = GENERAL_CHAT_PROMPT.format(
        current_query="hello",
        conversation_history="No previous conversation.",
    )
    assert "## RESPONSE FORMAT RULES (ZERO MARKDOWN ASTERISKS — CLEAN STRUCTURED TEXT):" in rendered
    assert "1. ZERO ASTERISKS: NEVER use '*' or '**' anywhere in your response." in rendered


def test_strip_raw_ui_markers_sanitizes_markdown_asterisks():
    """Verify strip_raw_ui_markers cleans stray markdown asterisks from assistant text while preserving code blocks."""
    raw_text = (
        "Main Units in MRPL\n\n"
        "1. **Crude Unit**\n"
        "- **Purpose:** Primary processing of crude oil.\n"
        "- **Function:** Separates crude oil into various products.\n"
        "- **Role:** Primary processing.\n\n"
        "2. **Hydrocracker Unit**\n"
        "* **Purpose:** Upgrades heavy oil.\n"
        "* **Function:** Catalytic cracking.\n"
    )

    cleaned = strip_raw_ui_markers(raw_text)

    # Asterisks for bold/italic/bullets must be stripped
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "1. Crude Unit" in cleaned
    assert "Purpose: Primary processing of crude oil." in cleaned
    assert "Function: Separates crude oil into various products." in cleaned
    assert "Role: Primary processing." in cleaned
    assert "2. Hydrocracker Unit" in cleaned


def test_strip_raw_ui_markers_preserves_code_block_math():
    """Verify strip_raw_ui_markers does NOT strip asterisks inside code blocks (e.g. multiplication/exponentiation)."""
    raw_code = (
        "Here is the calculation script:\n"
        "```python\n"
        "def compute(x):\n"
        "    return x * 2 + (x ** 3)\n"
        "```\n"
        "**Note:** Script executed successfully."
    )

    cleaned = strip_raw_ui_markers(raw_code)

    assert "x * 2 + (x ** 3)" in cleaned
    assert "Note: Script executed successfully." in cleaned
    assert "**Note:**" not in cleaned

"""
MRPL Sovereign AI Workbench — Speech Rendering Layer.
Transforms rich markdown/technical LLM outputs into natural conversational speech text.
Strips markdown symbols, code fences, citations, JSON, and UI metadata while strictly
preserving technical tags, equipment IDs, numeric values, and engineering units.
"""

import re


def render_speech_text(text: str) -> str:
    """
    Convert full markdown/technical response into natural speech-friendly text.

    Preserves:
      - Equipment tags (e.g., '11-P-101A', 'CDU-101')
      - Refining units (e.g., 'CDU', 'HCU', 'PFCCU', 'VDU', 'API 510')
      - Numerical values (e.g., '12.4', '150', '350')
      - Engineering units (e.g., 'bar', '°C', 'kg/cm²', 'MTPA', 'ppm')

    Removes:
      - Fenced code blocks ```...``` (replaces with concise notification)
      - Inline backticks `code`
      - Markdown headings #, ##, ###
      - Markdown bold/italics **text**, *text*
      - Markdown links [label](url) -> label
      - Citation footnotes [1], [Page 12], [P&ID Drawing], [Ref ...]
      - JSON/YAML formatting blocks
      - Bullet points and lists (- , * , 1. ) -> converted to clean pause separators
      - Robotic filler phrases ("According to the retrieved documents", "Based on my knowledge base")
    """
    if not text:
        return ""

    s = text

    # 1. Replace fenced code blocks with natural audio placeholder
    s = re.sub(r"```(?:python|sql|bash|json|yaml|sh)?[\s\S]*?```", " Code snippet shown on screen. ", s)

    # 2. Strip JSON-like objects or metadata dictionaries
    s = re.sub(r"\{[\s\S]*?\}", "", s)

    # 3. Strip inline code marks
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # 4. Strip markdown links [text](url) -> text and raw URLs
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)

    # 5. Strip citation footnotes and parenthetical source citations:
    # e.g., [1], [Page 12], (Source: 36th Annual Report, p. 10), (Document: ...), Source: ...
    s = re.sub(r"\[(?:\d+|Page\s+\d+|Ref\s+[^\]]+|Table\s+\d+|P&ID\s+[^\]]+)\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\((?:Source|Doc|Document|Ref|Table|Page)\s*:[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:Source|Doc|Document)\s*:\s*[^,\n.]*(?:,\s*p\.\s*\d+)?", "", s, flags=re.IGNORECASE)

    # Strip trailing "Sources:" or "References:" list entirely
    s = re.sub(r"(?:\n|\A)\s*(?:Sources|References|Citations)\s*:\s*[\s\S]*$", "", s, flags=re.IGNORECASE)

    # Strip HTML / XML tags
    s = re.sub(r"<[^>]+>", "", s)

    # 6. Strip markdown headings #, ##, ###
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)

    # 7. Strip markdown bold / italics
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)

    # 8. Convert bullet points and numbered list items into natural flowing pauses
    # e.g., "- Pressure: 5 bar" -> "Pressure: 5 bar."
    s = re.sub(r"^\s*[-*•]\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s*", "", s, flags=re.MULTILINE)

    # 9. Strip robotic prefixes that make voice sound unnatural
    robotic_prefixes = [
        r"(?:(?:Based on|According to) the (?:retrieved )?(?:context|information|documents|records)[,\s]*)",
        r"(?:As an AI assistant[,\s]*)",
        r"(?:In response to your query[,\s]*)",
    ]
    for pattern in robotic_prefixes:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()

    # 10. Normalize whitespace, remove duplicate punctuation while preserving decimals like 12.4
    s = re.sub(r"([.,!?;:])\1+", r"\1", s)
    s = re.sub(r"\s*([,!?;:])\s*", r"\1 ", s)
    s = re.sub(r"(?<!\d)\.(?!\d)\s*", ". ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 11. Acronym pronunciation expansion for natural acoustics
    # (keeps equipment tags verbatim like 11-P-101A, but splits isolated acronyms like CDU -> C D U)
    acronym_map = {
        r"\bCDU\b": "C D U",
        r"\bVDU\b": "V D U",
        r"\bHCU\b": "H C U",
        r"\bPFCCU\b": "P F C C U",
        r"\bFCCU\b": "F C C U",
        r"\bP&ID\b": "P and I D",
        r"\bPID\b": "P and I D",
        r"\bHSE\b": "H S E",
        r"\bGRM\b": "G R M",
        r"\bAPI\s+(\d+)\b": r"A P I \1",
    }
    for pattern, repl in acronym_map.items():
        s = re.sub(pattern, repl, s)

    return s.strip()

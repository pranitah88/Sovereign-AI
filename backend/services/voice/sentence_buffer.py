"""
MRPL Sovereign AI Workbench — Sentence Chunk Buffer.
Buffers streaming LLM tokens into natural conversational sentence chunks
to enable immediate Text-to-Speech synthesis before the entire LLM response finishes.
"""

import re


# Delimiters for natural phrase and sentence streaming
# Major: sentence-ending punctuation across English, Hindi, and Marathi (. ! ? \n ।)
MAJOR_DELIMITERS = re.compile(r'([.!?\n।]+(?:\s+|$))')
# Minor: clause/phrase delimiters (, : ; —)
CLAUSE_DELIMITERS = re.compile(r'([,;:—]+(?:\s+|$))')
# Natural conversational conjunction breaks when preceding phrase is substantial
CONJUNCTION_BREAK = re.compile(r'(\s+(?:and|but|which|where|while|because)\s+)', re.IGNORECASE)


class SentenceChunkBuffer:
    """
    Accumulates streaming tokens and yields natural conversational phrase and clause chunks.
    Allows immediate Text-to-Speech synthesis on meaningful phrases (4-8 words) without
    waiting for an entire multi-clause sentence to finish generating.
    """

    def __init__(self, min_words: int = 4, max_words: int = 25):
        self.min_words = min_words
        self.max_words = max_words
        self.buffer = ""

    def add_token(self, token: str) -> list[str]:
        """
        Add a newly arrived token from the LLM stream.
        Returns a list of completed phrases/sentences ready for TTS synthesis.
        """
        self.buffer += token
        return self._extract_ready_chunks(is_final=False)

    def flush(self) -> list[str]:
        """
        Flush remaining buffer at the end of the LLM stream.
        Returns any remaining text chunk for final TTS synthesis.
        """
        return self._extract_ready_chunks(is_final=True)

    def _extract_ready_chunks(self, is_final: bool = False) -> list[str]:
        ready_chunks = []

        while True:
            # Collect potential split candidates in the current buffer
            punc_matches = []
            for m in list(MAJOR_DELIMITERS.finditer(self.buffer)) + list(CLAUSE_DELIMITERS.finditer(self.buffer)):
                punc_matches.append((m.start(), m.end(), m.group(0), "punc"))
            for m in CONJUNCTION_BREAK.finditer(self.buffer):
                punc_matches.append((m.start(), m.start(), m.group(0), "conj"))

            punc_matches.sort(key=lambda x: x[0])

            found_split = False
            for start_idx, end_idx, matched_str, match_type in punc_matches:
                candidate = self.buffer[:start_idx if match_type == "conj" else end_idx].strip()
                words = candidate.split()
                n_words = len(words)

                if match_type == "punc" and any(p in matched_str for p in ".!?\n।"):
                    # Major sentence boundary: allow if at least 3 words, or if final
                    if n_words >= min(3, self.min_words) or is_final:
                        ready_chunks.append(candidate)
                        self.buffer = self.buffer[end_idx:].lstrip()
                        found_split = True
                        break
                elif match_type in ("punc", "conj"):
                    # Minor clause or natural conjunction: require soft minimum words (4-8)
                    if n_words >= self.min_words:
                        ready_chunks.append(candidate)
                        self.buffer = self.buffer[start_idx if match_type == "conj" else end_idx:].lstrip()
                        found_split = True
                        break

            if found_split:
                continue

            # Fallback if buffer is getting too long without any punctuation delimiter
            words = self.buffer.split()
            if len(words) >= self.max_words and not is_final:
                space_pos = self.buffer.rfind(" ")
                if space_pos != -1:
                    candidate = self.buffer[:space_pos].strip()
                    if len(candidate.split()) >= self.min_words:
                        ready_chunks.append(candidate)
                        self.buffer = self.buffer[space_pos + 1:].lstrip()
                        continue

            if is_final:
                rem = self.buffer.strip()
                if rem:
                    ready_chunks.append(rem)
                self.buffer = ""
            break

        return [c for c in ready_chunks if c]


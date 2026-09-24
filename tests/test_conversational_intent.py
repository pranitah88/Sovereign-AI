"""
Tests for MRPL Sovereign AI — Conversational Intent Detection & Local Response.

Validates all 10 requirements from the specification:
 1. Responses vary.
 2. No immediate duplicate response for the same intent.
 3. Responses remain natural (short, no system prompt leakage).
 4. Responses are short.
 5. No RAG is triggered.
 6. No Ollama call is triggered.
 7. TTS speaks the selected response. (verified via integration)
 8. Voice returns to LISTENING after TTS completes. (verified via integration)
 9. Technical questions still use the complete governed pipeline.
10. "What is MRPL?" is never classified as a greeting.
"""

import pytest
from backend.services.voice.conversational_intent import (
    ConversationalIntent,
    detect_conversational_intent,
    get_rotator,
    RESPONSE_POOLS,
)


@pytest.fixture(autouse=True)
def reset_rotator():
    """Reset the response rotator before each test for deterministic results."""
    get_rotator().reset()
    yield
    get_rotator().reset()


# ════════════════════════════════════════════════════════════════════════════
# 1. Responses vary — rotating through the pool
# ════════════════════════════════════════════════════════════════════════════

class TestResponseVariety:
    def test_greeting_responses_cycle_through_pool(self):
        """Multiple 'Hi' inputs should produce different responses."""
        responses = []
        for _ in range(len(RESPONSE_POOLS["GREETING"])):
            result = detect_conversational_intent("Hi")
            assert result.is_conversational
            responses.append(result.response_text)

        # All pool entries should appear
        assert set(responses) == set(RESPONSE_POOLS["GREETING"])

    def test_how_are_you_responses_cycle(self):
        responses = []
        for _ in range(len(RESPONSE_POOLS["HOW_ARE_YOU"])):
            result = detect_conversational_intent("How are you?")
            assert result.is_conversational
            responses.append(result.response_text)

        assert set(responses) == set(RESPONSE_POOLS["HOW_ARE_YOU"])

    def test_farewell_responses_cycle(self):
        responses = []
        for _ in range(len(RESPONSE_POOLS["FAREWELL"])):
            result = detect_conversational_intent("Bye")
            assert result.is_conversational
            responses.append(result.response_text)

        assert set(responses) == set(RESPONSE_POOLS["FAREWELL"])


# ════════════════════════════════════════════════════════════════════════════
# 2. No immediate duplicate response for the same intent
# ════════════════════════════════════════════════════════════════════════════

class TestNoDuplicates:
    def test_no_consecutive_greeting_duplicate(self):
        r1 = detect_conversational_intent("Hi")
        r2 = detect_conversational_intent("Hi")
        assert r1.response_text != r2.response_text, (
            f"Consecutive duplicate: {r1.response_text!r}"
        )

    def test_no_consecutive_how_are_you_duplicate(self):
        r1 = detect_conversational_intent("How are you?")
        r2 = detect_conversational_intent("How are you?")
        assert r1.response_text != r2.response_text

    def test_no_consecutive_farewell_duplicate(self):
        r1 = detect_conversational_intent("Bye")
        r2 = detect_conversational_intent("Bye")
        assert r1.response_text != r2.response_text

    def test_extended_no_immediate_repeats(self):
        """20 consecutive greetings — no two adjacent should be the same."""
        prev = None
        for _ in range(20):
            result = detect_conversational_intent("Hello")
            assert result.response_text != prev, (
                f"Consecutive duplicate at iteration: {result.response_text!r}"
            )
            prev = result.response_text


# ════════════════════════════════════════════════════════════════════════════
# 3 & 4. Responses are natural and short
# ════════════════════════════════════════════════════════════════════════════

class TestNaturalShortResponses:
    def test_greeting_response_short(self):
        result = detect_conversational_intent("Hi")
        assert len(result.response_text) < 60
        assert "refinery" not in result.response_text.lower()
        assert "engineering reports" not in result.response_text.lower()
        assert "technical queries" not in result.response_text.lower()

    def test_how_are_you_no_system_prompt_leakage(self):
        result = detect_conversational_intent("How are you?")
        assert "functioning optimally" not in result.response_text.lower()
        assert "refinery operations" not in result.response_text.lower()

    def test_farewell_response_short(self):
        result = detect_conversational_intent("Bye")
        assert len(result.response_text) < 60

    def test_all_pool_responses_are_short(self):
        for key, pool in RESPONSE_POOLS.items():
            for resp in pool:
                assert len(resp) < 80, f"Response too long in {key}: {resp!r}"
                word_count = len(resp.split())
                assert word_count <= 15, f"Too many words in {key}: {resp!r}"


# ════════════════════════════════════════════════════════════════════════════
# 5 & 6. No RAG or Ollama triggered (is_conversational must be True)
# ════════════════════════════════════════════════════════════════════════════

class TestNoHeavyPipeline:
    @pytest.mark.parametrize("text", [
        "Hi", "Hello", "Hey", "Hey there", "Hi there",
        "How are you?", "How are you doing?",
        "Good morning", "Good afternoon", "Good evening",
        "Bye", "Goodbye", "See you", "Take care",
    ])
    def test_conversational_inputs_bypass_pipeline(self, text):
        result = detect_conversational_intent(text)
        assert result.is_conversational is True, f"{text!r} was not classified as conversational"
        assert result.response_text is not None
        assert result.intent != ConversationalIntent.NORMAL_QUERY


# ════════════════════════════════════════════════════════════════════════════
# 9. Technical questions use the complete governed pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestTechnicalQueriesNotIntercepted:
    @pytest.mark.parametrize("text", [
        "What is MRPL?",
        "Hi, what is MRPL?",
        "Hello, what is the pressure of 11-P-101A?",
        "Hey, tell me about the CDU unit",
        "Good morning, show me the audit report",
        "What is the temperature of the reactor?",
        "How is the pump performing?",
        "Compare the two documents",
        "Summarize the inspection report",
        "Check the status of the compressor",
    ])
    def test_technical_queries_are_normal(self, text):
        result = detect_conversational_intent(text)
        assert result.intent == ConversationalIntent.NORMAL_QUERY, (
            f"{text!r} was incorrectly classified as {result.intent.value}"
        )
        assert result.is_conversational is False
        assert result.response_text is None


# ════════════════════════════════════════════════════════════════════════════
# 10. "What is MRPL?" is never classified as a greeting
# ════════════════════════════════════════════════════════════════════════════

class TestWhatIsMRPLNotGreeting:
    def test_plain(self):
        result = detect_conversational_intent("What is MRPL?")
        assert result.intent == ConversationalIntent.NORMAL_QUERY

    def test_with_greeting_prefix(self):
        result = detect_conversational_intent("Hi, what is MRPL?")
        assert result.intent == ConversationalIntent.NORMAL_QUERY

    def test_with_hello_prefix(self):
        result = detect_conversational_intent("Hello, what is MRPL?")
        assert result.intent == ConversationalIntent.NORMAL_QUERY


# ════════════════════════════════════════════════════════════════════════════
# Edge cases: false-positive guards
# ════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveGuards:
    def test_why_did_he_say_goodbye(self):
        result = detect_conversational_intent("Why did he say goodbye?")
        assert result.intent == ConversationalIntent.NORMAL_QUERY

    def test_i_said_goodbye_not_high(self):
        result = detect_conversational_intent("I said goodbye, not high")
        assert result.intent == ConversationalIntent.NORMAL_QUERY

    def test_empty_input(self):
        result = detect_conversational_intent("")
        assert result.intent == ConversationalIntent.NORMAL_QUERY
        assert result.is_conversational is False

    def test_whitespace_only(self):
        result = detect_conversational_intent("   ")
        assert result.intent == ConversationalIntent.NORMAL_QUERY


# ════════════════════════════════════════════════════════════════════════════
# Intent classification correctness
# ════════════════════════════════════════════════════════════════════════════

class TestIntentClassification:
    def test_hi_is_greeting(self):
        assert detect_conversational_intent("Hi").intent == ConversationalIntent.GREETING

    def test_hello_is_greeting(self):
        assert detect_conversational_intent("Hello").intent == ConversationalIntent.GREETING

    def test_hey_is_greeting(self):
        assert detect_conversational_intent("Hey").intent == ConversationalIntent.GREETING

    def test_how_are_you_is_how_are_you(self):
        assert detect_conversational_intent("How are you?").intent == ConversationalIntent.HOW_ARE_YOU

    def test_good_morning_is_time_greeting(self):
        assert detect_conversational_intent("Good morning").intent == ConversationalIntent.TIME_GREETING

    def test_good_afternoon_is_time_greeting(self):
        assert detect_conversational_intent("Good afternoon").intent == ConversationalIntent.TIME_GREETING

    def test_good_evening_is_time_greeting(self):
        assert detect_conversational_intent("Good evening").intent == ConversationalIntent.TIME_GREETING

    def test_bye_is_farewell(self):
        assert detect_conversational_intent("Bye").intent == ConversationalIntent.FAREWELL

    def test_goodbye_is_farewell(self):
        assert detect_conversational_intent("Goodbye").intent == ConversationalIntent.FAREWELL

    def test_see_you_is_farewell(self):
        assert detect_conversational_intent("See you").intent == ConversationalIntent.FAREWELL


# ════════════════════════════════════════════════════════════════════════════
# Deterministic rotation wraps around
# ════════════════════════════════════════════════════════════════════════════

class TestDeterministicRotation:
    def test_greeting_wraps_around(self):
        """After exhausting all pool entries, rotation wraps back."""
        pool_size = len(RESPONSE_POOLS["GREETING"])
        first_round = []
        for _ in range(pool_size):
            r = detect_conversational_intent("Hi")
            first_round.append(r.response_text)

        # Next call should wrap around
        r_wrap = detect_conversational_intent("Hi")
        assert r_wrap.response_text in RESPONSE_POOLS["GREETING"]
        # And should not equal the last of the first round
        assert r_wrap.response_text != first_round[-1]

    def test_time_greeting_morning(self):
        r = detect_conversational_intent("Good morning")
        assert r.intent == ConversationalIntent.TIME_GREETING
        assert "morning" in r.response_text.lower()

    def test_time_greeting_afternoon(self):
        r = detect_conversational_intent("Good afternoon")
        assert r.intent == ConversationalIntent.TIME_GREETING
        assert "afternoon" in r.response_text.lower()

    def test_time_greeting_evening(self):
        r = detect_conversational_intent("Good evening")
        assert r.intent == ConversationalIntent.TIME_GREETING
        assert "evening" in r.response_text.lower()


# ════════════════════════════════════════════════════════════════════════════
# Full test sequence from specification
# ════════════════════════════════════════════════════════════════════════════

class TestSpecificationSequence:
    """Runs the exact sequence from the requirements document."""

    def test_full_sequence(self):
        inputs = [
            "Hi", "Hi", "Hello", "Hey",
            "How are you?", "How are you?",
            "Good morning",
            "Bye", "Bye",
        ]
        responses = []
        for text in inputs:
            result = detect_conversational_intent(text)
            assert result.is_conversational, f"{text!r} not conversational"
            assert result.response_text is not None
            responses.append((text, result.response_text, result.intent.value))

        # Verify no consecutive same-intent duplicate
        # "Hi", "Hi" → two different responses
        assert responses[0][1] != responses[1][1], "Hi/Hi produced same response"
        # "How are you?", "How are you?" → two different responses
        assert responses[4][1] != responses[5][1], "How are you?/How are you? produced same response"
        # "Bye", "Bye" → two different responses
        assert responses[7][1] != responses[8][1], "Bye/Bye produced same response"

        # All responses should be short
        for text, resp, intent in responses:
            assert len(resp) < 80, f"Response too long for {text!r}: {resp!r}"

        # Print for manual inspection
        for text, resp, intent in responses:
            print(f"  [{intent}] {text!r:25s} → {resp!r}")

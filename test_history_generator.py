#!/usr/bin/env python3
"""
test_history_generator.py — Contract tests for history_story_generator.py

Tested contracts (no real LLM calls — all LLM calls are mocked):
  1. valid_output          — well-formed story passes validate_story_contract
  2. sentence_order        — sentences array preserves generation order
  3. malformed_output      — bad JSON / missing keys / empty sentences raise RuntimeError
  4. leakage               — forbidden metadata in sentences is detected and rejected
  5. invalid_topic         — blank / non-string topics raise ValueError / TypeError

Run with:
    python test_history_generator.py
    python -m pytest test_history_generator.py -v
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import llm_client  # noqa: E402  (needed so the import inside generator works)
from history_story_generator import (  # noqa: E402
    generate_history_story,
    validate_story_contract,
    find_sentence_metadata_artifacts,
    ALLOWED_LENGTH_MODES,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_valid_story(**overrides) -> dict:
    """Return a minimal valid story dict."""
    story = {
        "title": "The Rise of Kano",
        "sentences": [
            "In the seventh century, a small settlement emerged near the Jakara stream.",
            "Its position on the Saharan trade routes made it indispensable to merchants.",
            "Salt, cloth, and gold flowed through its markets for centuries.",
            "By the fifteenth century, Kano had become a major Islamic centre of learning.",
            "Its legacy endures in the city's ancient walls, still standing today.",
        ],
    }
    story.update(overrides)
    return story


def _llm_responses(*json_payloads):
    """Return a side_effect list for call_llm: first call returns research text,
    subsequent calls return each json_payload serialised (or as-is if str)."""
    _research = "• Founded circa 7th century CE.\n• Trans-Saharan trade hub."
    responses = [_research]
    for p in json_payloads:
        responses.append(p if isinstance(p, str) else json.dumps(p))
    return responses


# ════════════════════════════════════════════════════════════════════════════
# 1. Valid output
# ════════════════════════════════════════════════════════════════════════════

class TestValidOutput(unittest.TestCase):
    """generate_history_story returns a well-formed {title, sentences} dict."""

    def _run(self, topic="The founding of Kano", length_mode="short", extra_keys=None):
        story_payload = _make_valid_story()
        if extra_keys:
            story_payload.update(extra_keys)
        side_effects = _llm_responses(story_payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            return generate_history_story(topic, length_mode=length_mode)

    def test_returns_dict(self):
        result = self._run()
        self.assertIsInstance(result, dict)

    def test_exact_keys(self):
        result = self._run()
        self.assertEqual(set(result.keys()), {"title", "sentences"})

    def test_title_is_nonempty_str(self):
        result = self._run()
        self.assertIsInstance(result["title"], str)
        self.assertTrue(result["title"].strip())

    def test_sentences_is_nonempty_list(self):
        result = self._run()
        self.assertIsInstance(result["sentences"], list)
        self.assertGreater(len(result["sentences"]), 0)

    def test_every_sentence_is_nonempty_str(self):
        result = self._run()
        for i, s in enumerate(result["sentences"]):
            with self.subTest(i=i):
                self.assertIsInstance(s, str)
                self.assertTrue(s.strip(), f"Sentence {i + 1} is blank")

    def test_json_serialisable(self):
        result = self._run()
        serialised = json.dumps(result)
        round_tripped = json.loads(serialised)
        self.assertEqual(result, round_tripped)

    def test_topic_key_not_present(self):
        """The old contract had a 'topic' key; the new one does not."""
        result = self._run()
        self.assertNotIn("topic", result)

    def test_topic_key_stripped_if_model_echoes_it(self):
        """If the model echoes back a 'topic' field, the generator strips it."""
        story_with_topic = _make_valid_story()
        story_with_topic["topic"] = "the founding of kano"  # model echoed it
        side_effects = _llm_responses(story_with_topic)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            result = generate_history_story("The founding of Kano")
        self.assertNotIn("topic", result)

    def test_passes_validate_story_contract(self):
        result = self._run()
        # Should not raise
        validate_story_contract(result)

    def test_long_mode_accepted(self):
        long_story = _make_valid_story(
            sentences=[f"Sentence number {i + 1} about history." for i in range(60)]
        )
        side_effects = _llm_responses(long_story)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            result = generate_history_story("The Roman Empire", length_mode="long")
        self.assertGreaterEqual(len(result["sentences"]), 60)

    def test_two_llm_calls_made(self):
        """Exactly two calls: research + story."""
        story_payload = _make_valid_story()
        side_effects = _llm_responses(story_payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects) as mock_call:
            generate_history_story("The founding of Kano")
        self.assertEqual(mock_call.call_count, 2)


# ════════════════════════════════════════════════════════════════════════════
# 2. Sentence order
# ════════════════════════════════════════════════════════════════════════════

class TestSentenceOrder(unittest.TestCase):
    """Sentences are returned in exactly the order the model produced them."""

    def _run(self, sentences):
        payload = _make_valid_story(sentences=sentences)
        side_effects = _llm_responses(payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            return generate_history_story("Order test topic")

    def test_order_preserved(self):
        ordered = [f"This is sentence number {i + 1}." for i in range(7)]
        result = self._run(ordered)
        self.assertEqual(result["sentences"], ordered)

    def test_single_sentence_preserved(self):
        single = ["One and only sentence about history."]
        result = self._run(single)
        self.assertEqual(result["sentences"], single)

    def test_first_sentence_unchanged(self):
        sentences = [
            "The very first sentence must not be reordered.",
            "Second sentence.",
            "Third sentence.",
        ]
        result = self._run(sentences)
        self.assertEqual(result["sentences"][0], sentences[0])

    def test_last_sentence_unchanged(self):
        sentences = [
            "First sentence.",
            "Middle sentence.",
            "The final sentence must appear last.",
        ]
        result = self._run(sentences)
        self.assertEqual(result["sentences"][-1], sentences[-1])


# ════════════════════════════════════════════════════════════════════════════
# 3. Malformed output
# ════════════════════════════════════════════════════════════════════════════

class TestMalformedOutput(unittest.TestCase):
    """Broken / incomplete LLM responses raise RuntimeError."""

    def _run_with_story_response(self, story_raw: str):
        research = "• Some research notes."
        with patch.object(llm_client, "call_llm", side_effect=[research, story_raw]):
            generate_history_story("Some topic")

    def test_non_json_raises(self):
        with self.assertRaises(RuntimeError):
            self._run_with_story_response("This is not JSON at all.")

    def test_json_array_raises(self):
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(["not", "a", "dict"]))

    def test_missing_title_raises(self):
        payload = {"sentences": ["One sentence."]}
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(payload))

    def test_missing_sentences_raises(self):
        payload = {"title": "A Title"}
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(payload))

    def test_empty_sentences_list_raises(self):
        payload = {"title": "A Title", "sentences": []}
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(payload))

    def test_blank_sentence_raises(self):
        payload = {"title": "A Title", "sentences": ["Good sentence.", "   ", "Another good one."]}
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(payload))

    def test_empty_title_raises(self):
        payload = {"title": "   ", "sentences": ["A sentence."]}
        with self.assertRaises(RuntimeError):
            self._run_with_story_response(json.dumps(payload))

    def test_markdown_fenced_json_is_accepted(self):
        """```json ... ``` wrapping must be stripped gracefully (not an error)."""
        valid = _make_valid_story()
        fenced = f"```json\n{json.dumps(valid)}\n```"
        research = "• Some research notes."
        with patch.object(llm_client, "call_llm", side_effect=[research, fenced]):
            result = generate_history_story("Some topic")
        self.assertIn("title", result)


# ════════════════════════════════════════════════════════════════════════════
# 4. Leakage detection
# ════════════════════════════════════════════════════════════════════════════

class TestLeakageDetection(unittest.TestCase):
    """Sentences containing forbidden metadata/leakage terms are rejected."""

    def _story_with_sentence(self, bad_sentence: str) -> dict:
        return {
            "title": "A Fine Title",
            "sentences": [
                "A perfectly clean opening sentence.",
                bad_sentence,
                "A clean closing sentence.",
            ],
        }

    def _run_with_bad_sentence(self, bad_sentence: str):
        payload = self._story_with_sentence(bad_sentence)
        side_effects = _llm_responses(payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            generate_history_story("Leakage test topic")

    # -- metadata --
    def test_narrator_label_stripped_not_rejected(self):
        """_clean_sentence strips 'Narrator:' labels before validation.
        The generator should succeed and the label must not appear in output."""
        payload = self._story_with_sentence("Narrator: The empire fell in 476 AD.")
        side_effects = _llm_responses(payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects):
            result = generate_history_story("Leakage test topic")
        # The label must have been stripped from the returned sentence
        for s in result["sentences"]:
            self.assertNotIn("Narrator:", s)

    def test_narrator_label_direct_validation_raises(self):
        """validate_story_contract itself rejects narrator labels (no auto-clean)."""
        story = self._story_with_sentence("Narrator: Some narration text here.")
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_sfx_label_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("SFX: Battle drums pound the earth.")

    def test_bracketed_timestamp_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("[00:15] The army advanced.")

    def test_visual_direction_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("Visual: Pan across the battlefield.")

    # -- pipeline leakage --
    def test_whiteboard_reference_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("Draw this on the Whiteboard as narration plays.")

    def test_remotion_reference_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("Use Remotion to animate the timeline here.")

    def test_manifest_reference_rejected(self):
        with self.assertRaises(RuntimeError):
            self._run_with_bad_sentence("See manifest for the full list of scenes.")

    def test_clean_sentences_pass(self):
        """find_sentence_metadata_artifacts returns [] for clean narration."""
        clean = [
            "In 1453 the Ottoman forces breached the walls of Constantinople.",
            "The city's fall marked the end of the Byzantine Empire.",
            "Scholars fled westward, carrying ancient manuscripts to Europe.",
        ]
        for s in clean:
            with self.subTest(sentence=s):
                self.assertEqual(find_sentence_metadata_artifacts(s), [])

    def test_find_sentence_metadata_artifacts_narrator(self):
        artifacts = find_sentence_metadata_artifacts("Narrator: Some text here.")
        self.assertIn("narration label", artifacts)

    def test_find_sentence_metadata_artifacts_sfx(self):
        artifacts = find_sentence_metadata_artifacts("SFX: Sword clash.")
        self.assertIn("sound effect label", artifacts)

    def test_find_sentence_metadata_artifacts_whiteboard(self):
        artifacts = find_sentence_metadata_artifacts("Show this on the Whiteboard.")
        self.assertIn("whiteboard reference", artifacts)


# ════════════════════════════════════════════════════════════════════════════
# 5. Invalid topic
# ════════════════════════════════════════════════════════════════════════════

class TestInvalidTopic(unittest.TestCase):
    """Bad topics are rejected before any LLM call is made."""

    def _assert_raises_before_llm(self, exc_type, *args, **kwargs):
        with patch.object(llm_client, "call_llm") as mock_call:
            with self.assertRaises(exc_type):
                generate_history_story(*args, **kwargs)
            mock_call.assert_not_called()

    def test_empty_string_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, "")

    def test_whitespace_only_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, "   ")

    def test_none_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, None)

    def test_int_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, 42)

    def test_list_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, ["some", "topic"])

    def test_invalid_length_mode_raises_value_error(self):
        self._assert_raises_before_llm(ValueError, "Valid topic", length_mode="medium")

    def test_invalid_length_mode_raises_value_error_2(self):
        self._assert_raises_before_llm(ValueError, "Valid topic", length_mode="")

    def test_invalid_model_key_type_raises_type_error(self):
        self._assert_raises_before_llm(TypeError, "Valid topic", model_key=42)

    def test_valid_topic_does_call_llm(self):
        """Sanity check: a valid topic reaches the LLM."""
        story_payload = _make_valid_story()
        side_effects = _llm_responses(story_payload)
        with patch.object(llm_client, "call_llm", side_effect=side_effects) as mock_call:
            generate_history_story("The Black Death in medieval Europe")
        self.assertEqual(mock_call.call_count, 2)


# ════════════════════════════════════════════════════════════════════════════
# validate_story_contract direct tests
# ════════════════════════════════════════════════════════════════════════════

class TestValidateStoryContract(unittest.TestCase):
    """Unit tests for the validate_story_contract helper itself."""

    def test_valid_story_passes(self):
        validate_story_contract(_make_valid_story())

    def test_not_a_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            validate_story_contract(["not", "a", "dict"])

    def test_extra_key_raises_value_error(self):
        story = _make_valid_story()
        story["scene"] = "extra"
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_topic_key_raises_value_error(self):
        """The old 'topic' key must not be present."""
        story = _make_valid_story()
        story["topic"] = "some topic"
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_missing_title_raises_value_error(self):
        story = {"sentences": ["A sentence."]}
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_missing_sentences_raises_value_error(self):
        story = {"title": "A Title"}
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_empty_title_raises_value_error(self):
        story = _make_valid_story(title="   ")
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_empty_sentences_raises_value_error(self):
        story = _make_valid_story(sentences=[])
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_blank_sentence_raises_value_error(self):
        story = _make_valid_story(sentences=["OK.", "   ", "OK again."])
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_sentence_with_narrator_label_raises(self):
        story = _make_valid_story(sentences=["Narrator: Forbidden label here."])
        with self.assertRaises(ValueError):
            validate_story_contract(story)

    def test_sentence_with_whiteboard_ref_raises(self):
        story = _make_valid_story(sentences=["Display on Whiteboard."])
        with self.assertRaises(ValueError):
            validate_story_contract(story)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
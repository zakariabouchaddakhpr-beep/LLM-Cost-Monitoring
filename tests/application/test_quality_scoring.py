"""The judge call is the only paid step here, so litellm is mocked throughout."""
import json
from unittest.mock import MagicMock, patch

from llm_observability.core.application.quality_scoring import QUERY_TYPES, evaluate


def _judge(payload: dict):
    """Patch litellm.completion to return one JSON judge response."""
    message = MagicMock(content=json.dumps(payload))
    response = MagicMock(choices=[MagicMock(message=message)])
    return patch("litellm.completion", return_value=response)


class TestNoCallWhenNothingWanted:
    def test_both_flags_off_makes_no_api_call(self):
        with patch("litellm.completion") as completion:
            out = evaluate("p", "r", want_score=False, want_query_type=False)
        completion.assert_not_called()
        assert out == {"quality_score": None, "quality_reasoning": None,
                       "query_type": None}


class TestOneCallServesBoth:
    def test_both_fields_from_a_single_call(self):
        """The reason classification is free when scoring is already on."""
        with _judge({"score": 8, "reasoning": "clear",
                     "query_type": "summarization"}) as completion:
            out = evaluate("p", "r", want_score=True, want_query_type=True)
        assert completion.call_count == 1        # not two
        assert out["quality_score"] == 8
        assert out["quality_reasoning"] == "clear"
        assert out["query_type"] == "summarization"

    def test_score_only_leaves_query_type_none(self):
        with _judge({"score": 7, "reasoning": "ok"}):
            out = evaluate("p", "r", want_score=True, want_query_type=False)
        assert out["quality_score"] == 7
        assert out["query_type"] is None

    def test_query_type_only_leaves_score_none(self):
        with _judge({"query_type": "code-generation"}):
            out = evaluate("p", "r", want_score=False, want_query_type=True)
        assert out["query_type"] == "code-generation"
        assert out["quality_score"] is None


class TestVocabulary:
    def test_has_exactly_ten_labels_including_other(self):
        assert len(QUERY_TYPES) == 10
        assert "other" in QUERY_TYPES

    def test_unknown_label_becomes_other(self):
        """Free-text answers would give forty spellings of the same idea and
        make GROUP BY query_type worthless."""
        with _judge({"query_type": "Summarising Things"}):
            out = evaluate("p", "r", want_query_type=True)
        assert out["query_type"] == "other"

    def test_label_is_matched_case_insensitively(self):
        with _judge({"query_type": "Summarization"}):
            out = evaluate("p", "r", want_query_type=True)
        assert out["query_type"] == "summarization"

    def test_every_valid_label_survives(self):
        for label in QUERY_TYPES:
            with _judge({"query_type": label}):
                assert evaluate("p", "r", want_query_type=True)["query_type"] == label


class TestFailureIsSurvivable:
    def test_judge_exception_returns_nones(self):
        with patch("litellm.completion", side_effect=Exception("judge down")):
            out = evaluate("p", "r", want_score=True, want_query_type=True)
        assert out == {"quality_score": None, "quality_reasoning": None,
                       "query_type": None}

    def test_unparseable_json_returns_nones(self):
        message = MagicMock(content="not json at all")
        response = MagicMock(choices=[MagicMock(message=message)])
        with patch("litellm.completion", return_value=response):
            out = evaluate("p", "r", want_score=True)
        assert out["quality_score"] is None

    def test_code_fenced_json_is_still_parsed(self):
        message = MagicMock(content='```json\n{"score": 9, "reasoning": "x"}\n```')
        response = MagicMock(choices=[MagicMock(message=message)])
        with patch("litellm.completion", return_value=response):
            out = evaluate("p", "r", want_score=True)
        assert out["quality_score"] == 9

    def test_empty_response_text_skips_the_call(self):
        with patch("litellm.completion") as completion:
            out = evaluate("p", "", want_score=True)
        completion.assert_not_called()
        assert out["quality_score"] is None

    def test_score_is_clamped_to_the_one_to_ten_range(self):
        with _judge({"score": 47, "reasoning": "x"}):
            assert evaluate("p", "r", want_score=True)["quality_score"] == 10

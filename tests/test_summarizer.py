"""Tests for SummarizationEngine, metadata extraction, and faithfulness scoring."""

import pytest
from tok_n_compress.summarizer import SummarizationEngine


@pytest.fixture
def summarizer():
    return SummarizationEngine()


def test_summarize_segment_heuristic(summarizer):
    """Verify segment summarization without LLM client."""
    transcript = (
        "[USER]: How do I configure SSL in Nginx for domain staging.app.com?\n\n"
        "[ASSISTANT]: You should add listen 443 ssl and ssl_certificate directives in nginx.conf.\n\n"
        "[USER]: I tested it and it works, thanks!"
    )
    result = summarizer.summarize_segment(transcript)

    assert "summary" in result
    assert "files_metadata" in result
    assert "key_prompts" in result
    assert len(result["summary"]) > 0


def test_file_metadata_extraction(summarizer):
    """Verify extraction of files mentioned in conversation."""
    transcript = (
        "We modified config/database.yml and updated migrations/001_init.sql. "
        "Also check out documentation in docs/architecture.md."
    )
    files = summarizer._extract_file_metadata(transcript)
    filenames = [f["filename"] for f in files]

    assert "config/database.yml" in filenames or "database.yml" in filenames
    assert any("init.sql" in f for f in filenames)
    assert any("architecture.md" in f for f in filenames)


def test_key_prompts_extraction(summarizer):
    """Verify extraction of user intents and outcomes."""
    transcript = (
        "[USER]: Fix bug with null pointer exception in user service.\n\n"
        "[ASSISTANT]: Added null check before dereferencing user object.\n\n"
        "[USER]: Also write a regression test.\n\n"
        "[ASSISTANT]: Added test_null_user in test_service.py."
    )
    prompts = summarizer._extract_key_prompts(transcript, limit=3)
    assert len(prompts) >= 1
    assert "Fix bug" in prompts[0]["prompt"] or "null pointer" in prompts[0]["prompt"].lower()


def test_faithfulness_and_hallucination_check(summarizer):
    """Verify mathematical evaluation of summary faithfulness and hallucination penalty."""
    orig = "The payment gateway returned HTTP 504 Gateway Timeout for transaction txn_8831."
    grounded_summary = "Payment gateway failed with HTTP 504 Timeout on transaction txn_8831."
    hallucinated_summary = "Server had CPU overload on port 8080 and lost 500 dollars."

    grounded_metrics = summarizer.evaluate_summary_faithfulness(orig, grounded_summary)
    hallucinated_metrics = summarizer.evaluate_summary_faithfulness(orig, hallucinated_summary)

    assert grounded_metrics["faithfulness"] > hallucinated_metrics["faithfulness"]
    assert grounded_metrics["key_entities_preserved"] is True
    # The hallucinated summary introduced new entities (8080, 500)
    assert hallucinated_metrics["key_entities_preserved"] is False


def test_llm_client_integration():
    """Verify that when an LLM client callable is passed, it is used."""
    mock_llm = lambda prompt: "Mocked LLM summary output for testing."
    custom_engine = SummarizationEngine(llm_client=mock_llm)
    summary = custom_engine._generate_summary("sample transcript", max_length=50)
    assert summary == "Mocked LLM summary output for testing."

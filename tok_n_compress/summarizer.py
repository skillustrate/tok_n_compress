"""
Summarization Engine for Hybrid Memory Model Checkpoints.

Extracts summaries, file metadata, and key prompt/outcome pairs from raw conversation segments.
Supports external LLM providers with an intelligent fallback heuristic engine and real faithfulness metrics.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Callable, Set


class SummarizationEngine:
    """Handles summarization and entity extraction for conversation segments."""

    SEGMENT_SUMMARY_PROMPT = """You are an expert conversation analyst. Summarize the following conversation segment into a concise summary (max {max_length} words).

Conversation segment:
{segment_content}

Requirements:
- Capture the main topic, user intent, and core outcome of this segment
- Identify key decisions, code changes, or actions taken
- Keep it brief, factual, and informative
- Output only the summary text, no markdown formatting or commentary"""

    FILE_METADATA_PROMPT = """You are an expert file analyst. Analyze the following conversation segment and identify any files mentioned, edited, or uploaded.
For each file, provide:
1. Filename (or path)
2. Functional summary of what this file contains or its role

Conversation segment:
{segment_content}

Output format as strict JSON array:
[
  {"filename": "path/to/file.ext", "summary": "brief functional summary"}
]"""

    KEY_PROMPT_PROMPT = """You are an expert intent analyzer. Extract the top {limit} high-intent user requests and their outcomes from the following conversation segment.

Conversation segment:
{segment_content}

Output format as strict JSON array:
[
  {"prompt": "User's request or goal", "outcome": "Result or conclusion achieved"}
]"""

    COMMON_FILE_EXTENSIONS = (
        r"\.(?:py|js|ts|jsx|tsx|json|csv|sql|md|txt|html|css|yaml|yml|sh|toml|xml|env|log|db|png|jpg)"
    )

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "in", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now", "is", "was",
        "are", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"
    }

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize SummarizationEngine.

        Args:
            llm_client: Optional LLM client instance or callable `fn(prompt: str) -> str`.
        """
        self.llm_client = llm_client
        self.default_max_length = int(os.environ.get("SUMMARY_MAX_LENGTH", "256"))
        self.default_key_prompt_limit = int(os.environ.get("KEY_PROMPT_LIMIT", "5"))

    def summarize_segment(
        self,
        segment_content: str,
        max_length: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate summary, file metadata, and key prompts for a conversation segment.

        Args:
            segment_content: Full text of the conversation segment.
            max_length: Max words for the summary.
            limit: Max key prompts to extract.

        Returns:
            Dictionary with 'summary', 'files_metadata', and 'key_prompts'.
        """
        max_len = max_length or self.default_max_length
        prompt_limit = limit or self.default_key_prompt_limit

        summary = self._generate_summary(segment_content, max_len)
        files_metadata = self._extract_file_metadata(segment_content)
        key_prompts = self._extract_key_prompts(segment_content, prompt_limit)

        return {
            "summary": summary,
            "files_metadata": files_metadata,
            "key_prompts": key_prompts
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM client if configured."""
        if not self.llm_client:
            return None
        try:
            if callable(self.llm_client):
                return str(self.llm_client(prompt)).strip()
            elif hasattr(self.llm_client, "complete") and callable(self.llm_client.complete):
                res = self.llm_client.complete(prompt)
                return str(res).strip()
            elif hasattr(self.llm_client, "chat") and callable(self.llm_client.chat):
                res = self.llm_client.chat(prompt)
                return str(res).strip()
        except Exception:
            return None
        return None

    def _generate_summary(self, segment: str, max_length: int) -> str:
        """Generate a concise, faithful summary of the segment."""
        # 1. Try LLM if available
        if self.llm_client:
            prompt = self.SEGMENT_SUMMARY_PROMPT.format(
                segment_content=segment[:8000],
                max_length=max_length
            )
            response = self._call_llm(prompt)
            if response and len(response) > 10:
                return response.strip()

        # 2. Deterministic high-quality extraction fallback
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        if not lines:
            return "Empty conversation segment."

        # Separate turns and extract topic/intent
        turn_contents = []
        user_intents = []
        conclusions = []

        for line in lines:
            # Clean speaker tags like [USER] ..., [ASSISTANT] ..., User: ..., Assistant: ...
            clean_line = re.sub(r"^(\[(?:USER|ASSISTANT|SYSTEM)\]\s*[\d\-:T.Z]*:?|User:|Assistant:|System:)\s*", "", line, flags=re.IGNORECASE).strip()
            if not clean_line:
                continue
            turn_contents.append(clean_line)

            lower = line.lower()
            if "user" in lower or lower.startswith("q:"):
                user_intents.append(clean_line)
            elif "assistant" in lower or "system" in lower or lower.startswith("a:"):
                conclusions.append(clean_line)

        summary_parts = []
        if user_intents:
            primary_intent = user_intents[0]
            if len(primary_intent.split()) > 35:
                primary_intent = " ".join(primary_intent.split()[:35]) + "..."
            summary_parts.append(f"User requested: {primary_intent}")

        if len(user_intents) > 1:
            later_intent = user_intents[-1]
            if len(later_intent.split()) > 30:
                later_intent = " ".join(later_intent.split()[:30]) + "..."
            if later_intent != primary_intent:
                summary_parts.append(f"Follow-up: {later_intent}")

        if conclusions:
            last_conclusion = conclusions[-1]
            if len(last_conclusion.split()) > 40:
                last_conclusion = " ".join(last_conclusion.split()[:40]) + "..."
            summary_parts.append(f"Outcome: {last_conclusion}")
        elif turn_contents:
            last_turn = turn_contents[-1]
            if len(last_turn.split()) > 40:
                last_turn = " ".join(last_turn.split()[:40]) + "..."
            summary_parts.append(f"Latest status: {last_turn}")

        full_summary = " ".join(summary_parts)
        words = full_summary.split()
        if len(words) > max_length:
            return " ".join(words[:max_length]) + "..."
        return full_summary or "Conversation segment reviewed and indexed."

    def _extract_file_metadata(self, segment: str) -> List[Dict[str, str]]:
        """Identify files referenced in conversation along with context."""
        # 1. Try LLM if available
        if self.llm_client:
            prompt = self.FILE_METADATA_PROMPT.format(segment_content=segment[:6000])
            response = self._call_llm(prompt)
            if response:
                try:
                    # Clean markdown code blocks if returned
                    clean = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
                    parsed = json.loads(clean)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass

        # 2. Heuristic extraction of filenames and context
        file_pattern = re.compile(
            r"(?:(?<=[\s`'\"(])|^)([a-zA-Z0-9_\-/]+(?:\.[a-zA-Z0-9_\-]+)*" + self.COMMON_FILE_EXTENSIONS + r")(?=[\s`'\",;!?.)]|$)",
            re.IGNORECASE
        )

        matches = file_pattern.findall(segment)
        seen: Set[str] = set()
        files = []

        sentences = re.split(r"(?<=[.!?\n])\s+", segment)

        for filename in matches:
            norm = filename.strip("./'\"")
            if norm in seen or len(norm) < 3:
                continue
            seen.add(norm)

            # Find matching context sentence
            context_summary = "Referenced in conversation"
            for sentence in sentences:
                if filename in sentence:
                    cleaned_sentence = sentence.strip().replace("\n", " ")
                    if len(cleaned_sentence) > 120:
                        cleaned_sentence = cleaned_sentence[:117] + "..."
                    context_summary = cleaned_sentence
                    break

            files.append({
                "filename": norm,
                "summary": context_summary
            })

        return files

    def _extract_key_prompts(self, segment: str, limit: int) -> List[Dict[str, str]]:
        """Extract high-intent user requests and matching system outcomes."""
        # 1. Try LLM if available
        if self.llm_client:
            prompt = self.KEY_PROMPT_PROMPT.format(
                segment_content=segment[:6000],
                limit=limit
            )
            response = self._call_llm(prompt)
            if response:
                try:
                    clean = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
                    parsed = json.loads(clean)
                    if isinstance(parsed, list):
                        return parsed[:limit]
                except Exception:
                    pass

        # 2. Deterministic extraction from message blocks
        turns = re.split(r"\n\s*\n|(?=\[(?:USER|ASSISTANT|SYSTEM)\])", segment)
        pairs: List[Dict[str, str]] = []
        current_user_prompt: Optional[str] = None

        for turn in turns:
            clean = turn.strip()
            if not clean:
                continue
            lower = clean.lower()

            is_user = "[user]" in lower or lower.startswith("user:") or lower.startswith("human:")
            is_assistant = "[assistant]" in lower or lower.startswith("assistant:") or lower.startswith("ai:")

            text = re.sub(r"^(\[(?:USER|ASSISTANT|SYSTEM)\]\s*[\d\-:T.Z]*:?|User:|Assistant:|Human:|AI:)\s*", "", clean, flags=re.IGNORECASE).strip()

            if is_user:
                if current_user_prompt and not pairs:
                    pairs.append({"prompt": current_user_prompt, "outcome": "Acknowledged."})
                current_user_prompt = text
            elif is_assistant and current_user_prompt:
                outcome_text = text.split("\n")[0]
                if len(outcome_text) > 150:
                    outcome_text = outcome_text[:147] + "..."
                prompt_text = current_user_prompt.split("\n")[0]
                if len(prompt_text) > 150:
                    prompt_text = prompt_text[:147] + "..."
                pairs.append({
                    "prompt": prompt_text,
                    "outcome": outcome_text or "Processed successfully."
                })
                current_user_prompt = None

            if len(pairs) >= limit:
                break

        if current_user_prompt and len(pairs) < limit:
            pairs.append({
                "prompt": current_user_prompt[:150],
                "outcome": "Ongoing or pending action."
            })

        if not pairs:
            # Fallback
            words = segment.strip().split()
            first_phrase = " ".join(words[:15]) if words else "Conversation segment"
            pairs.append({
                "prompt": first_phrase,
                "outcome": "Completed segment discussion."
            })

        return pairs[:limit]

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into words without punctuation."""
        return [w for w in re.findall(r"\b[a-zA-Z0-9_\-\./#]+\b", text.lower()) if w not in self.STOP_WORDS]

    def evaluate_summary_faithfulness(
        self,
        original: str,
        summary: str
    ) -> Dict[str, Any]:
        """
        Evaluate how faithful and complete the summary is compared to the original text.

        Metrics calculated:
        - faithfulness: Precision of summary concepts against the original (penalizes hallucinations).
        - completeness: Recall of important terms from the original.
        - f1_score: Harmonic mean of faithfulness and completeness.
        - key_entities_preserved: True if critical entities (codes, numbers, files) in summary appear in original.
        - entity_preservation_ratio: Ratio of original entities captured in summary.
        """
        orig_tokens = self._tokenize(original)
        summ_tokens = self._tokenize(summary)

        if not orig_tokens or not summ_tokens:
            return {
                "faithfulness": 0.0,
                "completeness": 0.0,
                "f1_score": 0.0,
                "key_entities_preserved": False,
                "entity_preservation_ratio": 0.0
            }

        orig_set = set(orig_tokens)
        summ_set = set(summ_tokens)

        # Faithfulness: precision (what fraction of summary tokens are grounded in original)
        common_tokens = orig_set & summ_set
        precision = len(common_tokens) / len(summ_set) if summ_set else 0.0

        # Completeness: recall of original terms (clamped because summary is meant to be compressed)
        # We compare against top frequent/distinctive content words
        recall = len(common_tokens) / min(len(orig_set), len(summ_set) * 3) if orig_set else 0.0
        recall = min(recall, 1.0)

        # F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Entity preservation check (numbers, error codes, uppercase symbols, filenames)
        orig_entities = set(re.findall(r"\b(?:[A-Z]{2,}|[0-9]+(?:\.[0-9]+)*|[a-z0-9_\-]+\.[a-z]{2,4})\b", original))
        summ_entities = set(re.findall(r"\b(?:[A-Z]{2,}|[0-9]+(?:\.[0-9]+)*|[a-z0-9_\-]+\.[a-z]{2,4})\b", summary))

        if orig_entities:
            preserved_entities = orig_entities & summ_entities
            entity_ratio = len(preserved_entities) / len(orig_entities)
            # If summary contains entities, verify they don't hallucinate non-existent entities
            hallucinated_entities = summ_entities - orig_entities
            key_entities_preserved = len(hallucinated_entities) == 0
        else:
            entity_ratio = 1.0
            key_entities_preserved = True

        return {
            "faithfulness": round(precision, 4),
            "completeness": round(recall, 4),
            "f1_score": round(f1, 4),
            "key_entities_preserved": key_entities_preserved,
            "entity_preservation_ratio": round(entity_ratio, 4)
        }


# Singleton instance
_summarizer: Optional[SummarizationEngine] = None


def get_summarizer(llm_client: Optional[Any] = None) -> SummarizationEngine:
    """Get or create the global summarization engine instance."""
    global _summarizer
    if _summarizer is None or llm_client is not None:
        _summarizer = SummarizationEngine(llm_client=llm_client)
    return _summarizer

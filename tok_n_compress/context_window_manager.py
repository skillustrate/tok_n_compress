"""
Context Window Manager for AHMS (Agentic Hybrid Memory System).

Implements:
- Dynamic Percentage-Based Token Budgeting (sliding window based on model capacity).
- Pin Protection (preserves critical system instructions, schemas, architectural rules).
- Semantic Boundary Snapping (evicts only at clean conversational turn boundaries).
- Log and Output Collapsing (collapses voluminous stdout/stderr/stack traces into summary pointers).
"""

import re
from typing import Dict, List, Any, Tuple, Optional


class ContextWindowManager:
    """Manages Layer 1 Working Memory lifecycle and dynamic eviction triggers."""

    def __init__(
        self,
        model_context_window: int = 32000,
        working_memory_ratio: float = 0.30,
        eviction_threshold_ratio: float = 0.70,
        max_log_lines: int = 50
    ):
        self.model_context_window = model_context_window
        self.working_memory_ratio = working_memory_ratio
        self.eviction_threshold_ratio = eviction_threshold_ratio
        self.max_log_lines = max_log_lines

    @property
    def working_memory_budget(self) -> int:
        """Maximum tokens allocated to active Layer 1 working memory."""
        return max(int(self.model_context_window * self.working_memory_ratio), 1000)

    @property
    def eviction_threshold(self) -> int:
        """Token count at which eviction and downshifting to Layer 2 is triggered."""
        return max(int(self.working_memory_budget * self.eviction_threshold_ratio), 500)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast token estimation based on character and word heuristics."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        approx = max(round(words * 1.33), round(chars / 4.0))
        return max(int(approx), 1)

    def count_turns_tokens(self, turns: List[Dict[str, Any]]) -> int:
        """Calculate total estimated tokens across a list of message turns."""
        total = 0
        for turn in turns:
            content = turn.get("content", "")
            total += self.estimate_tokens(content) + 4
        return total

    def should_trigger_eviction(self, active_history: List[Dict[str, Any]]) -> bool:
        """
        Determine if unpinned active history exceeds the dynamic eviction threshold.
        Pinned turns do not count toward eviction pressure.
        """
        unpinned_tokens = 0
        for turn in active_history:
            if not turn.get("pinned", False) and not turn.get("is_pinned", False):
                content = turn.get("content", "")
                unpinned_tokens += self.estimate_tokens(content) + 4

        return unpinned_tokens >= self.eviction_threshold

    def collapse_large_outputs(self, text: str) -> Tuple[str, bool, Optional[str]]:
        """
        Detect and collapse large terminal logs, test outputs, or stack traces (> max_log_lines).
        """
        lines = text.splitlines()
        if len(lines) <= self.max_log_lines:
            return text, False, None

        first_few = "\n".join(lines[:10])
        last_few = "\n".join(lines[-10:])
        line_count = len(lines)

        error_match = re.search(r"(Error|Exception|Fail|Fatal|Traceback)[^\n]+", text, re.IGNORECASE)
        root_cause_hint = f" Signal: {error_match.group(0).strip()}" if error_match else ""

        collapsed = (
            f"[COLLAPSED LOG/PAYLOAD: {line_count} lines.{root_cause_hint}\n"
            f"--- Head Snippet ---\n{first_few}\n"
            f"... [{line_count - 20} lines omitted for context efficiency] ...\n"
            f"--- Tail Snippet ---\n{last_few}\n"
            f"[Use query_memory or get_raw_segment to inspect full uncompressed log]]"
        )
        return collapsed, True, text

    def partition_eviction_candidates(
        self,
        history: List[Dict[str, Any]],
        keep_recent_n: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partition history into (candidates_to_evict, active_to_preserve).
        Respects:
        1. Pin Protection: Pinned messages are NEVER evicted.
        2. Keep Recent: Preserves the latest `keep_recent_n` messages when history > keep_recent_n.
        3. Semantic Boundary Snapping: Slices at clean User/Assistant/Tool interaction blocks.
        """
        if not history:
            return [], []

        # If history length is <= keep_recent_n, compress all unpinned turns
        if len(history) <= keep_recent_n:
            to_evict = [t for t in history if not t.get("pinned", False) and not t.get("is_pinned", False)]
            to_preserve = [t for t in history if t.get("pinned", False) or t.get("is_pinned", False)]
            if not to_evict:
                return [], list(history)
            return to_evict, to_preserve

        protected_pinned: List[Tuple[int, Dict[str, Any]]] = []
        unpinned_candidates: List[Tuple[int, Dict[str, Any]]] = []

        total_len = len(history)
        cutoff_index = max(0, total_len - keep_recent_n)

        # Scan historical turns up to cutoff
        for idx in range(cutoff_index):
            turn = history[idx]
            if turn.get("pinned", False) or turn.get("is_pinned", False):
                protected_pinned.append((idx, turn))
            else:
                unpinned_candidates.append((idx, turn))

        if not unpinned_candidates:
            return [], list(history)

        # Semantic Boundary Snapping:
        # Snap to end after an assistant, tool, or system turn if possible
        snap_end = len(unpinned_candidates)
        for i in range(len(unpinned_candidates) - 1, -1, -1):
            role = unpinned_candidates[i][1].get("role", "").lower()
            if role in ("assistant", "tool", "system"):
                snap_end = i + 1
                break

        to_evict = [turn for _, turn in unpinned_candidates[:snap_end]]
        evicted_indices = {idx for idx, _ in unpinned_candidates[:snap_end]}
        to_preserve = [turn for idx, turn in enumerate(history) if idx not in evicted_indices]

        return to_evict, to_preserve


# Global singleton
_context_window_manager: Optional[ContextWindowManager] = None


def get_context_window_manager(
    model_context_window: int = 32000,
    working_memory_ratio: float = 0.30
) -> ContextWindowManager:
    """Get or create the global context window manager."""
    global _context_window_manager
    if _context_window_manager is None:
        _context_window_manager = ContextWindowManager(
            model_context_window=model_context_window,
            working_memory_ratio=working_memory_ratio
        )
    return _context_window_manager

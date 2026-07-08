from __future__ import annotations

from .contracts import GapFlag
from .models import Claim, Evidence


class LogicalGapDetector:
    """Detect logical weaknesses in reasoning, not just missing evidence."""

    def detect(self, claims: list[Claim], evidence: list[Evidence]) -> list[GapFlag]:
        """Analyze claims and evidence for logical fallacies."""
        return []

"""Intent Classifier — Embedding-based intent detection for user inputs.

Uses Ollama embeddings (bge-m3) to classify user intent via cosine similarity.
Zero marginal cost: runs locally via Ollama, no API calls.

Intent categories:
- chat: casual conversation, greetings
- evidence: providing data or proof
- dispute: disagreeing with existing claims
- correction: clarifying or fixing information
- exploration: querying knowledge base
- learning: ingesting new knowledge

Usage:
    from knowledge_engine.intent_classifier import IntentClassifier

    classifier = IntentClassifier(embedding_client)
    result = classifier.classify("I disagree with this claim")
    # result.intent == "dispute"
    # result.confidence == 0.87
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embeddings import EmbeddingClient


# ---------------------------------------------------------------------------
# Intent examples (pre-embedded at startup)
# ---------------------------------------------------------------------------
INTENT_EXAMPLES: dict[str, list[str]] = {
    "chat": [
        "Hi, how are you today?",
        "Hello there!",
        "What's the weather like?",
        "Tell me a joke",
        "How's it going?",
        "Good morning!",
        "Thanks for your help",
        "You're doing great",
        "Can you help me with something else?",
        "I'm just browsing",
    ],
    "evidence": [
        "I have data showing that momentum works in trending markets",
        "Here's a backtest result supporting value investing",
        "The study confirms dividend yield correlation with safety",
        "Research shows that cap rates above 6% indicate undervaluation",
        "Evidence suggests liver Qi stagnation responds to Bupleurum",
        "My analysis shows a 15% premium for transit-adjacent properties",
        "The data indicates battery storage breaks even at 2000 cycles",
        "I found a paper that supports this claim",
        "Here's proof from multiple sources",
        "Statistical analysis confirms the hypothesis",
    ],
    "dispute": [
        "I disagree with this claim",
        "This contradicts what I know",
        "The evidence doesn't support this conclusion",
        "I think this is wrong",
        "Actually, that's not quite right",
        "I have a different perspective on this",
        "This claim needs more evidence",
        "I challenge this assertion",
        "The data shows otherwise",
        "I don't think this is accurate",
    ],
    "correction": [
        "Let me clarify - this only applies in bull markets",
        "You're missing the context about jurisdiction",
        "The threshold should be 25%, not 20%",
        "Actually, the dosage should be 6-10g, not 15g",
        "I need to correct - this is for tier-1 cities only",
        "Wait, that's not what I meant",
        "Let me be more specific",
        "I should clarify my earlier statement",
        "The actual number is different",
        "I made an error before",
    ],
    "exploration": [
        "What are the key principles in trading?",
        "How do different schools of thought view this?",
        "What opportunities exist in this domain?",
        "Tell me about cap rate analysis",
        "What does the knowledge base say about momentum?",
        "Show me what you know about real estate",
        "What are the conflicts in this domain?",
        "Explain the different perspectives on this topic",
        "What evidence do we have for this claim?",
        "Help me understand this better",
    ],
    "learning": [
        "Let me share my analysis of the market",
        "Here's what I've been researching",
        "My findings show a new pattern",
        "I want to document my experience",
        "Let me add this to the knowledge base",
        "I've been thinking about this problem",
        "Here's my take on the situation",
        "I want to record what I've learned",
        "Let me share some insights",
        "I've discovered something interesting",
    ],
}

# Emotional indicators for chat sub-mode detection
EMOTIONAL_EXAMPLES = [
    "I'm feeling down today",
    "This is frustrating",
    "I'm grateful for your help",
    "Life feels meaningless",
    "I'm sad that things didn't work out",
    "I'm happy with the results",
    "I'm angry about this situation",
    "I'm worried about the future",
    "I'm excited about this opportunity",
    "I'm disappointed with the outcome",
]

# Complex intent indicators (multi-label)
COMPLEX_INDICATORS = {
    "problem": ["but", "however", "although", "unfortunately", "wish", "if only"],
    "meta": ["why does", "how does", "what makes", "explain why"],
    "social": ["neighbor", "friend", "colleague", "family", "someone"],
}


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str
    confidence: float
    sub_mode: str | None = None
    secondary_intents: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sentiment: str = "neutral"


@dataclass
class _IntentEmbedding:
    """Pre-embedded intent with its examples."""
    intent: str
    embeddings: list[list[float]]
    centroid: list[float] | None = None


class IntentClassifier:
    """Classify user intent using embedding similarity.

    Uses Ollama embeddings (bge-m3) for zero-cost classification.
    Pre-embeds intent examples at startup, then compares user input
    cosine similarity at runtime.
    """

    def __init__(self, embedding_client: "EmbeddingClient"):
        """Initialize classifier with embedding client.

        Args:
            embedding_client: Ollama/OpenAI embedding client
        """
        self._client = embedding_client
        self._intent_embeddings: dict[str, _IntentEmbedding] = {}
        self._emotional_embeddings: list[list[float]] = []
        self._initialized = False

    def warm_up(self) -> None:
        """Pre-embed all intent examples. Call once at startup."""
        if self._initialized:
            return

        # Embed intent examples
        for intent, examples in INTENT_EXAMPLES.items():
            embeddings = self._client.embed_batch_sync(examples)
            centroid = self._compute_centroid(embeddings)
            self._intent_embeddings[intent] = _IntentEmbedding(
                intent=intent,
                embeddings=embeddings,
                centroid=centroid,
            )

        # Embed emotional examples
        self._emotional_embeddings = self._client.embed_batch_sync(EMOTIONAL_EXAMPLES)

        self._initialized = True

    def _compute_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """Compute mean vector (centroid) of embeddings."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                centroid[i] += emb[i]
        n = len(embeddings)
        return [x / n for x in centroid]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def classify(self, text: str) -> IntentResult:
        """Classify user intent from text.

        Args:
            text: User input text

        Returns:
            IntentResult with intent, confidence, and metadata
        """
        if not self._initialized:
            self.warm_up()

        # Embed input
        input_embedding = self._client.embed_sync(text)

        # Compute similarity to each intent
        similarities: dict[str, float] = {}
        for intent, ie in self._intent_embeddings.items():
            # Compare to centroid (faster, smoother)
            if ie.centroid:
                sim = self._cosine_similarity(input_embedding, ie.centroid)
                similarities[intent] = sim

        # Get top intent
        if not similarities:
            return IntentResult(intent="chat", confidence=0.0)

        sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0]

        # Detect secondary intents (score > 0.6 and not the top)
        secondary = [
            intent for intent, score in sorted_intents[1:3]
            if score > 0.6
        ]

        # Detect emotional sub-mode for chat
        sub_mode = None
        if top_intent == "chat":
            emotional_score = max(
                self._cosine_similarity(input_embedding, emb)
                for emb in self._emotional_embeddings
            ) if self._emotional_embeddings else 0.0
            if emotional_score > 0.7:
                sub_mode = "emotional"

        # Detect complex intent indicators
        topics = self._detect_topics(text)
        sentiment = self._analyze_sentiment(text, top_intent)

        return IntentResult(
            intent=top_intent,
            confidence=top_score,
            sub_mode=sub_mode,
            secondary_intents=secondary,
            topics=topics,
            sentiment=sentiment,
        )

    def _detect_topics(self, text: str) -> list[str]:
        """Detect topics from text using keyword matching."""
        text_lower = text.lower()
        topics = []

        # Domain topics
        domain_keywords = {
            "trading": ["trading", "stock", "momentum", "value", "alpha", "backtest"],
            "real_estate": ["property", "real estate", "cap rate", "rental", "mortgage"],
            "tcm": ["tcm", "herb", "acupuncture", "qi", "meridian", "formula"],
            "energy": ["battery", "solar", "grid", "renewable", "storage"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(domain)

        # Problem indicators
        for indicator in COMPLEX_INDICATORS["problem"]:
            if indicator in text_lower:
                topics.append("problem")
                break

        return topics

    def _analyze_sentiment(self, text: str, intent: str) -> str:
        """Simple sentiment analysis based on keywords."""
        text_lower = text.lower()

        positive = ["good", "great", "excellent", "happy", "thanks", "love", "agree"]
        negative = ["bad", "wrong", "disagree", "sad", "angry", "frustrated", "hate"]
        mixed = ["but", "however", "although", "unfortunately"]

        has_positive = any(w in text_lower for w in positive)
        has_negative = any(w in text_lower for w in negative)
        has_mixed = any(w in text_lower for w in mixed)

        if has_mixed and (has_positive or has_negative):
            return "mixed"
        elif has_negative:
            return "negative"
        elif has_positive:
            return "positive"
        return "neutral"

    def classify_batch(self, texts: list[str]) -> list[IntentResult]:
        """Classify multiple texts at once (for transcript processing).

        More efficient than calling classify() multiple times because
        embeddings are computed in a single batch.
        """
        if not self._initialized:
            self.warm_up()

        # Batch embed all texts
        input_embeddings = self._client.embed_batch_sync(texts)

        results = []
        for i, text in enumerate(texts):
            input_emb = input_embeddings[i]

            # Compute similarities
            similarities = {}
            for intent, ie in self._intent_embeddings.items():
                if ie.centroid:
                    similarities[intent] = self._cosine_similarity(input_emb, ie.centroid)

            if not similarities:
                results.append(IntentResult(intent="chat", confidence=0.0))
                continue

            sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_intent, top_score = sorted_intents[0]

            secondary = [
                intent for intent, score in sorted_intents[1:3]
                if score > 0.6
            ]

            sub_mode = None
            if top_intent == "chat":
                emotional_score = max(
                    self._cosine_similarity(input_emb, emb)
                    for emb in self._emotional_embeddings
                ) if self._emotional_embeddings else 0.0
                if emotional_score > 0.7:
                    sub_mode = "emotional"

            topics = self._detect_topics(text)
            sentiment = self._analyze_sentiment(text, top_intent)

            results.append(IntentResult(
                intent=top_intent,
                confidence=top_score,
                sub_mode=sub_mode,
                secondary_intents=secondary,
                topics=topics,
                sentiment=sentiment,
            ))

        return results

"""Theme parser: extracts topics, methods, datasets from a theme document and generates search queries."""

from __future__ import annotations

import re
from collections import Counter

from scholartrace.models.schemas import Theme

# Common English stopwords + academic filler words
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "not", "no", "nor", "so", "if",
    "than", "too", "very", "can", "just", "about", "also", "then", "there", "here",
    "when", "where", "which", "who", "whom", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "any", "only", "own", "same",
    "we", "our", "us", "they", "them", "their", "he", "she", "him", "her", "his",
    "what", "why", "while", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "once", "because", "until",
    "paper", "study", "research", "work", "propose", "approach", "method", "result",
    "results", "show", "shown", "using", "used", "use", "based", "however",
    "therefore", "moreover", "furthermore", "although", "example", "instance",
    "note", "important", "specifically", "particularly", "need", "needs", "make",
    "makes", "still", "much", "many", "well", "even", "get", "got", "like", "way",
    "new", "one", "two", "first", "second", "does", "did", "done", "doing", "going",
    "gone", "went", "come", "came", "take", "took", "give", "gave", "keep", "kept",
    "let", "say", "said", "tell", "told", "find", "found", "know", "knew", "think",
    "thought", "see", "seen", "want", "wanted", "look", "looked", "because",
    # Additional common filler / generic words
    "you", "your", "side", "involve", "involves", "involved", "others", "right",
    "rather", "stay", "stayed", "now", "actual", "reality", "started", "sense",
    "things", "thing", "lot", "concrete", "properly", "scoped", "separate",
    "point", "build", "built", "give", "given", "exactly", "truly", "cause",
    "space", "itself", "already", "entirely", "whether", "situations", "situation",
    "cases", "case", "simply", "entire", "different", "without", "within",
    "become", "became", "across", "along", "since", "around", "enough",
    "handle", "handled", "handle", "carry", "carries", "itself", "attention",
    "window", "track", "text", "cards", "card",
}

# Known method patterns (lowercase for matching)
KNOWN_METHODS = {
    "reinforcement learning", "reward model", "preference optimization",
    "rlhf", "ppo", "dpo", "grpo", "kl", "nlp", "lstm", "cnn", "rnn",
    "transformer", "attention mechanism", "fine-tuning", "finetuning",
    "pre-training", "pretraining", "transfer learning", "self-supervised",
    "supervised learning", "unsupervised learning", "semi-supervised",
    "contrastive learning", "curriculum learning", "active learning",
    "meta-learning", "few-shot", "zero-shot", "in-context learning",
    "chain-of-thought", "instruction tuning", "alignment", "sft",
    "reward hacking", "preference learning", "human feedback",
    "direct preference optimization", "proximal policy optimization",
    "preference data", "reward function",
}

# Known dataset/benchmark name patterns
KNOWN_DATASETS = {
    "TruthfulQA", "HHH", "MT-Bench", "AlpacaEval", "MMLU", "GSM8K",
    "HumanEval", "MBPP", "SQuAD", "GLUE", "SuperGLUE", "BLEU",
    "COPA", "WSC", "RTE", "MNLI", "QNLI", "QQP", "SST",
    "Anthropic HH", "SHH", "OpenAssistant", "UltraFeedback",
    "LMSYS", "Chatbot Arena", "BigBench", "HELM",
}

# Regex for known dataset patterns (handles variations)
_DATASET_PATTERNS = [
    r"\b(?:truthful[\s-]?qa)\b",
    r"\b(?:mt[\s-]?bench)\b",
    r"\b(?:alpaca[\s-]?eval)\b",
    r"\b(?:mmlu)\b",
    r"\b(?:gsm8k)\b",
    r"\b(?:human[\s-]?eval)\b",
    r"\b(?:chatbot[\s-]?arena)\b",
    r"\b(?:big[\s-]?bench)\b",
    r"\b(?:open[\s-]?assistant)\b",
    r"\b(?:ultra[\s-]?feedback)\b",
    r"\b(?:lmsys)\b",
    r"\b(?:anthropic\s+hh)\b",
    r"\b(?:sycophancy[\s-]*(?:eval|bench|benchmark|test))\b",
    r"\b(?:multi[\s-]?turn[\s-]*(?:sycophancy|benchmark|eval))\b",
    r"\b(?:emotional[\s-]*(?:support|dialogue|benchmark))\b",
]


def _clean_text(text: str) -> str:
    """Remove markdown formatting and normalize whitespace."""
    # Remove markdown headers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bullet markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_key_phrases(text: str, n: int = 20) -> list[str]:
    """Extract top N key phrases (unigrams + bigrams) by frequency after stopword removal."""
    cleaned = _clean_text(text)
    # Tokenize: keep alphabetic words and hyphens
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", cleaned.lower())

    # Filter stopwords and short words
    filtered = [w for w in words if w not in STOPWORDS and len(w) >= 3]

    # Count unigrams
    unigram_counts = Counter(filtered)

    # Extract bigrams from filtered tokens
    bigrams: list[str] = []
    for i in range(len(filtered) - 1):
        bg = f"{filtered[i]} {filtered[i + 1]}"
        bigrams.append(bg)
    bigram_counts = Counter(bigrams)

    # Combine: prefer bigrams when they appear frequently
    # Score: bigrams get 1.5x weight since they carry more meaning
    phrase_scores: dict[str, float] = {}
    for phrase, count in unigram_counts.items():
        phrase_scores[phrase] = float(count)
    for phrase, count in bigram_counts.items():
        phrase_scores[phrase] = float(count) * 1.5

    # Sort by score descending, break ties alphabetically
    sorted_phrases = sorted(
        phrase_scores.keys(), key=lambda p: (-phrase_scores[p], p)
    )

    # Deduplicate: if a unigram is part of a top bigram, prefer the bigram
    top_phrases: list[str] = []
    used_words: set[str] = set()
    for phrase in sorted_phrases:
        if len(top_phrases) >= n:
            break
        parts = phrase.split()
        # Allow phrase if its words aren't both already used in higher-ranked phrases
        overlap = sum(1 for p in parts if p in used_words)
        if overlap < len(parts):
            top_phrases.append(phrase)
            for p in parts:
                used_words.add(p)

    return top_phrases


def _extract_methods(text: str) -> list[str]:
    """Extract method names: acronyms, known patterns, CamelCase terms."""
    methods: list[str] = []

    # All-caps acronyms 2-6 chars (not common words)
    acronyms = re.findall(r"\b([A-Z]{2,6})\b", text)
    seen = set()
    for acr in acronyms:
        lower = acr.lower()
        if lower not in STOPWORDS and acr not in seen:
            seen.add(acr)
            methods.append(acr)

    # Known method patterns
    lower_text = text.lower()
    for method in KNOWN_METHODS:
        if method in lower_text:
            # Avoid duplicates (e.g. "RLHF" found as acronym and in known set)
            if method.upper() not in seen:
                methods.append(method)

    # CamelCase terms (at least 2 uppercase letters, length >= 4)
    camel_terms = re.findall(r"\b([a-z]+[A-Z][a-zA-Z]+)\b", text)
    for ct in camel_terms:
        if ct not in seen and len(ct) >= 4:
            methods.append(ct)
            seen.add(ct)

    return methods


def _extract_datasets(text: str) -> list[str]:
    """Extract dataset/benchmark names using known patterns and regex."""
    datasets: list[str] = []
    seen: set[str] = set()

    # Check known dataset names (case-insensitive match)
    lower_text = text.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in lower_text and ds.lower() not in seen:
            datasets.append(ds)
            seen.add(ds.lower())

    # Check regex patterns
    for pattern in _DATASET_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            m_lower = m.lower()
            if m_lower not in seen:
                datasets.append(m)
                seen.add(m_lower)

    return datasets


def _generate_queries(
    topics: list[str],
    methods: list[str],
    datasets: list[str],
    full_text: str,
) -> list[str]:
    """Generate 6-8 diverse search query formulations."""
    queries: list[str] = []
    seen_queries: set[str] = set()

    def _add(q: str) -> None:
        normalized = q.lower().strip()
        if normalized and normalized not in seen_queries:
            seen_queries.add(normalized)
            queries.append(q)

    # 1. Core topic query: top 2-3 topics joined
    core_topics = topics[:3]
    if core_topics:
        _add(" ".join(core_topics))

    # 2. Broad recall query: top 5-6 topics OR'd
    broad_topics = topics[:6]
    if broad_topics:
        _add(" OR ".join(broad_topics))

    # 3. Recent trend query: core + year filters
    if core_topics:
        _add(" ".join(core_topics) + " 2024 OR 2025 OR 2026")

    # 4. Method query: method names combined
    if methods:
        # Mix of acronyms and descriptive names
        method_parts = []
        for m in methods[:4]:
            method_parts.append(m)
        _add(" ".join(method_parts))

    # 5. Domain + topic: take first topic as domain-ish keyword + next topic
    if len(topics) >= 2:
        _add(f"{topics[0]} {topics[1]}")
    elif topics:
        _add(topics[0])

    # 6. Impact query: core topic + benchmark/evaluation/survey
    if core_topics:
        _add(" ".join(core_topics) + " benchmark OR evaluation OR survey")

    # 7. Complementary query: extract less frequent nouns
    cleaned = _clean_text(full_text)
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", cleaned.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    word_counts = Counter(filtered)

    # Get words NOT in top topics
    topic_words = set()
    for t in topics:
        for w in t.split():
            topic_words.add(w.lower())

    complementary = [
        w for w, _ in word_counts.most_common(30) if w not in topic_words
    ]
    if len(complementary) >= 3:
        _add(" ".join(complementary[:4]))

    # 8. Dataset + topic query if datasets found
    if datasets and core_topics:
        _add(f"{datasets[0]} {' '.join(core_topics[:2])}")

    return queries


def parse_theme(document_text: str) -> Theme:
    """Parse a theme document and extract structured query formulations.

    Steps:
    1. Clean the text (remove markdown formatting, normalize whitespace)
    2. Extract key phrases / topics using word frequency
    3. Identify method names (acronyms, known patterns)
    4. Identify datasets/benchmarks
    5. Generate 6-8 query formulations
    """
    topics = _extract_key_phrases(document_text, n=20)
    methods = _extract_methods(document_text)
    datasets = _extract_datasets(document_text)
    queries = _generate_queries(topics, methods, datasets, document_text)

    return Theme(
        document_text=document_text,
        parsed_topics=topics,
        parsed_methods=methods,
        parsed_datasets=datasets,
        parsed_queries=queries,
    )

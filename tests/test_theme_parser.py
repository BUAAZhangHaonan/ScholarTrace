from scholartrace.services.theme_parser import parse_theme


def test_parse_theme_extracts_topics():
    text = "RLHF sycophancy in language models: when reward models amplify agreeable behavior"
    theme = parse_theme(text)
    assert len(theme.parsed_topics) > 0
    assert len(theme.parsed_queries) >= 5


def test_parse_theme_from_research_brief():
    with open("docs/examples/sycophancy_affective_hallucination_research_brief.md") as f:
        text = f.read()
    theme = parse_theme(text)
    assert len(theme.parsed_queries) >= 5
    # Should find sycophancy-related topics
    all_queries = " ".join(theme.parsed_queries).lower()
    assert "sycophancy" in all_queries or "sycophantic" in all_queries


def test_parse_theme_extracts_methods():
    text = "We use RLHF with PPO to train reward models that exhibit sycophantic behavior"
    theme = parse_theme(text)
    assert len(theme.parsed_methods) > 0


def test_queries_are_unique():
    text = "Deep learning for natural language processing with transformers"
    theme = parse_theme(text)
    assert len(theme.parsed_queries) == len(set(theme.parsed_queries))

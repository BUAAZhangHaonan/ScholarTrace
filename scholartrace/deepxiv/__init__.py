"""DeepXiv integration for ScholarTrace.

Provides async access to the DeepXiv API (data.rag.ac.cn) for:
- arXiv paper search (hybrid BM25 + vector)
- Paper metadata with section TLDRs
- Full text extraction
- Semantic Scholar access
- Agent-based paper filtering
"""

from .reader import DeepXivReader
from .token_pool import TokenPool
from .agent import DeepXivAgent

__all__ = ["DeepXivReader", "TokenPool", "DeepXivAgent"]

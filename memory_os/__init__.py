"""
memory-os: Brain-like memory for AI agents.
Grounded in Ebbinghaus forgetting curves, emotional tagging,
and memory consolidation research.
"""

from .store       import MemoryStore
from .decay       import DecayReranker
from .emotion     import EmotionTagger
from .extractor   import MemoryExtractor
from .compression import MemoryConsolidator
from .router      import MemoryRouter
from .agent       import MemoryAgent

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "DecayReranker",
    "EmotionTagger",
    "MemoryExtractor",
    "MemoryConsolidator",
    "MemoryRouter",
    "MemoryAgent",
]

"""AITF Instrumentation modules for AI system telemetry.

Provides auto-instrumentation for LLM inference, agent execution,
MCP protocol, RAG pipelines, and skill invocations.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from aitf.instrumentation.agent import AgentInstrumentor
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.instrumentation.mcp import MCPInstrumentor
from aitf.instrumentation.rag import RAGInstrumentor
from aitf.instrumentation.skills import SkillInstrumentor


class AITFInstrumentor:
    """Main AITF instrumentor that manages all sub-instrumentors.

    Usage:
        instrumentor = AITFInstrumentor()
        instrumentor.instrument_all()

        # Or selectively:
        instrumentor.instrument(llm=True, agent=True, mcp=True)
    """

    def __init__(self, tracer_provider: TracerProvider | None = None):
        self._tracer_provider = tracer_provider
        self._llm = LLMInstrumentor(tracer_provider=tracer_provider)
        self._agent = AgentInstrumentor(tracer_provider=tracer_provider)
        self._mcp = MCPInstrumentor(tracer_provider=tracer_provider)
        self._rag = RAGInstrumentor(tracer_provider=tracer_provider)
        self._skills = SkillInstrumentor(tracer_provider=tracer_provider)
        self._instrumented = False

    def instrument_all(self) -> None:
        """Instrument all supported AI components."""
        self._llm.instrument()
        self._agent.instrument()
        self._mcp.instrument()
        self._rag.instrument()
        self._skills.instrument()
        self._instrumented = True

    def instrument(
        self,
        llm: bool = False,
        agent: bool = False,
        mcp: bool = False,
        rag: bool = False,
        skills: bool = False,
    ) -> None:
        """Selectively instrument AI components."""
        if llm:
            self._llm.instrument()
        if agent:
            self._agent.instrument()
        if mcp:
            self._mcp.instrument()
        if rag:
            self._rag.instrument()
        if skills:
            self._skills.instrument()
        self._instrumented = True

    def uninstrument_all(self) -> None:
        """Remove all instrumentation."""
        self._llm.uninstrument()
        self._agent.uninstrument()
        self._mcp.uninstrument()
        self._rag.uninstrument()
        self._skills.uninstrument()
        self._instrumented = False

    @property
    def is_instrumented(self) -> bool:
        return self._instrumented

    @property
    def llm(self) -> LLMInstrumentor:
        return self._llm

    @property
    def agent(self) -> AgentInstrumentor:
        return self._agent

    @property
    def mcp(self) -> MCPInstrumentor:
        return self._mcp

    @property
    def rag(self) -> RAGInstrumentor:
        return self._rag

    @property
    def skills(self) -> SkillInstrumentor:
        return self._skills


__all__ = [
    "AITFInstrumentor",
    "LLMInstrumentor",
    "AgentInstrumentor",
    "MCPInstrumentor",
    "RAGInstrumentor",
    "SkillInstrumentor",
]

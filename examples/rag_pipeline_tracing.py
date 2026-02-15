"""AITF Example: RAG Pipeline Tracing.

Demonstrates how to trace a complete RAG (Retrieval-Augmented Generation)
pipeline including vector search, reranking, and quality evaluation.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.rag import RAGInstrumentor
from aitf.instrumentation.llm import LLMInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_rag_events.jsonl",
)))
trace.set_tracer_provider(provider)

rag = RAGInstrumentor(tracer_provider=provider)
llm = LLMInstrumentor(tracer_provider=provider)

# --- Full RAG Pipeline ---

print("=== RAG Pipeline: Question Answering ===\n")

with rag.trace_pipeline(
    pipeline_name="knowledge-base-qa",
    query="What are the key components of the AITF framework?",
) as pipeline:

    # Step 1: Generate query embedding
    with llm.trace_inference(
        model="text-embedding-3-small",
        system="openai",
        operation="embeddings",
    ) as embed_span:
        embed_span.set_usage(input_tokens=12)
        embed_span.set_cost(input_cost=0.00000024)

    # Step 2: Vector retrieval
    with pipeline.retrieve(
        database="pinecone",
        top_k=10,
    ) as retrieval:
        retrieval.set_results(
            count=8,
            min_score=0.72,
            max_score=0.95,
        )

    # Step 3: Reranking
    with pipeline.rerank(model="cross-encoder/ms-marco-MiniLM-L-12-v2") as rerank:
        rerank.set_results(input_count=8, output_count=5)

    # Step 4: LLM Generation with context
    with llm.trace_inference(
        model="gpt-4o",
        system="openai",
        operation="chat",
        temperature=0.3,
    ) as gen_span:
        gen_span.set_prompt(
            "Based on the following context, answer the question...\n"
            "Context: [5 retrieved documents]\n"
            "Question: What are the key components of the AITF framework?"
        )
        gen_span.set_completion(
            "The AITF framework has four key layers: "
            "1) Instrumentation (LLM, Agent, MCP, RAG, Skills), "
            "2) Collection (OTel Collector + AITF processors), "
            "3) Normalization (OCSF mapper + compliance mapper), "
            "4) Analytics (SIEM, XDR, dashboards)."
        )
        gen_span.set_usage(input_tokens=800, output_tokens=120)
        gen_span.set_cost(input_cost=0.002, output_cost=0.0012)

    # Step 5: Quality evaluation
    pipeline.set_quality(
        context_relevance=0.88,
        answer_relevance=0.92,
        faithfulness=0.95,
        groundedness=0.90,
    )

print("\nRAG pipeline tracing complete. Events at /tmp/aitf_rag_events.jsonl")

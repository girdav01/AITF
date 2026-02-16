"""AITF Example: Model Operations (LLMOps/MLOps) Tracing.

Demonstrates how to trace the full AI model lifecycle with AITF,
including training, evaluation, registry, deployment, serving
(routing, fallback, caching), monitoring, and prompt versioning.

Usage:
    pip install opentelemetry-sdk aitf
    python model_ops_tracing.py
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from aitf.instrumentation.model_ops import ModelOpsInstrumentor
from aitf.exporters.ocsf_exporter import OCSFExporter

# --- Setup ---

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(SimpleSpanProcessor(OCSFExporter(
    output_file="/tmp/aitf_model_ops_events.jsonl",
    compliance_frameworks=["nist_ai_rmf", "eu_ai_act"],
)))
trace.set_tracer_provider(provider)

model_ops = ModelOpsInstrumentor(tracer_provider=provider)
model_ops.instrument()

# --- Example 1: Model Training ---

print("=== Example 1: Model Training (LoRA Fine-Tuning) ===\n")

with model_ops.trace_training(
    training_type="lora",
    base_model="meta-llama/Llama-3.1-70B",
    framework="pytorch",
    dataset_id="customer-support-v3",
    dataset_version="3.2.1",
    dataset_size=85000,
    hyperparameters='{"learning_rate": 2e-5, "lora_rank": 16, "lora_alpha": 32}',
    epochs=3,
    experiment_id="exp-cs-llama-001",
    experiment_name="Customer Support Llama Fine-Tune",
) as run:
    # Simulate training progress
    run.set_compute(gpu_type="A100-80GB", gpu_count=4, gpu_hours=12.5)
    run.set_code_commit("a1b2c3d4e5f6")
    run.set_loss(loss=0.42, val_loss=0.48)
    run.set_output_model(
        model_id="cs-llama-70b-lora-v3",
        model_hash="sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    )

print(f"  Training run: {run.run_id}")
print("  Output model: cs-llama-70b-lora-v3\n")

# --- Example 2: Model Evaluation ---

print("=== Example 2: Model Evaluation ===\n")

# Benchmark evaluation
with model_ops.trace_evaluation(
    model_id="cs-llama-70b-lora-v3",
    eval_type="benchmark",
    dataset_id="cs-eval-suite-v2",
    dataset_size=5000,
    baseline_model="meta-llama/Llama-3.1-70B",
) as eval_run:
    eval_run.set_metrics({
        "accuracy": 0.94,
        "f1": 0.91,
        "bleu": 0.78,
        "toxicity_score": 0.02,
    })
    eval_run.set_pass(passed=True, regression_detected=False)

print(f"  Benchmark eval: {eval_run.run_id} - PASSED")

# LLM-as-judge evaluation
with model_ops.trace_evaluation(
    model_id="cs-llama-70b-lora-v3",
    eval_type="llm_judge",
    dataset_id="cs-judge-prompts-v1",
    dataset_size=500,
    judge_model="gpt-4o",
) as eval_run:
    eval_run.set_metrics({
        "helpfulness": 4.2,
        "accuracy": 4.5,
        "safety": 4.8,
        "overall": 4.5,
    })
    eval_run.set_pass(passed=True)

print(f"  LLM judge eval: {eval_run.run_id} - PASSED\n")

# --- Example 3: Model Registry ---

print("=== Example 3: Model Registry ===\n")

with model_ops.trace_registry(
    model_id="cs-llama-70b-lora-v3",
    operation="register",
    model_version="3.0.0",
    stage="staging",
    model_alias="customer-support-latest",
    owner="ml-platform-team",
    training_run_id=run.run_id,
    parent_model_id="meta-llama/Llama-3.1-70B",
) as span:
    span.add_event("model.registered", attributes={
        "registry": "internal-model-registry",
        "tags": "customer-support,lora,llama-3.1",
    })

print("  Registered cs-llama-70b-lora-v3 v3.0.0 (staging)")

# Promote to production
with model_ops.trace_registry(
    model_id="cs-llama-70b-lora-v3",
    operation="promote",
    model_version="3.0.0",
    stage="production",
) as span:
    span.add_event("model.promoted", attributes={
        "from_stage": "staging",
        "to_stage": "production",
    })

print("  Promoted to production\n")

# --- Example 4: Model Deployment ---

print("=== Example 4: Model Deployment (Canary) ===\n")

with model_ops.trace_deployment(
    model_id="cs-llama-70b-lora-v3",
    strategy="canary",
    environment="production",
    endpoint="https://models.internal.example.com/cs-llama",
    canary_percent=10.0,
    infrastructure_provider="aws",
) as deployment:
    deployment.set_infrastructure(gpu_type="A100-80GB", replicas=4)
    deployment.set_health(status="healthy", latency_ms=45.2)

print(f"  Deployment ID: {deployment.deployment_id}")
print("  Strategy: canary (10%)")
print("  Health: healthy (45.2ms)\n")

# --- Example 5: Model Serving - Routing ---

print("=== Example 5: Model Serving - Routing ===\n")

with model_ops.trace_route(
    selected_model="cs-llama-70b-lora-v3",
    reason="capability",
    candidates=["cs-llama-70b-lora-v3", "gpt-4o", "claude-sonnet-4-5-20250929"],
) as span:
    span.add_event("route.selected", attributes={
        "selection_criteria": "domain-specific customer support query",
    })

print("  Routed to cs-llama-70b-lora-v3 (capability match)\n")

# --- Example 6: Model Serving - Fallback ---

print("=== Example 6: Model Serving - Fallback ===\n")

with model_ops.trace_fallback(
    original_model="cs-llama-70b-lora-v3",
    final_model="gpt-4o",
    trigger="timeout",
    chain=["cs-llama-70b-lora-v3", "gpt-4o"],
    depth=1,
) as span:
    span.add_event("fallback.triggered", attributes={
        "original_error": "Model inference timed out after 30s",
    })

print("  Fallback: cs-llama-70b-lora-v3 -> gpt-4o (timeout)\n")

# --- Example 7: Model Serving - Cache Lookup ---

print("=== Example 7: Model Serving - Cache Lookup ===\n")

# Cache hit
with model_ops.trace_cache_lookup(cache_type="semantic") as lookup:
    lookup.set_hit(hit=True, similarity_score=0.97, cost_saved_usd=0.003)

print("  Semantic cache HIT (similarity: 0.97, saved: $0.003)")

# Cache miss
with model_ops.trace_cache_lookup(cache_type="semantic") as lookup:
    lookup.set_hit(hit=False, similarity_score=0.45)

print("  Semantic cache MISS (similarity: 0.45)\n")

# --- Example 8: Model Monitoring ---

print("=== Example 8: Model Monitoring ===\n")

# Performance check
with model_ops.trace_monitoring_check(
    model_id="cs-llama-70b-lora-v3",
    check_type="performance",
    metric_name="p95_latency_ms",
) as check:
    check.set_result(
        result="pass",
        metric_value=180.5,
        baseline_value=200.0,
    )

print("  Performance check: PASS (p95 latency 180.5ms < 200ms baseline)")

# Drift detection
with model_ops.trace_monitoring_check(
    model_id="cs-llama-70b-lora-v3",
    check_type="drift",
    metric_name="input_distribution",
) as check:
    check.set_result(
        result="warning",
        drift_score=0.35,
        drift_type="feature",
        baseline_value=0.10,
        metric_value=0.35,
        action_triggered="alert",
    )

print("  Drift check: WARNING (drift score 0.35, alert triggered)")

# SLA compliance check
with model_ops.trace_monitoring_check(
    model_id="cs-llama-70b-lora-v3",
    check_type="sla",
    metric_name="availability_pct",
) as check:
    check.set_result(
        result="pass",
        metric_value=99.95,
        baseline_value=99.9,
    )

print("  SLA check: PASS (availability 99.95% >= 99.9% target)\n")

# --- Example 9: Prompt Lifecycle ---

print("=== Example 9: Prompt Lifecycle ===\n")

# Register a new prompt version
with model_ops.trace_prompt(
    name="customer-support-system",
    operation="register",
    version="2.1.0",
    label="production",
    model_target="cs-llama-70b-lora-v3",
) as prompt_op:
    prompt_op.set_content_hash("sha256:e3b0c44298fc1c149afbf4c8996fb924")

print("  Registered prompt 'customer-support-system' v2.1.0")

# Evaluate the prompt
with model_ops.trace_prompt(
    name="customer-support-system",
    operation="evaluate",
    version="2.1.0",
    model_target="cs-llama-70b-lora-v3",
) as prompt_op:
    prompt_op.set_evaluation(score=0.92, passed=True)

print("  Prompt evaluation: score=0.92, PASSED")

# A/B test
with model_ops.trace_prompt(
    name="customer-support-system",
    operation="ab_test",
    version="2.1.0",
    model_target="cs-llama-70b-lora-v3",
) as prompt_op:
    prompt_op.set_ab_test(
        test_id="ab-prompt-cs-001",
        variant="B",
        traffic_pct=20.0,
    )

print("  A/B test: variant B at 20% traffic\n")

# --- Summary ---

print("Model ops tracing complete. Events at /tmp/aitf_model_ops_events.jsonl")

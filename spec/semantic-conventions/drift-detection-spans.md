# Model Drift Detection Span Conventions

AITF defines structured semantic conventions for model drift detection — covering data distribution drift, concept drift, performance degradation, and calibration drift. These conventions align with CoSAI's identification of model drift as a top-level AI incident category requiring dedicated telemetry.

## Overview

The `aitf.drift.*` namespace provides structured drift telemetry distinct from the basic monitoring checks in `aitf.model_ops.monitoring`. While `model_ops.monitoring` captures routine health checks, `aitf.drift.*` provides deep, forensic-quality drift analysis with segment-level granularity, reference dataset comparisons, and statistical test results.

| Stage | Span Name | Description |
|-------|-----------|-------------|
| Detection | `aitf.drift.detect` | Run drift detection analysis |
| Baseline | `aitf.drift.baseline` | Establish or update drift baseline |
| Investigation | `aitf.drift.investigate` | Deep-dive investigation of detected drift |
| Remediation | `aitf.drift.remediate` | Remediation action (retrain, rollback, etc.) |

---

## Span: `aitf.drift.detect`

Represents a drift detection analysis — comparing current model behavior against an established baseline.

### Span Name

Format: `drift.detect {aitf.drift.type} {aitf.drift.model_id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.model_id` | string | Model being monitored |
| `aitf.drift.type` | string | Drift type (see below) |
| `aitf.drift.score` | double | Drift magnitude (0.0–1.0) |
| `aitf.drift.result` | string | `"normal"`, `"warning"`, `"alert"`, `"critical"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.detection_method` | string | Statistical method used (see below) |
| `aitf.drift.baseline_metric` | double | Baseline metric value |
| `aitf.drift.current_metric` | double | Current metric value |
| `aitf.drift.metric_name` | string | Name of metric being tracked |
| `aitf.drift.threshold` | double | Alert threshold |
| `aitf.drift.p_value` | double | Statistical significance (p-value) |
| `aitf.drift.reference_dataset` | string | Reference dataset identifier |
| `aitf.drift.reference_period` | string | Reference time period (`"2026-01-01/2026-01-31"`) |
| `aitf.drift.evaluation_window` | string | Current evaluation window |
| `aitf.drift.sample_size` | int | Number of samples in evaluation |
| `aitf.drift.affected_segments` | string[] | Impacted user/data segments |
| `aitf.drift.feature_name` | string | Specific feature exhibiting drift (for feature drift) |
| `aitf.drift.feature_importance` | double | Importance of drifted feature (0–1) |
| `aitf.drift.action_triggered` | string | Automated action (`"none"`, `"alert"`, `"retrain"`, `"rollback"`, `"quarantine"`) |

### Drift Types

| Value | Description |
|-------|-------------|
| `data_distribution` | Input data distribution shift (covariate shift) |
| `concept` | Concept drift — relationship between inputs and outputs changes |
| `performance` | Model performance degradation (accuracy, latency, etc.) |
| `calibration` | Prediction confidence calibration drift |
| `embedding` | Embedding space drift |
| `feature` | Individual feature distribution shift |
| `prediction` | Prediction distribution shift (prior probability shift) |
| `label` | Label distribution shift in supervised feedback |

### Detection Methods

| Value | Description |
|-------|-------------|
| `psi` | Population Stability Index |
| `ks_test` | Kolmogorov-Smirnov test |
| `chi_squared` | Chi-squared test |
| `js_divergence` | Jensen-Shannon divergence |
| `kl_divergence` | Kullback-Leibler divergence |
| `wasserstein` | Wasserstein distance (Earth Mover's Distance) |
| `mmd` | Maximum Mean Discrepancy |
| `adwin` | ADWIN adaptive windowing |
| `ddm` | Drift Detection Method |
| `page_hinkley` | Page-Hinkley test |
| `custom` | Custom detection method |

---

## Span: `aitf.drift.baseline`

Represents baseline establishment or refresh — capturing the reference distribution or performance profile against which drift is measured.

### Span Name

Format: `drift.baseline {aitf.drift.baseline.operation} {aitf.drift.model_id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.model_id` | string | Model for baseline |
| `aitf.drift.baseline.operation` | string | `"create"`, `"refresh"`, `"validate"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.baseline.id` | string | Baseline identifier |
| `aitf.drift.baseline.dataset` | string | Dataset used for baseline |
| `aitf.drift.baseline.sample_size` | int | Number of samples |
| `aitf.drift.baseline.period` | string | Time period covered |
| `aitf.drift.baseline.metrics` | string | JSON of baseline metric values |
| `aitf.drift.baseline.features` | string[] | Features tracked |
| `aitf.drift.baseline.previous_id` | string | Previous baseline ID (for refresh) |

---

## Span: `aitf.drift.investigate`

Represents a deep-dive investigation triggered by drift detection — analyzing root causes, affected segments, and impact scope.

### Span Name

Format: `drift.investigate {aitf.drift.type} {aitf.drift.model_id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.model_id` | string | Model under investigation |
| `aitf.drift.investigation.trigger_id` | string | Detection span ID that triggered investigation |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.investigation.root_cause` | string | Identified root cause |
| `aitf.drift.investigation.root_cause_category` | string | `"data_quality"`, `"upstream_change"`, `"seasonal"`, `"adversarial"`, `"model_degradation"`, `"unknown"` |
| `aitf.drift.investigation.affected_segments` | string[] | Impacted segments with details |
| `aitf.drift.investigation.affected_users_estimate` | int | Estimated affected users |
| `aitf.drift.investigation.blast_radius` | string | `"isolated"`, `"segment"`, `"widespread"`, `"global"` |
| `aitf.drift.investigation.severity` | string | `"low"`, `"medium"`, `"high"`, `"critical"` |
| `aitf.drift.investigation.recommendation` | string | Recommended remediation |

---

## Span: `aitf.drift.remediate`

Represents a remediation action taken in response to detected drift.

### Span Name

Format: `drift.remediate {aitf.drift.remediation.action} {aitf.drift.model_id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.model_id` | string | Model being remediated |
| `aitf.drift.remediation.action` | string | Action type (see below) |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.drift.remediation.trigger_id` | string | Detection/investigation span that triggered this |
| `aitf.drift.remediation.automated` | boolean | Whether automatically triggered |
| `aitf.drift.remediation.initiated_by` | string | Identity that initiated remediation |
| `aitf.drift.remediation.status` | string | `"pending"`, `"in_progress"`, `"completed"`, `"failed"` |
| `aitf.drift.remediation.rollback_to` | string | Model version rolled back to |
| `aitf.drift.remediation.retrain_dataset` | string | New training dataset |
| `aitf.drift.remediation.validation_passed` | boolean | Whether post-remediation validation passed |

### Remediation Actions

| Value | Description |
|-------|-------------|
| `retrain` | Retrain model on updated data |
| `rollback` | Roll back to previous model version |
| `recalibrate` | Recalibrate model predictions |
| `feature_gate` | Gate drifted features |
| `traffic_shift` | Shift traffic to stable model |
| `alert_only` | Alert without automated action |
| `quarantine` | Quarantine model from serving |

---

## Examples

### Example 1: Data distribution drift detected

```
Span: drift.detect data_distribution customer-support-llama-70b
  aitf.drift.model_id: "customer-support-llama-70b"
  aitf.drift.type: "data_distribution"
  aitf.drift.score: 0.73
  aitf.drift.result: "alert"
  aitf.drift.detection_method: "psi"
  aitf.drift.baseline_metric: 0.12
  aitf.drift.current_metric: 0.73
  aitf.drift.metric_name: "psi_score"
  aitf.drift.threshold: 0.25
  aitf.drift.reference_dataset: "prod-baseline-jan-2026"
  aitf.drift.reference_period: "2026-01-01/2026-01-31"
  aitf.drift.evaluation_window: "2026-02-10/2026-02-16"
  aitf.drift.sample_size: 50000
  aitf.drift.affected_segments: ["enterprise_customers", "apac_region"]
  aitf.drift.action_triggered: "alert"
```

### Example 2: Concept drift with auto-retrain

```
Span: drift.detect concept fraud-detection-v2
  aitf.drift.model_id: "fraud-detection-v2"
  aitf.drift.type: "concept"
  aitf.drift.score: 0.85
  aitf.drift.result: "critical"
  aitf.drift.detection_method: "adwin"
  aitf.drift.baseline_metric: 0.94
  aitf.drift.current_metric: 0.78
  aitf.drift.metric_name: "precision"
  aitf.drift.p_value: 0.001
  aitf.drift.affected_segments: ["crypto_transactions", "new_account_holders"]
  aitf.drift.action_triggered: "retrain"

  └─ Span: drift.remediate retrain fraud-detection-v2
       aitf.drift.model_id: "fraud-detection-v2"
       aitf.drift.remediation.action: "retrain"
       aitf.drift.remediation.automated: true
       aitf.drift.remediation.retrain_dataset: "fraud-data-feb-2026"
       aitf.drift.remediation.status: "completed"
       aitf.drift.remediation.validation_passed: true
```

### Example 3: Feature-level drift investigation

```
Span: drift.investigate feature customer-churn-model
  aitf.drift.model_id: "customer-churn-model"
  aitf.drift.investigation.trigger_id: "span-drift-det-001"
  aitf.drift.investigation.root_cause: "Upstream CRM data pipeline changed field encoding"
  aitf.drift.investigation.root_cause_category: "upstream_change"
  aitf.drift.investigation.affected_segments: ["premium_tier", "monthly_subscribers"]
  aitf.drift.investigation.affected_users_estimate: 12500
  aitf.drift.investigation.blast_radius: "segment"
  aitf.drift.investigation.severity: "high"
  aitf.drift.investigation.recommendation: "Rollback model and coordinate with data engineering on field encoding fix"
```

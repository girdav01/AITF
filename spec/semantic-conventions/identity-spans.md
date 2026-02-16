# Agentic Identity Span Conventions

AITF defines comprehensive semantic conventions for AI agent identity management — covering the full lifecycle from identity creation through authentication, authorization, delegation, and revocation. These conventions enable secure, auditable, and observable identity operations in single-agent and multi-agent systems.

## Overview

The `aitf.identity.*` namespace covers the complete agent identity lifecycle:

| Stage | Span Name | Description |
|-------|-----------|-------------|
| Lifecycle | `aitf.identity.lifecycle` | Identity creation, rotation, suspension, revocation |
| Authentication | `aitf.identity.authentication` | Agent authentication (OAuth, mTLS, SPIFFE, JWT) |
| Authorization | `aitf.identity.authorization` | Permission checks, policy evaluation |
| Delegation | `aitf.identity.delegation` | Credential delegation, token exchange, OBO flows |
| Trust | `aitf.identity.trust` | Agent-to-agent trust establishment, VC verification |
| Session | `aitf.identity.session` | Identity session management |

---

## Span: `aitf.identity.lifecycle`

Represents an agent identity lifecycle event (creation, rotation, revocation).

### Span Name

Format: `identity.lifecycle.{aitf.identity.lifecycle.operation} {aitf.identity.agent_id}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_id` | string | Agent identity identifier |
| `aitf.identity.agent_name` | string | Agent name |
| `aitf.identity.lifecycle.operation` | string | Lifecycle operation (see below) |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.type` | string | Identity type (see below) |
| `aitf.identity.provider` | string | Identity provider (`"okta"`, `"entra_id"`, `"auth0"`, `"spiffe"`, `"custom"`) |
| `aitf.identity.owner` | string | Human or system owner of this identity |
| `aitf.identity.owner_type` | string | `"human"`, `"service"`, `"organization"` |
| `aitf.identity.credential_type` | string | `"api_key"`, `"oauth_token"`, `"jwt"`, `"mtls_cert"`, `"spiffe_svid"`, `"did"` |
| `aitf.identity.credential_id` | string | Credential identifier (not the secret) |
| `aitf.identity.expires_at` | string | Credential expiration ISO timestamp |
| `aitf.identity.ttl_seconds` | int | Time to live in seconds |
| `aitf.identity.auto_rotate` | boolean | Whether credential auto-rotates |
| `aitf.identity.rotation_interval_seconds` | int | Rotation interval |
| `aitf.identity.scope` | string[] | Granted scopes |
| `aitf.identity.tags` | string | JSON-encoded identity metadata tags |
| `aitf.identity.status` | string | `"active"`, `"suspended"`, `"revoked"`, `"expired"` |
| `aitf.identity.previous_status` | string | Previous status (for transitions) |

### Lifecycle Operations

| Value | Description |
|-------|-------------|
| `create` | Create new agent identity |
| `register` | Register identity with an IdP or service |
| `activate` | Activate a pending identity |
| `rotate` | Rotate credentials (key, cert, token) |
| `suspend` | Temporarily suspend an identity |
| `reactivate` | Reactivate a suspended identity |
| `revoke` | Permanently revoke an identity |
| `expire` | Identity expired automatically |
| `update` | Update identity metadata or scope |

### Identity Types

| Value | Description |
|-------|-------------|
| `persistent` | Long-lived agent identity with stable credentials |
| `ephemeral` | Short-lived identity created for a single task |
| `delegated` | Identity derived from delegation of another identity |
| `federated` | Identity from a federated trust domain |
| `workload` | SPIFFE-style workload identity |

---

## Span: `aitf.identity.authentication`

Represents an agent authentication attempt.

### Span Name

Format: `identity.auth {aitf.identity.agent_name}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_id` | string | Agent identity ID |
| `aitf.identity.agent_name` | string | Agent name |
| `aitf.identity.auth.method` | string | Authentication method (see below) |
| `aitf.identity.auth.result` | string | `"success"`, `"failure"`, `"denied"`, `"expired"`, `"revoked"` |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.auth.provider` | string | Auth provider service |
| `aitf.identity.auth.target_service` | string | Service being authenticated to |
| `aitf.identity.auth.failure_reason` | string | Reason for failure (if applicable) |
| `aitf.identity.auth.token_type` | string | Token type received (`"bearer"`, `"dpop"`, `"mtls_bound"`) |
| `aitf.identity.auth.token_lifetime_seconds` | int | Token lifetime |
| `aitf.identity.auth.scope_requested` | string[] | Scopes requested |
| `aitf.identity.auth.scope_granted` | string[] | Scopes actually granted |
| `aitf.identity.auth.mfa_used` | boolean | Whether MFA was used |
| `aitf.identity.auth.continuous` | boolean | Whether this is continuous re-authentication |
| `aitf.identity.auth.protocol_version` | string | Auth protocol version |
| `aitf.identity.auth.client_id` | string | OAuth client ID |
| `aitf.identity.auth.pkce_used` | boolean | Whether PKCE was used |
| `aitf.identity.auth.dpop_used` | boolean | Whether DPoP proof was included |

### Authentication Methods

| Value | Description |
|-------|-------------|
| `api_key` | Static API key authentication |
| `oauth2` | OAuth 2.0/2.1 flow |
| `oauth2_pkce` | OAuth 2.1 with PKCE (MCP standard) |
| `jwt_bearer` | JWT Bearer token (RFC 7523) |
| `mtls` | Mutual TLS certificate authentication |
| `spiffe_svid` | SPIFFE Verifiable Identity Document |
| `did_vc` | Decentralized Identifier + Verifiable Credential |
| `http_signature` | HTTP Message Signature (W3C) |
| `token_exchange` | OAuth Token Exchange (RFC 8693) |
| `saml` | SAML assertion |

---

## Span: `aitf.identity.authorization`

Represents a permission check or policy evaluation.

### Span Name

Format: `identity.authz {aitf.identity.agent_name} -> {aitf.identity.authz.resource}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_id` | string | Agent identity ID |
| `aitf.identity.agent_name` | string | Agent name |
| `aitf.identity.authz.decision` | string | `"allow"`, `"deny"`, `"conditional"` |
| `aitf.identity.authz.resource` | string | Resource being accessed |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.authz.action` | string | Action being performed (`"read"`, `"write"`, `"execute"`, `"delete"`) |
| `aitf.identity.authz.policy_engine` | string | Policy engine (`"opa"`, `"cedar"`, `"casbin"`, `"custom"`) |
| `aitf.identity.authz.policy_id` | string | Matched policy identifier |
| `aitf.identity.authz.policy_version` | string | Policy version |
| `aitf.identity.authz.deny_reason` | string | Reason for denial (if denied) |
| `aitf.identity.authz.conditions` | string | JSON-encoded conditions (for conditional allow) |
| `aitf.identity.authz.scope_required` | string[] | Scopes required for this action |
| `aitf.identity.authz.scope_present` | string[] | Scopes present in token |
| `aitf.identity.authz.risk_score` | double | Risk-based authorization score (0-100) |
| `aitf.identity.authz.context` | string | JSON-encoded authorization context (time, location, etc.) |
| `aitf.identity.authz.privilege_level` | string | `"standard"`, `"elevated"`, `"admin"` |
| `aitf.identity.authz.jea` | boolean | Whether Just-Enough-Access was applied |
| `aitf.identity.authz.time_limited` | boolean | Whether permission is time-limited |
| `aitf.identity.authz.expires_at` | string | Permission expiration ISO timestamp |

---

## Span: `aitf.identity.delegation`

Represents credential delegation in a multi-agent system — On-Behalf-Of flows, token exchange, and authority attenuation.

### Span Name

Format: `identity.delegate {aitf.identity.delegation.delegator} -> {aitf.identity.delegation.delegatee}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.delegation.delegator` | string | Agent/user delegating authority |
| `aitf.identity.delegation.delegator_id` | string | Delegator identity ID |
| `aitf.identity.delegation.delegatee` | string | Agent receiving delegated authority |
| `aitf.identity.delegation.delegatee_id` | string | Delegatee identity ID |
| `aitf.identity.delegation.type` | string | Delegation type (see below) |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.delegation.chain` | string[] | Full delegation chain (ordered from origin) |
| `aitf.identity.delegation.chain_depth` | int | Depth of delegation chain |
| `aitf.identity.delegation.scope_delegated` | string[] | Scopes being delegated |
| `aitf.identity.delegation.scope_attenuated` | boolean | Whether scope was reduced (should always be true) |
| `aitf.identity.delegation.original_scope` | string[] | Original (broader) scope |
| `aitf.identity.delegation.result` | string | `"success"`, `"failure"`, `"denied"`, `"expired"` |
| `aitf.identity.delegation.reason` | string | Reason for delegation |
| `aitf.identity.delegation.task_id` | string | Task this delegation is for |
| `aitf.identity.delegation.expires_at` | string | Delegation expiration ISO timestamp |
| `aitf.identity.delegation.ttl_seconds` | int | Delegation time to live |
| `aitf.identity.delegation.revocable` | boolean | Whether delegation can be revoked |
| `aitf.identity.delegation.proof_type` | string | `"dpop"`, `"mtls_binding"`, `"signed_assertion"` |
| `aitf.identity.delegation.obo_token_id` | string | On-Behalf-Of token identifier |
| `aitf.identity.delegation.act_claim` | string | JWT `act` claim value |

### Delegation Types

| Value | Description |
|-------|-------------|
| `on_behalf_of` | OAuth On-Behalf-Of (OBO) delegation |
| `token_exchange` | OAuth Token Exchange (RFC 8693) |
| `credential_forwarding` | Forwarding credentials to sub-agent |
| `impersonation` | Agent impersonating delegator (with explicit grant) |
| `capability_grant` | Granting specific capabilities to another agent |
| `scoped_proxy` | Scoped proxy access through the delegator |

---

## Span: `aitf.identity.trust`

Represents trust establishment between agents — agent-to-agent authentication, verifiable credential exchange, and trust boundary crossings.

### Span Name

Format: `identity.trust.{aitf.identity.trust.operation} {aitf.identity.trust.peer_agent}`

### Span Kind

`CLIENT`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_id` | string | This agent's identity ID |
| `aitf.identity.agent_name` | string | This agent's name |
| `aitf.identity.trust.operation` | string | Trust operation (see below) |
| `aitf.identity.trust.peer_agent` | string | Peer agent name |
| `aitf.identity.trust.peer_agent_id` | string | Peer agent identity ID |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.trust.result` | string | `"established"`, `"failed"`, `"rejected"`, `"revoked"` |
| `aitf.identity.trust.method` | string | `"mtls"`, `"spiffe"`, `"did_vc"`, `"http_signature"`, `"pki"` |
| `aitf.identity.trust.trust_domain` | string | Trust domain (e.g., SPIFFE trust domain) |
| `aitf.identity.trust.peer_trust_domain` | string | Peer's trust domain |
| `aitf.identity.trust.cross_domain` | boolean | Whether this is a cross-domain trust operation |
| `aitf.identity.trust.vc_type` | string | Verifiable Credential type presented |
| `aitf.identity.trust.vc_issuer` | string | VC issuer |
| `aitf.identity.trust.vc_verified` | boolean | Whether VC was successfully verified |
| `aitf.identity.trust.trust_level` | string | `"none"`, `"basic"`, `"verified"`, `"high"`, `"full"` |
| `aitf.identity.trust.protocol` | string | Trust protocol (`"mcp"`, `"a2a"`, `"custom"`) |
| `aitf.identity.trust.federation_id` | string | Federation identifier (for federated trust) |

### Trust Operations

| Value | Description |
|-------|-------------|
| `establish` | Establish trust with a peer agent |
| `verify` | Verify a peer agent's identity |
| `present_credential` | Present a verifiable credential to a peer |
| `verify_credential` | Verify a peer's verifiable credential |
| `federation_join` | Join a trust federation |
| `federation_leave` | Leave a trust federation |
| `boundary_cross` | Cross a trust boundary between domains |
| `revoke_trust` | Revoke trust with a peer |

---

## Span: `aitf.identity.session`

Represents an identity session — binding an authenticated identity to a set of operations.

### Span Name

Format: `identity.session {aitf.identity.agent_name}`

### Span Kind

`INTERNAL`

### Required Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.agent_id` | string | Agent identity ID |
| `aitf.identity.agent_name` | string | Agent name |
| `aitf.identity.session.id` | string | Identity session ID |
| `aitf.identity.session.operation` | string | Session operation (see below) |

### Recommended Attributes

| Attribute | Type | Notes |
|-----------|------|-------|
| `aitf.identity.session.auth_method` | string | Authentication method used |
| `aitf.identity.session.scope` | string[] | Active session scopes |
| `aitf.identity.session.expires_at` | string | Session expiration ISO timestamp |
| `aitf.identity.session.created_at` | string | Session creation ISO timestamp |
| `aitf.identity.session.last_activity` | string | Last activity ISO timestamp |
| `aitf.identity.session.actions_count` | int | Number of actions in this session |
| `aitf.identity.session.delegations_count` | int | Number of delegations from this session |
| `aitf.identity.session.ip_address` | string | Source IP (if network-based) |
| `aitf.identity.session.user_agent` | string | Client user agent |
| `aitf.identity.session.termination_reason` | string | `"completed"`, `"timeout"`, `"revoked"`, `"error"`, `"manual"` |

### Session Operations

| Value | Description |
|-------|-------------|
| `create` | Create new identity session |
| `refresh` | Refresh session credentials |
| `validate` | Validate an existing session |
| `terminate` | Terminate a session |
| `timeout` | Session timed out |
| `hijack_detected` | Potential session hijacking detected |

---

## Example: Multi-Agent Delegation Chain

```
Span: identity.lifecycle.create agent-orchestrator
  aitf.identity.type: "persistent"
  aitf.identity.credential_type: "spiffe_svid"
  aitf.identity.provider: "spiffe"
  aitf.identity.owner: "platform-team"
  │
  └─ Span: identity.auth agent-orchestrator
       aitf.identity.auth.method: "spiffe_svid"
       aitf.identity.auth.result: "success"
       aitf.identity.auth.scope_granted: ["tools:*", "agents:delegate", "data:read"]
       │
       ├─ Span: identity.authz agent-orchestrator -> customer-db
       │    aitf.identity.authz.decision: "allow"
       │    aitf.identity.authz.action: "read"
       │    aitf.identity.authz.policy_engine: "opa"
       │
       └─ Span: identity.delegate agent-orchestrator -> agent-researcher
            aitf.identity.delegation.type: "on_behalf_of"
            aitf.identity.delegation.scope_delegated: ["data:read"]
            aitf.identity.delegation.scope_attenuated: true
            aitf.identity.delegation.chain: ["user-alice", "agent-orchestrator", "agent-researcher"]
            aitf.identity.delegation.chain_depth: 2
            │
            └─ Span: identity.trust.establish agent-writer
                 aitf.identity.trust.method: "mtls"
                 aitf.identity.trust.result: "established"
                 aitf.identity.trust.cross_domain: false
```

## Example: Ephemeral Task Identity

```
Span: identity.lifecycle.create task-agent-7f3a
  aitf.identity.type: "ephemeral"
  aitf.identity.credential_type: "jwt"
  aitf.identity.ttl_seconds: 300
  aitf.identity.owner: "agent-orchestrator"
  aitf.identity.scope: ["tools:web_search", "data:read:public"]
  │
  ├─ Span: identity.session task-agent-7f3a
  │    aitf.identity.session.operation: "create"
  │    aitf.identity.session.scope: ["tools:web_search", "data:read:public"]
  │    aitf.identity.session.expires_at: "2026-02-16T10:05:00Z"
  │
  ├─ Span: identity.authz task-agent-7f3a -> web-search-api
  │    aitf.identity.authz.decision: "allow"
  │    aitf.identity.authz.jea: true
  │    aitf.identity.authz.time_limited: true
  │
  └─ Span: identity.lifecycle.expire task-agent-7f3a
       aitf.identity.status: "expired"
       aitf.identity.previous_status: "active"
```

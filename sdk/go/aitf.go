// Package aitf provides the AI Telemetry Framework SDK for Go.
//
// AITF is a comprehensive, security-first telemetry framework for AI systems
// built on OpenTelemetry and OCSF. It extends OTel GenAI semantic conventions
// with native support for agentic AI, MCP, Skills, and multi-agent orchestration.
//
// Usage:
//
//	import "github.com/girdav01/AITF/sdk/go"
//
//	instrumentor := aitf.NewInstrumentor()
//	instrumentor.InstrumentAll()
package aitf

import (
	"go.opentelemetry.io/otel/trace"
)

const Version = "1.0.0"

// Instrumentor manages all AITF sub-instrumentors.
type Instrumentor struct {
	tp     trace.TracerProvider
	LLM    *LLMInstrumentor
	Agent  *AgentInstrumentor
	MCP    *MCPInstrumentor
	RAG    *RAGInstrumentor
	Skills *SkillInstrumentor
}

// NewInstrumentor creates a new AITF instrumentor.
// If tp is nil, the global TracerProvider is used.
func NewInstrumentor(opts ...Option) *Instrumentor {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt.apply(&cfg)
	}

	return &Instrumentor{
		tp:     cfg.TracerProvider,
		LLM:    NewLLMInstrumentor(cfg.TracerProvider),
		Agent:  NewAgentInstrumentor(cfg.TracerProvider),
		MCP:    NewMCPInstrumentor(cfg.TracerProvider),
		RAG:    NewRAGInstrumentor(cfg.TracerProvider),
		Skills: NewSkillInstrumentor(cfg.TracerProvider),
	}
}

// InstrumentAll enables all AITF instrumentors.
func (i *Instrumentor) InstrumentAll() {
	i.LLM.Instrument()
	i.Agent.Instrument()
	i.MCP.Instrument()
	i.RAG.Instrument()
	i.Skills.Instrument()
}

// config holds instrumentor configuration.
type config struct {
	TracerProvider trace.TracerProvider
}

func defaultConfig() config {
	return config{}
}

// Option applies configuration to an Instrumentor.
type Option interface {
	apply(*config)
}

type optionFunc func(*config)

func (f optionFunc) apply(c *config) { f(c) }

// WithTracerProvider sets the TracerProvider.
func WithTracerProvider(tp trace.TracerProvider) Option {
	return optionFunc(func(c *config) {
		c.TracerProvider = tp
	})
}

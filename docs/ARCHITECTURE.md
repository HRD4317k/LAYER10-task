Execution Model

The system models GitHub discussions as an event-sourced workflow.

Each stage functions as a deterministic transformation:

Raw Data
  → Canonical Events
      → State Timeline
          → Conflict Analysis
              → Relational Extraction
                  → Canonical Entities
                      → Graph Representation

The LLM is treated as a bounded transformation step inside a controlled execution graph rather than an autonomous agent.

This design aligns with production AI workflow principles:

State progression

Observable checkpoints

Deterministic replay

Controlled stochastic components
# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

This is a learning repository (日积月累) containing hands-on implementations of deep learning models as Jupyter notebooks. Each notebook demonstrates both the PyTorch official API usage and a from-scratch forward pass implementation, verified against PyTorch's output for correctness.

## Repository Structure

- `classical_models/` — Jupyter notebooks implementing classical sequence models (RNN, LSTM, GRU) with from-scratch forward functions validated against `torch.nn` APIs
- `classical_models/pictures/` — Diagrams referenced by notebooks (forward pass formulas, architecture illustrations)
- `linear_attention&sparse_attention/` — Placeholder for future attention mechanism implementations

## Conventions

- **Language**: All notebook commentary and variable comments are in Chinese
- **Notebook pattern**: Each notebook follows the same structure: (1) explain the model with formula diagrams, (2) call the PyTorch API, (3) implement the forward pass manually, (4) verify outputs match with `print` or `torch.allclose`
- **Variants**: Notebooks cover model variants (e.g., single/multi-layer, unidirectional/bidirectional for RNN; standard/projected for LSTM)
- **Framework**: PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)

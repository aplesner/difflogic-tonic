# Week 0–3 (17.09–08.10.2026)

## Key achievements
- Literature review of LUT-based neural network approaches and identification of gaps we target in the thesis.
- Written thesis proposal including a generalized mathematical formulation of the problem and novel theorems describing key properties.
- Implemented an initial, end-to-end difflut library that integrates with PyTorch. Highlights:
    - Core node abstractions and several node implementations (see `difflut/nodes/`).
    - CUDA kernels for performance-critical operations (under `difflut/nodes/cuda/`).
    - Layer implementations to define connectivity patterns (`difflut/layers/`).
    - Encoder modules (e.g., thermometer encoder in `difflut/encoder/`).
    - A lightweight `registry` for configurable component lookup and wiring.
    - Example usage and tutorial for MNIST in `examples/` (both a notebook and a short script).

## TODO (short-term, prioritized)
1. High priority
     - [ ] experiments:
          - accuracy / num Weights -> nodes
          - num_in,num_out / accuracy -> nodes
          - network_depth / acccuracy -> nodes
          - weights,gradients/epoch -> nodes

     - [ ] Create interface for SystemVerilog extraction
2. Medium priority
     - [ ] Add automated unit tests for node forward/backward (happy path + a few edge cases).

## MISC
- Draft notice military service
- Official Start date 01.11.2026?
- Reimplemented codebase 5 times focused on implemnting based on other ml libaries
- Implemented own WANDB as DB with hydra and 

---

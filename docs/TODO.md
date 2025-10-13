# DiffLogic Tonic - Task & TODO List

This file tracks tasks, improvements, and known issues for the DiffLogic Tonic project.

## Legend
- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority
- ✅ Completed
- 🚧 In Progress
- 📝 Planned

---

## Phase 1: Infrastructure & Config 🔴

### Hydra Configuration Migration
- [✅] 🔴 Replace OmegaConf direct usage with Hydra
- [✅] 🔴 Restructure configs/ with composition (base + overrides)
- [✅] 🔴 Update main.py and prepare_data.py entry points for Hydra
- [✅] 🔴 Update train.sh and prepare_data.sh for Hydra CLI
- [ ] 🟡 Update SLURM job scripts to use Hydra syntax
- [ ] 🟡 Prepare WandB sweeps integration with Hydra
- [ ] 🟡 Update container with hydra-core dependency

**Why**: Current configs have redundancy and repetition. Hydra provides cleaner CLI overrides and better composition.

**Status**: ✅ Core migration complete! Legacy scripts in `legacy/` folder. See [docs/HYDRA_MIGRATION.md](HYDRA_MIGRATION.md).

### SLURM & Cluster Tools
- [✅] 🟡 Add container rsync to train.sh (pre-flight sync)
- [✅] 🟡 Create slurm_jupyter.sh for Jupyter server jobs

---

## Phase 2: Data Pipeline 🔴

### Parquet Dataloader for Raw Events
- [ ] 🔴 Create separate module/codebase for raw event data → Parquet conversion
- [ ] 🔴 Define Parquet schema: (timestamp, x, y, polarity, label, sample_id)
- [ ] 🔴 Implement PyArrow-based DataLoader
- [ ] 🔴 Integrate with current caching system
- [ ] 🟡 HuggingFace datasets integration (replace broken Tonic downloads)

**Why**: Tonic datasets have broken download links. Parquet provides efficient columnar storage for event data and better interoperability.

### Cython Optimization
- [ ] 🟡 Cythonize Tonic ToFrame function
- [ ] 🟡 Cythonize key augmentation operations
- [ ] 🟡 Add Python fallback if Cython unavailable
- [ ] 🟢 Benchmark performance improvements
- [ ] 🟢 Package as separate importable module

**Why**: Event processing (ToFrame, augmentations) is computationally expensive. Cython can provide significant speedups.

---

## Phase 3: Classical CV Experiments 🟡

### RGB Dataset Experiments
- [ ] 🟡 Extend classic_cv.ipynb for CIFAR10/100 experiments
- [ ] 🟡 Add configs for ImageNette dataset
- [ ] 🟡 Implement classical CV filter baselines (Gabor, HOG, SIFT)
- [ ] 🟢 Compare with event-based approaches

**Why**: Classical CV filters on RGB provide baselines for understanding event-based performance.

---

## Phase 4: Temporal Processing 🟡

### Short Frame Classification & Ensemble
- [ ] 🟡 Implement dataset format: [n_samples, n_frames, C, H, W]
- [ ] 🟡 Support multiple short frames per sample (1k-5k events each vs current 15k)
- [ ] 🟡 Implement ensemble/voting aggregation strategy
- [ ] 🟡 Add configs for frame count and aggregation method
- [ ] 🟢 Analyze impact of event aggregation window sizes

**Goal**: Aggregate fewer events per frame (1k-5k instead of 15k), predict per-frame, then ensemble using voting or other strategies.

**Data structure**: Split 15k events into multiple short frames → predict on each → aggregate predictions.

### Recurrent Encoder Network
- [ ] 🟡 Implement GRU-based encoder model
- [ ] 🟡 Add simple decoder for logits from representation
- [ ] 🟡 Process temporal sequence of short frames
- [ ] 🟡 Integrate as new model type (alongside DiffLogic, CNN, MLP)
- [ ] 🟢 Add time-as-channel variant (flatten with time channel)

**Goal**: Introduce time relationships between samples using recurrent models for per-sample predictions.

---

## Phase 5: Advanced Features 🟢

### Learned Connections for DiffLogic
- [ ] 🟢 Literature review: LUT learning for FPGAs
- [ ] 🟢 Make DiffLogic connections learnable (vs random/fixed)
- [ ] 🟢 Implement learnable connection parameters
- [ ] 🟢 Maintain backward compatibility with existing models

**Why**: Papers show learned LUTs improve FPGA performance. Apply similar concepts to DiffLogic.

### CUDA Integration (Simon's LUT6 Functions)
- [ ] 🟢 Integrate Simon's CUDA functions for learned connections
- [ ] 🟢 Implement fast LUT6 model inference
- [ ] 🟢 Benchmark CUDA vs PyTorch implementations
- [ ] 🟢 Add conditional compilation/import

**Why**: Simon implemented CUDA functions for learned connections and LUT6 models. Leverage for performance.

---

## Completed Features ✅

### Training Pipeline
- [✅] Add support for learning rate schedulers (PyTorch Lightning)
- [✅] Implement early stopping mechanism (PyTorch Lightning)
- [✅] Add gradient clipping options (PyTorch Lightning)
- [✅] Support for mixed precision training optimization
- [✅] Add validation metrics tracking (F1, precision, recall)
- [✅] Migrate to PyTorch Lightning for cleaner training code
- [✅] Add wandb integration (PyTorch Lightning WandbLogger)

### Data Processing
- [✅] Do test/train split before processing samples
- [✅] Include metadata extraction pipeline for frame duration analysis
- [✅] Refactor data preparation for code reuse
- [✅] Use torchvision transforms v2 for data augmentation
- [✅] Add configurable data augmentation (flip probability, random crop, etc.)

### Models
- [✅] Discretize difflogic models

### Checkpointing
- [✅] Implement automatic checkpoint cleanup (Lightning ModelCheckpoint)
- [✅] Add support for loading specific checkpoint by timestamp

---

## Backlog & Future Work

### Models
- [ ] 🟢 Implement IWP variant
- [ ] 🟢 Implement CLGNs

### Data Processing
- [ ] 🟢 Add support for additional neuromorphic datasets (DVS-Gesture, N-Caltech101)
- [ ] 🟢 Add dataset statistics and visualization tools

### Infrastructure
- [ ] 🟢 Export final models to ONNX format
- [ ] 🟢 Validate checkpoint recovery with different configurations
- [ ] 🟢 Review error handling for missing environment variables
- [ ] 🟢 Add performance benchmarking tests
- [ ] 🟢 Add docstrings to all public functions and classes

---

## Priority Guidelines
- 🔴 **High Priority**: Critical for core functionality or blocking other work
- 🟡 **Medium Priority**: Important improvements or useful features
- 🟢 **Low Priority**: Nice-to-have enhancements or quality-of-life improvements

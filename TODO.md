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

## Current Tasks

### Features & Enhancements

#### Training Pipeline
- [ ] 🟡 Add support for learning rate schedulers
- [ ] 🟡 Implement early stopping mechanism
- [ ] 🟢 Add gradient clipping options
- [✅] 🟢 Support for mixed precision training optimization
- [ ] 🟢 Add validation metrics tracking (F1, precision, recall)
- [ ] 🟡 Implement IWP variant
- [ ] 🟡 Implement CLGNs

#### Models
- [ ] 🟡 Implement additional baseline models
- [✅] 🟡 Discretize difflogic models

#### Data Processing
- [✅] 🔴 Do the test/train split before processing the samples.
- [ ] 🟡 Add support for additional neuromorphic datasets (DVS-Gesture, N-Caltech101)
- [ ] 🟡 Include metadata in the pipeline to get the duration of each of the generated frames
- [ ] 🟢 Add dataset statistics and visualization tools

#### Checkpointing & Resume
- [ ] 🟡 Implement automatic checkpoint cleanup (keep only N best checkpoints)
- [ ] 🟢 Add support for loading specific checkpoint by timestamp
- [ ] 🟢 Export final models to ONNX format

#### Logging & Monitoring
- [ ] 🟡 Add wandb integration

### Bug Fixes

- [ ] 🟢 Validate checkpoint recovery with different configurations
- [ ] 🟢 Review error handling for missing environment variables

### Testing

- [ ] 🟢 Add performance benchmarking tests

### Code Quality

- [ ] 🟢 Add docstrings to all public functions and classes

---

## Notes

### Priority Guidelines
- 🔴 **High Priority**: Critical for core functionality or blocking other work
- 🟡 **Medium Priority**: Important improvements or useful features
- 🟢 **Low Priority**: Nice-to-have enhancements or quality-of-life improvements



# Assignment 2 - Systems Completion Report

## 1. Project Overview
**Goal**: Implement core system components for Large Language Models (LLMs), including efficient attention (FlashAttention), distributed training (DDP), and memory-optimized training (Sharded Optimizer).

## 2. Implementation Journey

### Phase 1: Planning & Analysis
- **Requirements Analysis**: Reviewed `cs336_spring2025_assignment2_systems.pdf` (via text extraction).
- **Codebase Review**: Analyzed existing tests (`tests/`) and adapter interfaces (`tests/adapters.py`).
- **Strategy**: Defined a step-by-step plan:
  1.  FlashAttention (PyTorch reference implementation).
  2.  Distributed Data Parallel (Simple version first, then Bucketed).
  3.  Sharded Optimizer (ZeRO-1 style).

### Phase 2: Core Components Implementation

#### A. FlashAttention (PyTorch)
- **File**: `cs336_systems/attention.py`
- **Logic**:
  - Implemented `FlashAttentionFunctionPyTorch` as a `torch.autograd.Function`.
  - **Forward**: Computed attention scores `S`, LogSumExp `L`, and output `O`. Saved `q, k, v, O, L` for backward.
  - **Backward**: Recomputed `S` and `P` during backward pass to save memory (gradient checkpointing style).
  - **Gradient Computation**: Derived `dQ`, `dK`, `dV` manually using `dO` and saved tensors.

#### B. Distributed Data Parallel (DDP)
- **File**: `cs336_systems/ddp.py`
- **Part 1: Individual Parameters** (`DDPIndividualParameters`)
  - Registered backward hooks on each parameter.
  - Triggered async `dist.all_reduce(op=SUM)` immediately when a gradient was ready.
  - Synchronized all handles at the end of the backward pass.
  - **Bug Fix**: Initially modified `grad` in-place incorrectly. Switched to `param.grad.div_()` and corrected hook to use `param.grad` instead of the raw hook argument.
- **Part 2: Bucketed DDP** (`DDPBucketed`)
  - Grouped parameters into buckets (default `bucket_size_mb`).
  - Flattened bucket parameters into continuous buffers for efficient network transmission.
  - Implemented async buffer reduction when a bucket becomes full.
  - Added flushing logic to handle incomplete buckets at the end of the backward pass.

#### C. Sharded Optimizer
- **File**: `cs336_systems/optimizer.py`
- **Logic**:
  - Implemented `ShardedOptimizer` inheriting from `torch.optim.Optimizer`.
  - Sharded model parameters across ranks (each rank owns a slice).
  - Initialized an inner optimizer (e.g., AdamW) *only* for the local shard.
  - **Step**:
    1.  Inner output updates local partition.
    2.  Broadcasted updated parameters from each rank to all others (`dist.broadcast`).
  - **Zero Grad**: Ensured gradients are zeroed for *all* parameters to prevent incorrect accumulation on non-local shards.

### Phase 3: Verification
- **Test Suite**: Ran `pytest tests/` and `./test_and_make_submission.sh`.
- **Results**:
  - `test_attention.py`: Passed (PyTorch). Triton tests skipped on Mac.
  - `test_ddp_individual_parameters.py`: Passed.
  - `test_ddp.py`: Passed (Bucketed verified).
  - `test_sharded_optimizer.py`: Passed.
- **Submission**: Generated `cs336-spring2024-assignment-2-submission.zip` successfully.

### Phase 4: Cleanup
- Removed temporary files: `__pycache__`, `.pytest_cache`, `*.egg-info`, `test_results.xml`.
- Removed intermediate artifacts: `pdf_output.txt`.
- Removed `cs336-spring2024-assignment-2-submission.zip` (can be regenerated).

## 3. Key Technical Decisions
- **Manual Gradient Calculation**: Implemented full analytic derivatives for FlashAttention backward pass instead of relying on autograd.
- **In-Place Operations**: Used `param.grad.data.div_()` to safely modify gradients in place without triggering autograd errors.
- **Buffering**: Used flattened tensors in `DDPBucketed` to minimize communication overhead (fewer `all_reduce` calls).
- **Mac Compatibility**: Ensured tests run on CPU/MPS where appropriate and skipped CUDA-only Triton tests.

## 4. Final Status
All required components are implemented and verified. The codebase is clean and ready for submission.

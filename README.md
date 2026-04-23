# Adaptive Recursive RAG for Multi-Hop QA

This repository contains an end-to-end research notebook that benchmarks three Retrieval-Augmented Generation (RAG) control strategies for multi-hop question answering on HotpotQA.

The project focuses on one core question:

Can adaptive, verification-guided retrieval preserve strong answer quality while reducing unnecessary model calls compared to fixed-depth recursion?

## Repository Scope

Current repository contents:

- `nb.ipynb`: primary implementation, benchmarking, and visualization notebook

At this stage, the notebook is the project. It includes setup, model loading, retrieval, generation, adaptive control logic, evaluation, and plotting.

## Problem Statement

Single-pass RAG is often insufficient for multi-hop QA because required evidence may be distributed across multiple passages. A fixed recursive strategy can recover some missed evidence but spends the same compute budget on easy and hard questions.

This project implements an adaptive controller that uses claim-level verification to decide whether to:

- stop and return an answer,
- refine the query,
- retrieve more evidence, or
- abstain when support is insufficient.

## Implemented Modes

All modes share the same dataset sample, retriever stack, and generator model. The difference is only in control policy.

1. `standard`
- One retrieval pass, one generation pass.
- Fastest, lowest control overhead.

2. `recursive`
- Fixed retrieval schedule across multiple steps.
- More compute, no adaptive stopping.

3. `adaptive`
- Iterative loop with claim extraction and NLI-based verification.
- Uses policy actions to continue, refine, stop, or abstain.

## Technical Architecture

The notebook pipeline combines the following components:

- Dataset: HotpotQA (`distractor` split)
- Retrieval embeddings: `BAAI/bge-small-en-v1.5`
- Vector index: FAISS (`IndexFlatIP`)
- Generator: `microsoft/Phi-3-mini-4k-instruct` (4-bit quantized)
- Verifier: `cross-encoder/nli-deberta-v3-small`

### Models and Core Tech (Explicit)

| Layer | Component | Used In Notebook |
|---|---|---|
| Generator LLM | `microsoft/Phi-3-mini-4k-instruct` | Answer generation, reasoning traces, query decomposition |
| Quantization | `bitsandbytes` (4-bit NF4) | Memory-efficient model loading |
| Embedding model | `BAAI/bge-small-en-v1.5` | Query/passage embeddings for retrieval |
| Verifier model | `cross-encoder/nli-deberta-v3-small` | Claim-level entailment checking |
| Vector DB | `FAISS` (`IndexFlatIP`) | Similarity search over passage embeddings |
| Dataset | `hotpot_qa` (`distractor`, validation sample) | Multi-hop QA benchmark data |
| Framework stack | `transformers`, `accelerate`, `sentence-transformers` | Model loading and inference |
| Analysis stack | `pandas`, `matplotlib`, `numpy` | Metrics aggregation and plots |

High-level flow:

1. Build a deduplicated passage corpus from sampled HotpotQA contexts.
2. Embed corpus and index with FAISS.
3. Retrieve top passages for a query.
4. Generate answer (+ reasoning trace format).
5. Extract claims from answer/reasoning.
6. Verify claims against retrieved docs.
7. Use policy thresholds to stop, refine, retrieve more, or abstain.
8. Score outputs and aggregate metrics per mode.

## Notebook Section Map

The notebook is organized into clear stages:

- Setup and imports
- Dataset loading and corpus construction
- FAISS index construction
- Phi-3 model loading (4-bit)
- NLI verifier loading
- Core retrieval, generation, and verification functions
- Pipeline orchestrator (`run_pipeline`)
- Benchmark runner (all modes over sampled questions)
- Results and tabular evaluation
- Step-efficiency and ablation cells
- Final visualizations

## Evaluation Metrics

The notebook reports a broad set of quality and efficiency metrics:

- Exact Match (EM %)
- F1 and length-penalized F1
- Faithfulness / grounding score
- True hallucination rate
- Retrieval-failure proxy rate (faithful but wrong)
- Abstention rate
- Average steps
- Average LLM calls
- Average latency
- EM per LLM call (cost-efficiency proxy)
- Adaptive step-efficiency curve

## Pipeline Results and Findings

The notebook benchmark compares `standard`, `recursive`, and `adaptive` on 75 sampled HotpotQA validation questions.

### Main Results (from current notebook run)

| Mode | EM (%) | Length-Penalized F1 | Faithfulness (%) | True Hallucination (%) | Avg LLM Calls | Avg Latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 65.3 | 0.672 | 94.7 | 5.3 | 1.0 | 5.8 |
| Recursive | 64.0 | 0.665 | 95.5 | 4.0 | 3.0 | 20.5 |
| Adaptive | **68.0** | **0.697** | **100.0** | **0.0** | 1.2 | 6.3 |

### Step-Efficiency (Adaptive)

- Step 1: 65%
- Step 2: 71%
- Step 3: 0%

### Key Findings

- Adaptive mode achieves the best EM and best length-penalized F1 in this run.
- Adaptive mode reaches full faithfulness in the reported benchmark and eliminates true hallucinations.
- Recursive mode is the most expensive in call count and latency.
- Adaptive mode stays close to standard in cost while outperforming it on quality metrics.
- A substantial gap between faithfulness and EM indicates retrieval limitations still exist even when answers are evidence-grounded.

## Quick Start

1. Open `nb.ipynb` in Kaggle, VS Code Jupyter, or JupyterLab.
2. Ensure GPU runtime is enabled (recommended).
3. Run notebook cells sequentially from top to bottom.
4. Let model downloads and index construction finish.
5. Run benchmark and evaluation cells.
6. Inspect printed tables and generated plots.

## Dependencies

The notebook installs required packages in its setup cell:

- `transformers==4.41.2`
- `accelerate==0.30.1`
- `bitsandbytes`
- `sentence-transformers==2.7.0`
- `faiss-cpu`
- `datasets`
- `pandas`
- `matplotlib`

## Runtime and Hardware Notes

- GPU is strongly recommended; the notebook is tuned for Kaggle T4-like resources.
- First run may be slow due to model/dataset downloads.
- Benchmark runtime scales with sample size and mode count.
- Quantized loading is used to fit the generator model more reliably in constrained VRAM.

## Output Artifacts

When run in Kaggle-style environments, the notebook writes artifacts such as:

- summary CSV files (for benchmark metrics)
- benchmark figures (PNG plots)

Paths are currently configured to Kaggle working directories in parts of the notebook (for example, `/kaggle/working/...`). If you run locally, adjust output paths accordingly.

## Reproducibility Guidance

To keep runs comparable:

- Use the same sample size and split settings.
- Keep package versions aligned with the setup cell.
- Avoid changing policy thresholds unless conducting an ablation.
- Report both quality and efficiency metrics, not EM alone.

## Limitations

- Results depend on retrieval quality from the sampled context corpus.
- Benchmark outcomes can vary across hardware/runtime conditions.
- String/heuristic correctness checks may not capture all semantic equivalences.
- Some policy thresholds are tuned heuristically and may require retuning for new datasets.

## Intended Use

This repository is intended for:

- research experimentation on adaptive RAG control,
- classroom or project demonstrations of retrieval-verification loops,
- extending to stronger verifiers, retrievers, or alternative policy logic.

## Project Goal

Demonstrate that adaptive verification-driven retrieval can reach competitive multi-hop QA quality while reducing average compute compared to fixed recursive retrieval, and while lowering unsupported answer risk.

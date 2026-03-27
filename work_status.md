# Pair Programming Work Status

This document tracks our progress while implementing the **Adaptive Recursive RAG with Binary Evidence Verification** benchmark.

## Implementation Checklist

### 1. Setup & Foundations
- [x] Create Kaggle Notebook file (`adaptive_rag_project.ipynb`)
- [x] Set up library imports (`faiss`, `transformers`, `bitsandbytes`, `datasets`, `sentence_transformers`, `matplotlib`, `pandas`)
- [x] Write markdown structural headers for all sections

### 2. Data Pipeline
- [x] Load HotpotQA dataset (`validation` split, `range(75)`)
- [x] Build and deduplicate the retrieval corpus
- [x] Verify corpus size (expected 600–900 unique passages)

### 3. Retrieval System
- [x] Load `BGE-small-en-v1.5` embedder
- [x] Embed the corpus using `normalize_embeddings=True`
- [x] Build `faiss.IndexFlatIP` (Inner Product) index
- [x] Write `retrieve(query, existing_docs)` with deduplication logic
- [x] Verify query uses BGE retrieval prefix

### 4. Language Model & Prompts
- [x] Load Phi-3 Mini 4-bit (bitsandbytes)
- [x] Implement `llm_call()` base function (chat template + generation params)
- [x] Implement `generate()` with exact system + user prompt
- [x] Implement `extract_claims()` with numbered list parsing + fallback
- [x] Implement `verify()` with Yes/No parsing + conservative False default

### 5. Orchestration & Modes
- [x] Build `run_pipeline(query, ground_truth, mode)` orchestrator
- [x] Standard RAG control loop (1 step, no verify)
- [x] Always Recursive RAG control loop (exactly 3 steps, no verify)
- [x] Adaptive RAG control loop (verify → refine → append → max_depth=2 → abstain)
- [x] Confirm output dict format: `{answer, steps, verified, abstained, mode, correct, latency}`

### 6. Demo Cell
- [x] Pick 1 representative question
- [x] Run all 3 modes on it
- [x] Print full intermediate outputs (retrieved docs, answer, claims, verification results, final decision)

### 7. Benchmarking & Evaluation
- [ ] Run benchmark over all 75 questions × 3 modes
- [ ] Calculate accuracy per mode
- [ ] Calculate hallucination rate per mode (post-hoc verify for Standard + Recursive)
- [ ] Calculate abstention rate (Adaptive only)
- [ ] Calculate average steps per mode
- [ ] Calculate average latency per mode
- [ ] Render summary comparison table (Pandas)

### 8. Visualization & Save Outputs
- [ ] Bar chart: Accuracy by mode
- [ ] Bar chart: Average steps by mode
- [ ] Bar chart: Hallucination rate by mode
- [ ] Save `results.csv` to `/kaggle/working/`
- [ ] Save all 3 charts as PNG to `/kaggle/working/figures/`
- [ ] Print abstention examples cell (show which queries abstained and why)

**Current Status:** _Phase 6 (Demo Cell) completed. Ready for Phase 7 (Benchmarking & Evaluation)._

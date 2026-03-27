# Adaptive Recursive RAG Framework 🧠

This project implements a self-contained, deterministic **Retrieval-Augmented Generation (RAG)** benchmark using Kaggle's T4 GPU. Its core objective is to evaluate whether an active, verification-driven **Adaptive Retrieval** loop can achieve the accuracy of brute-force Recursive RAG, but with significantly fewer computational steps and lower hallucination rates.

## What It Does

The system evaluates multi-hop reasoning questions from the **HotpotQA** dataset across three distinct RAG architectures:

1. **Standard RAG (Baseline):** 
   - A single-shot retrieval loop. Retrieves $k$ documents, feeds them into the generic generation prompt, and halts immediately. Fast but natively weak on multi-hop reasoning.
2. **Always Recursive RAG:**
   - Brute-force accumulation. Forces the system into exactly 3 fixed retrieval-generation cycles (`k_schedule = [8, 9, 10]`) over the same static query before concluding. 
3. **Adaptive RAG (State-of-the-Art):**
   - An intelligent, autonomous reasoning loop. At each step, it extracts atomic factual claims from its generated answer and passes them through an independent LLM verification prompt against the retrieved context. 
   - If a majority vote of claims passes, the loop stops early.
   - If claims fail, it triggers an LLM to actively re-synthesize a brand-new focal subquery (`refine_query`) and loops again to retrieve new documents.
   - Includes protective guards against "abstention" hallucinations to elegantly surrender when data is provably absent.

## How the System Works

The entire pipeline is driven by small, powerful local models to ensure tight, 100% reproducible execution:
- **Embedding / Retrieval:** `BAAI/bge-small-en-v1.5` bound to a purely in-memory normalized inner-product `FAISS` index.
- **Generation / Validation:** `microsoft/Phi-3-mini-4k-instruct` loaded in 4-bit quantization, running strictly deterministic parameters (`temperature=0.0`, `do_sample=False`).

### Key Architectural Failsafes
- **Semantic Ground-Truth Alignment:** Pre-calculated cosine similarities automatically map manual queries directly to physical database objects to guarantee metric integrity.
- **Decoupled Request Flow:** Extracted refined queries are utilized exclusively for FAISS embedding space routing. Generation calls strictly evaluate the original user intention, immunizing the system against LLM query-drift.
- **Lexical Overlap Filters:** Extracted assertions are heuristically parsed for semantic overlap with the core string to purge grammatical fluff from entering the deep LLM verify-vote cycle.

## What is Expected

By the end of the final benchmark phase, the system should generate a structured `Pandas DataFrame` containing metrics for all 75 dataset questions across all 3 algorithmic modes. The expected hypothesis output is:
- **Standard:** Low latency, Low accuracy.
- **Recursive:** High latency, High accuracy, Maximum steps.
- **Adaptive:** High accuracy matching Recursive, but with statistically significant reductions in Average Steps and Hallucination Rates.

## What's Left (The Final Sprint)

All of the core RAG architectures and orchestration mechanics have entirely been built, verified, and cleanly tested in the `Demo` cell. 

We only have two final procedural checkpoints remaining:
- [ ] **Phase 7 (Benchmarking & Evaluation):** Executing the heavy 1,800-call LLM iteration loop over all 75 questions inside the Kaggle environment to formally calculate Accuracy, Hallucination %, Abstention Rate, Avg Steps, and Avg Latency metrics. *(Requires runtime diligence/JSON checkpoints to guard against Kaggle session disconnects).*
- [ ] **Phase 8 (Visualization):** Piping the calculated DataFrame into Matplotlib bars representing metric comparisons and saving them to disk (`/kaggle/working`). 

---
*Built incrementally in pair-programming workflow.*

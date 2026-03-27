# Complete Project Context — Adaptive Recursive RAG with Binary Evidence Verification

---

## 1. The Problem

Large Language Models generate text based on probability distributions learned during training. They do not verify facts before outputting them. This means even a confident, fluent, well-structured answer can be completely wrong. This failure mode is called hallucination — the model produces false information with no warning signal.

The standard fix for this is Retrieval-Augmented Generation, commonly called RAG. Instead of relying purely on internal weights, the model is given external documents at inference time and instructed to base its answer on those documents. This significantly reduces hallucination because the model now has a factual grounding.

However, standard RAG has a critical limitation: it retrieves once, generates once, and outputs. If the retrieval step misses an important document — which happens regularly on complex, multi-hop questions — the answer will still be wrong or incomplete. The model has no mechanism to recognize this failure and no way to recover from it.

There are systems that try to fix this with recursive retrieval — they simply loop several times regardless of whether the answer is already correct. This improves accuracy but wastes computation on easy questions and has no intelligent stopping condition.

There are also systems like Self-RAG that try to have the model reflect on its own outputs. But this relies on the model judging itself, which is particularly unreliable for Small Language Models — the very models you want to use because they are cheaper and faster.

The core unsolved problem is: **how does a system know when it does not have enough information, and how does it act on that knowledge efficiently?**

---

## 2. What This Project Is

This project builds a system called **Adaptive Recursive RAG with Binary Evidence Verification**.

It is not a new model. It is not a trained system. Nothing is fine-tuned. It is a **runtime architecture** — a control loop built around an existing Small Language Model that uses document-grounded verification to decide whether to retrieve more information or output a final answer.

The core idea is simple: after generating an answer, the system extracts factual claims from that answer and checks each one against the retrieved documents. If every claim is supported, the answer is output. If any claim is not supported, the system refines its query, retrieves more documents, appends them to the existing context, and regenerates. This loop runs until all claims pass or a maximum depth is reached, at which point the system abstains rather than hallucinating.

The research contribution is not the individual components — retrieval, generation, and verification all exist separately in the literature. The contribution is the **controlled integration**: using binary evidence verification as the decision signal for adaptive retrieval, implemented on a Small Language Model, with three configurable modes that allow direct comparison of control strategies.

---

## 3. What Problem This Specifically Solves

Standard RAG fails on multi-hop questions — questions that require synthesizing facts from multiple documents. HotpotQA is built entirely around these questions. A question like "What year did the director of Inception graduate from university?" requires finding who directed Inception, then finding where that person went to university, then finding the graduation year. One retrieval step is almost never enough.

The Adaptive system handles this naturally. The first retrieval might surface documents about Inception. The verification step finds the graduation year claim unsupported. The refined query now targets the director's education. The second retrieval surfaces the right document. Verification passes. Answer is output.

---

## 4. The Three Benchmark Modes

These are not three separate projects. They are three configurations of the exact same pipeline. Same model, same retriever, same FAISS index, same dataset. Only the control logic changes.

### Mode 1 — Standard RAG (Baseline 1)

The simplest possible configuration. Query arrives, retrieve top-3 documents, generate answer, output immediately. No verification, no looping, no checking of any kind. This represents the current default approach used in most production RAG systems. It is fast but unreliable on complex queries.

Expected behavior: works well on simple factual questions, fails frequently on multi-hop questions, never abstains, always takes exactly 1 retrieval step.

### Mode 2 — Always Recursive RAG (Baseline 2)

A brute-force improvement over Standard RAG. The pipeline always runs exactly 3 complete cycles of retrieve-then-generate regardless of answer quality. Each cycle appends new documents to the existing context. No verification is used. There is no early stopping.

This exists to answer the question: does simply retrieving more times improve accuracy? The answer is yes, but at fixed high cost. This mode cannot adapt — it wastes the same compute on a trivial question as it does on a complex one.

Expected behavior: higher accuracy than Standard RAG, always takes exactly 3 steps, slightly higher latency, no abstention.

### Mode 3 — Adaptive Recursive RAG (Your System)

The intelligent configuration. After each generation step, claims are extracted and verified against documents. If verification passes, output immediately. If it fails, refine the query, retrieve more, regenerate. Maximum depth is 2 cycles. If depth 2 is reached with claims still failing, abstain.

Expected behavior: accuracy comparable to Always Recursive, but average steps closer to 1–2 rather than always 3, lower average compute, can abstain on genuinely unanswerable questions.

---

## 5. System Architecture — All Components

### Component 1 — Embedding Model (BGE-small)

BGE-small-en is a lightweight sentence embedding model from BAAI. It converts text passages and queries into dense vectors in a shared semantic space. It was chosen because it is small enough to run efficiently on Kaggle's free GPU tier, produces high-quality embeddings for retrieval tasks, and is available directly from HuggingFace.

Every passage in the corpus is embedded once at the start of the notebook and stored in FAISS. At query time, the query is embedded and the nearest vectors are retrieved.

### Component 2 — Vector Database (FAISS)

FAISS (Facebook AI Similarity Search) is an in-memory vector search library. It stores all passage embeddings and performs fast approximate nearest-neighbor search. For our corpus size of roughly 500–1000 passages, FAISS runs entirely in RAM in under a second per query. No external database, no server, no connection required.

### Component 3 — Language Model (Phi-3 Mini, 4-bit quantized)

Phi-3 Mini is Microsoft's compact but capable language model. It is used for three distinct tasks in this pipeline: generating answers, extracting claims, and verifying claims. All three use the same loaded model instance — just different prompts.

It is loaded with 4-bit quantization using the bitsandbytes library. This reduces memory from roughly 14GB to around 4–5GB, making it stable on Kaggle's free T4 GPU (16GB VRAM) alongside FAISS and the embedding model.

### Component 4 — Retrieval Function

Takes a query string and the existing document list. Embeds the query, searches FAISS for top-3 nearest passages, returns those passages as strings. In the recursive and adaptive modes, new passages are appended to the existing list rather than replacing it. This means context accumulates across steps, giving the model increasingly rich grounding with each cycle.

### Component 5 — Generation Function

Takes the original query and the current document list. Constructs a prompt that includes all documents as context and instructs Phi-3 Mini to answer based only on the provided documents. Returns a plain text answer string.

### Component 6 — Claim Extraction Function

Takes the generated answer string. Sends it to Phi-3 Mini with a prompt instructing it to break the answer into 2–3 discrete factual claims. Returns a list of claim strings. If the LLM output cannot be parsed cleanly, falls back to sentence splitting.

### Component 7 — Binary Verification Function

Takes the list of claims and the full combined document context. For each claim, sends a prompt to Phi-3 Mini asking whether the claim is supported by the documents — answer must be Yes or No. Returns a list of boolean values. Verification is done against the combined context, not per individual document.

### Component 8 — Pipeline Orchestrator

The central function `run_pipeline(query, ground_truth, mode)`. Contains the complete control logic for all three modes. Returns a standardized result dictionary.

---

## 6. End-to-End System Flow

### Standard RAG flow:
Query → retrieve(query, []) → generate(query, docs) → return result

### Always Recursive flow:
Query → for i in range(3): retrieve(query, docs) → generate(query, docs) → return result

### Adaptive flow:
```
Query
→ retrieve(query, [])
→ generate(query, docs)
→ extract_claims(answer)
→ verify(claims, docs)
→ if all pass: return answer
→ if any fail:
    new_query = original_query + " " + failed_claim
    retrieve(new_query, docs)  [appends]
    generate(query, docs)
    extract_claims(answer)
    verify(claims, docs)
    → if pass: return answer
    → if still fail: return abstain
```

Maximum 2 cycles total. Abstention only possible in Adaptive mode.

---

## 7. Dataset

**HotpotQA** — a multi-hop question answering dataset specifically designed to require reasoning across multiple documents.

Each entry in HotpotQA contains:
- A natural language question
- A ground truth answer (short string — a name, date, yes/no, etc.)
- A list of context entries, each being a (title, list-of-sentences) pair — typically 10 paragraphs per question
- Supporting facts indicating which sentences are actually relevant

We use the `distractor` split from HuggingFace (`hotpot_qa`, `distractor` configuration). This split includes both supporting paragraphs and distractor paragraphs per question, making retrieval non-trivial.

We sample 50–100 questions for the benchmark run.

The corpus for FAISS is built by flattening all context paragraphs from all sampled questions into one global list. A passage is constructed as: `title + " " + " ".join(sentences)`. This global pool of roughly 500–1000 passages becomes the FAISS index. All three modes retrieve from this same index.

Ground truth answers are used for evaluation. Matching is done by checking if the lowercased, punctuation-stripped ground truth string appears as a substring of the lowercased, punctuation-stripped answer string.

---

## 8. Technology Stack

| Component | Tool | Why |
|---|---|---|
| Language Model | Phi-3 Mini (microsoft/Phi-3-mini-4k-instruct) | Small, capable, fits on free Kaggle GPU |
| Quantization | bitsandbytes (4-bit) | Reduces VRAM from ~14GB to ~4GB |
| Model loading | HuggingFace transformers + accelerate | Standard, well-documented |
| Embeddings | BGE-small-en-v1.5 (BAAI/bge-small-en-v1.5) | Lightweight, high quality |
| Vector search | FAISS (faiss-cpu) | Fast, in-memory, no server needed |
| Dataset | HuggingFace datasets (hotpot_qa) | Direct load, no manual download |
| Numerical ops | NumPy | Embedding array handling |
| Results & display | Pandas | Benchmark results table |
| Visualization | Matplotlib | Accuracy/steps/hallucination graphs |
| Platform | Kaggle Notebook (free T4 GPU) | Free GPU, sharable, reproducible |

---

## 9. Platform — Kaggle Notebook

The entire project lives in a single Kaggle notebook. There is no local setup required. No separate scripts, no Flask app, no separate files to run. The notebook is the project.

Kaggle provides:
- Free T4 GPU (16GB VRAM) — enough for Phi-3 Mini 4-bit + FAISS + BGE-small
- Persistent internet access during sessions for HuggingFace downloads
- Notebook sharing via URL for submission
- Up to 12 hours of runtime per session

The notebook is structured so it can be run top-to-bottom in one execution. Each section is clearly labeled with markdown headers.

---

## 10. Folder / File Structure

Since this is a single Kaggle notebook, there are no external files that need to be created ahead of time. Everything is self-contained. However, here is the complete logical structure of what exists:

```
kaggle-notebook/
│
├── adaptive_rag_project.ipynb        ← THE ENTIRE PROJECT (one file)
│
│   [Section 1]  Installation & imports
│   [Section 2]  Dataset loading & corpus construction
│   [Section 3]  FAISS index construction
│   [Section 4]  Model & tokenizer loading (Phi-3 Mini 4-bit)
│   [Section 5]  Embedding model loading (BGE-small)
│   [Section 6]  Core functions
│                  - retrieve(query, existing_docs)
│                  - generate(query, docs)
│                  - extract_claims(answer)
│                  - verify(claims, docs)
│   [Section 7]  Pipeline orchestrator
│                  - run_pipeline(query, ground_truth, mode)
│   [Section 8]  Benchmark runner
│                  - loops all 3 modes × all questions
│                  - stores results in list of dicts
│   [Section 9]  Results & evaluation
│                  - Pandas DataFrame of all results
│                  - Accuracy per mode
│                  - Hallucination rate per mode
│                  - Abstention rate (Adaptive only)
│                  - Average steps per mode
│                  - Average latency per mode
│   [Section 10] Visualization
│                  - Bar chart: Accuracy comparison
│                  - Bar chart: Average steps comparison
│                  - Bar chart: Hallucination rate comparison
│                  - Table: Full benchmark results
│   [Section 11] Demo cell
│                  - Single query walkthrough
│                  - Shows all intermediate outputs
│
└── /kaggle/working/
    ├── results.csv                   ← saved benchmark results
    └── figures/
        ├── accuracy_comparison.png
        ├── steps_comparison.png
        └── hallucination_comparison.png
```

The `/kaggle/working/` directory is Kaggle's writable output directory. CSV and PNG files saved there are automatically available as notebook outputs that can be downloaded.

---

## 11. What the Final System Looks Like

The final deliverable is a **Kaggle notebook** — not an app, not a terminal tool, not a web UI.

This is intentional. The project is a research benchmark. The notebook format is the standard for ML research demonstrations. It shows code, outputs, and analysis together in one readable document.

### What a grader / evaluator sees when they open the notebook:

**Top section** — markdown explaining the project, problem, and approach. Clean, readable.

**Setup sections** — installation, loading. These run silently.

**Demo cell (most important visual section)** — a single query run through all three modes with full intermediate output printed, like this:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY: Were Scott Derrickson and Ed Wood 
       both American directors?
GROUND TRUTH: yes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODE: standard
  Retrieved docs: 3
  Answer: "Yes, both Scott Derrickson and Ed Wood were American directors."
  Steps: 1
  Verified: False (not checked)
  Abstained: False
  Correct: True
  Latency: 4.2s

MODE: recursive
  Retrieved docs: 9 (3 per loop × 3 loops)
  Answer: "Yes. Scott Derrickson is an American director known for 
           horror films. Ed Wood was also an American director."
  Steps: 3
  Verified: False (not checked)
  Abstained: False
  Correct: True
  Latency: 18.7s

MODE: adaptive
  Retrieved docs: 3
  Answer: "Yes, both were American directors."
  Claims extracted: ["Scott Derrickson is American", 
                     "Scott Derrickson is a director",
                     "Ed Wood is American", 
                     "Ed Wood is a director"]
  Verification: [True, True, True, True]
  → All claims supported. Stopping at depth 1.
  Steps: 1
  Verified: True
  Abstained: False
  Correct: True
  Latency: 12.1s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Benchmark results table** — Pandas DataFrame printed inline showing all 50–100 questions × 3 modes with all metrics.

**Comparison summary table:**

```
Mode          Accuracy    Avg Steps    Hallucination    Abstention    Avg Latency
Standard      XX%         1.0          XX%              0%            ~4s
Recursive     XX%         3.0          XX%              0%            ~19s
Adaptive      XX%         ~1.5         XX%              ~X%           ~10s
```

**Three matplotlib charts** displayed inline:
- Grouped bar chart: Accuracy by mode
- Grouped bar chart: Average steps by mode  
- Grouped bar chart: Hallucination rate by mode

**Abstention examples cell** — prints the specific queries where Adaptive chose to abstain, showing what claims failed and why. This demonstrates the safety mechanism concretely.

---

## 12. Evaluation Metrics — Exact Definitions

**Accuracy** — percentage of questions where the final answer contains the ground truth string (after lowercasing and stripping punctuation for both). Abstained answers count as incorrect.

**Hallucination rate** — percentage of final answers (non-abstained) that contain at least one claim marked as unsupported (No) by the verifier. For Standard and Recursive modes where verification is not run, this is estimated by running the verifier post-hoc on the final answer without triggering any retrieval — purely for measurement.

**Abstention rate** — percentage of questions where the system returned no answer. Only possible in Adaptive mode. Reported separately from accuracy.

**Average steps** — mean number of retrieval cycles across all questions in that mode. Standard = always 1. Recursive = always 3. Adaptive = between 1 and 2.

**Average latency** — mean wall-clock time per query in seconds.

---

## 13. Expected Results Pattern

Standard RAG will show lowest accuracy on multi-hop questions, highest hallucination rate, fastest speed, always 1 step.

Always Recursive will show improved accuracy, medium hallucination rate, always 3 steps, highest total latency.

Adaptive will show accuracy comparable to Recursive, lowest hallucination rate, average steps between 1 and 2 (significantly less than 3), moderate latency, and a small abstention rate representing the system correctly refusing unanswerable questions.

The core finding being demonstrated: **you can match brute-force recursive performance at lower average cost by using verification as the stopping signal.**

---

## 14. Research Justification for Corpus Choice

The corpus used for FAISS is the union of all context paragraphs bundled within the sampled HotpotQA questions. This is a deliberate methodological choice. Using full Wikipedia would test retrieval quality, not control logic. This project's contribution is the control logic — the verification-driven adaptive loop. By using a controlled corpus of known-relevant passages mixed with distractors, we isolate the variable we actually care about: does verification-driven adaptive retrieval outperform fixed-loop retrieval as a control strategy? The corpus choice eliminates corpus quality as a confounding variable.

---

## 15. What This Project Is NOT

It is not a new model architecture. It is not a fine-tuned system. It is not a multi-agent framework. It is not a web application or production system. It is not claiming to beat state-of-the-art RAG systems on open benchmarks.

It is a focused, honest comparison of three retrieval control strategies on a Small Language Model, demonstrating that adaptive verification-driven retrieval achieves comparable accuracy to brute-force recursive retrieval while using fewer retrieval steps and producing fewer hallucinations.

---

## 16. One-Sentence Summary for Any Audience

> We built a system that checks whether each factual claim in its answer is supported by retrieved documents, and only searches for more information when a claim fails — making a small language model more accurate and more efficient than both single-shot and always-recursive retrieval approaches.

---

This is the complete, final, locked project context. Every implementation decision flows from this document. Code can now be written against this spec with no ambiguity.
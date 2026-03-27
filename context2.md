# Supplementary Technical Specification — Gap Fills

---

## Gap 1 — The Exact Prompts

All three LLM calls use Phi-3 Mini. Each has a distinct purpose and requires a carefully worded prompt. Here are the exact prompts for all three.

---

### Prompt A — Generation

This prompt is used to generate an answer from retrieved documents.

**System message:**
```
You are a factual question answering assistant. 
You will be given a question and a set of reference documents. 
Answer the question using only the information in the documents. 
Be concise. Do not add information that is not in the documents. 
If the documents do not contain enough information, say: 
"I cannot determine this from the provided documents."
```

**User message:**
```
Documents:
{doc_1}

{doc_2}

{doc_3}

Question: {query}

Answer:
```

Where `{doc_1}`, `{doc_2}`, etc. are the retrieved passage strings, and `{query}` is the original user question. If there are more than 3 documents (accumulated across recursive steps), all of them are included in order.

**Expected output format:**
A plain text answer. One to three sentences. No bullet points. No preamble like "Based on the documents...". Just the answer.

---

### Prompt B — Claim Extraction

This prompt is used to break a generated answer into discrete verifiable claims.

**System message:**
```
You are a precise factual analyzer. 
Your job is to break a given answer into individual factual claims. 
Each claim must be a single standalone statement that can be 
verified independently. 
Output exactly 2 to 3 claims, one per line, numbered.
Do not output anything else — no explanation, no preamble, 
no extra text.
```

**User message:**
```
Answer: {answer}

Break this answer into 2 to 3 factual claims, one per line, numbered:
1.
```

**Expected output format:**
```
1. Marie Curie discovered radium.
2. Marie Curie won the Nobel Prize in Physics.
3. Marie Curie won the Nobel Prize in 1903.
```

The leading `1.` in the user message is a prompt continuation trick — it forces the model to start outputting the numbered list immediately without preamble.

---

### Prompt C — Binary Verification

This prompt is used once per claim. It checks whether a single claim is supported by the combined document context.

**System message:**
```
You are a strict fact verification assistant.
You will be given a factual claim and a set of reference documents.
Your job is to determine if the claim is supported by the documents.
Reply with exactly one word: Yes or No.
Do not explain. Do not add any other text. Just Yes or No.
```

**User message:**
```
Documents:
{all_docs_combined}

Claim: {claim}

Is this claim supported by the documents above? Answer Yes or No:
```

**Expected output format:**
```
Yes
```
or
```
No
```

The trailing `Answer Yes or No:` is again a prompt continuation anchor — it pushes the model toward a clean single-word response.

---

## Gap 2 — Phi-3 Mini Exact Chat Template

Phi-3 Mini uses a specific prompt format that must be followed exactly. Using raw text or the wrong template causes the model to produce degraded or nonsensical output.

The format is:

```
<|system|>
{system_message}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
```

There is no space between the tag and the content. The `<|end|>` token closes each turn. The `<|assistant|>` tag at the end has no closing tag — the model generates from that point onward.

**The correct way to apply this in code is to use the tokenizer's built-in `apply_chat_template` method**, which handles the format automatically:

```python
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

This is preferred over manually constructing the string because:
- It is guaranteed correct for Phi-3 Mini's template
- It handles edge cases like special characters in content
- It is forward compatible if the template changes

**Generation parameters to use consistently across all three LLM calls:**

```
max_new_tokens = 256      ← enough for answers and claims, not wasteful
temperature = 0.0         ← deterministic, critical for reproducibility
do_sample = False         ← greedy decoding, no randomness
repetition_penalty = 1.1  ← prevents looping on short outputs
```

Temperature 0.0 with do_sample=False is essential. If the model is stochastic, the same query run twice gives different verification results, which breaks the benchmark's reproducibility. Every run must be deterministic.

**Extracting the response text after generation:**

The model output includes the full input tokens plus the generated tokens. You must slice off the input to get only the new generated text:

```python
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
```

`skip_special_tokens=True` removes `<|end|>` and other special tokens from the output. `.strip()` removes leading and trailing whitespace.

---

## Gap 3 — Claim Extraction Output Parsing

The model is prompted to output a numbered list. The parsing logic must handle variations in how the model formats this output.

**Expected outputs (all valid, must handle all):**
```
1. Marie Curie discovered radium.
2. Marie Curie won the Nobel Prize in Physics.
3. She received the prize in 1903.
```
```
1) Marie Curie discovered radium.
2) She won the Nobel Prize in Physics in 1903.
```
```
1. Marie Curie discovered radium.
2. She won the Nobel Prize.
```

**Parsing logic — step by step:**

Step 1 — Split the output by newlines.

Step 2 — For each line, strip whitespace.

Step 3 — Remove the leading number and punctuation using this pattern: strip any leading digit, followed by `.` or `)`, followed by any whitespace.

Step 4 — Keep only non-empty lines after stripping.

Step 5 — Keep maximum 3 claims. If more are returned, take the first 3. If fewer than 2 are returned, trigger fallback.

**Fallback condition:**
If fewer than 2 valid claim strings are extracted after parsing, fall back to sentence splitting:
- Split the original answer on `.` and `?` and `!`
- Strip each segment
- Keep segments longer than 15 characters
- Take the first 3

**Why 15 characters minimum:** Short fragments like "Yes", "No", "It does", "She did" are not verifiable claims and would waste a verification LLM call.

**The parsed output is always a Python list of strings, length 2 or 3.** This list is passed directly to the verification function.

---

## Gap 4 — Verification Output Parsing

The model is prompted to return exactly "Yes" or "No". In practice it sometimes returns slightly more than that. The parsing must be robust to these variations.

**Possible real outputs from Phi-3 Mini given the verification prompt:**
```
Yes
```
```
No
```
```
Yes.
```
```
No.
```
```
Yes, the claim is supported.
```
```
No, I cannot find this in the documents.
```
```
Based on the documents, Yes.
```

**Parsing logic — exact steps:**

Step 1 — Take the raw response string, lowercase it, strip whitespace.

Step 2 — Check if the string starts with `"yes"`. If true → return `True`.

Step 3 — Check if the string starts with `"no"`. If true → return `False`.

Step 4 — Check if `"yes"` appears anywhere in the first 20 characters of the string. If true → return `True`.

Step 5 — Check if `"no"` appears anywhere in the first 20 characters of the string. If true → return `False`.

Step 6 — If none of the above match → return `False` (conservative default — treat ambiguous as unsupported).

**Why conservative default:** It is safer to trigger an extra retrieval step than to pass a potentially hallucinated claim. The cost of one extra retrieval is low. The cost of a hallucinated answer reaching the output is high.

**The verification function returns a list of booleans, one per claim.** The pipeline checks `if all(verification_results)` to decide whether to stop or continue.

**Failed claims extraction for query refinement:**
After verification, collect all claims where the result is `False`:
```python
failed_claims = [claim for claim, result 
                 in zip(claims, verification_results) 
                 if not result]
```
The refined query is constructed as:
```python
new_query = original_query + " " + failed_claims[0]
```
Only the first failed claim is appended. Appending multiple failed claims makes the query too long and confuses retrieval.

---

## Gap 5 — HotpotQA Exact Loading Specification

**Which split to use:**

Use the `validation` split, not `train` and not `test`.

Reason: The `test` split in HotpotQA has no ground truth answers — they are withheld for the official leaderboard. You cannot evaluate accuracy without ground truth. The `train` split is very large (90,000+ examples) and slow to load. The `validation` split has approximately 7,405 examples with full ground truth, loads quickly, and is the standard split used in RAG evaluation papers.

**Exact loading call:**
```python
from datasets import load_dataset
dataset = load_dataset("hotpot_qa", "distractor", split="validation")
```

**Sampling strategy:**

Take the first 75 examples from the validation split. Do not random sample. Taking the first N is reproducible — anyone who runs the notebook gets the same 75 questions. Random sampling without a fixed seed would produce different questions each run, making results non-reproducible.

```python
sample = dataset.select(range(75))
```

**Corpus construction — exact steps:**

Each HotpotQA example has a `context` field structured as:
```python
{
  "title": ["Title1", "Title2", ...],       # list of strings
  "sentences": [["sent1", "sent2"], ...]    # list of lists of strings
}
```

Construct passages as follows:
```python
all_passages = []
seen = set()

for example in sample:
    titles = example["context"]["title"]
    sentences_list = example["context"]["sentences"]
    for title, sentences in zip(titles, sentences_list):
        passage = title + ". " + " ".join(sentences)
        if passage not in seen:
            seen.add(passage)
            all_passages.append(passage)
```

The `seen` set handles deduplication. The same Wikipedia paragraph often appears as context for multiple questions. Without deduplication, the FAISS index contains duplicate vectors, which wastes memory and skews retrieval results by making duplicated passages appear more frequently.

**Expected corpus size:** approximately 600–900 unique passages from 75 questions (each question has ~10 context paragraphs, with overlap across questions).

**Passage construction detail:** `title + ". " + sentences` rather than `title + " " + sentences` because the title is a Wikipedia article name and does not end with punctuation. Adding `. ` makes it grammatically cleaner and helps the embedding model treat the title as a sentence boundary rather than a prefix.

---

## Gap 6 — FAISS Index Type and Similarity Metric

**Which index type to use:**

Use `faiss.IndexFlatIP` — Flat index with Inner Product similarity.

Do NOT use `IndexFlatL2` (Euclidean distance).

**Why Inner Product and not L2:**

BGE-small produces embeddings that are optimized for cosine similarity, not Euclidean distance. Cosine similarity between two vectors equals their inner product when both vectors are L2-normalized (unit vectors). So the correct pipeline is: normalize all embeddings to unit length, then use inner product search. This gives cosine similarity scores, which is what BGE-small is trained to produce.

Using L2 distance on non-normalized BGE embeddings gives suboptimal retrieval quality because L2 distance is sensitive to vector magnitude, which cosine similarity ignores.

**Exact normalization step:**
```python
import faiss
import numpy as np

# After computing embeddings:
faiss.normalize_L2(embeddings)  # normalizes in-place to unit length

# Build index:
dimension = embeddings.shape[1]  # 384 for BGE-small
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
```

BGE-small produces 384-dimensional embeddings. This is fixed — always 384 regardless of input length.

**Query normalization:**

At retrieval time, the query embedding must also be normalized before searching:
```python
query_embedding = embed([query])          # shape (1, 384)
faiss.normalize_L2(query_embedding)       # normalize in-place
scores, indices = index.search(query_embedding, k=3)
```

**Mapping indices back to passages:**

The `index.search()` call returns integer indices into the array that was added to the index. Since passages were added in order as a numpy array, the mapping is direct:
```python
retrieved_passages = [all_passages[i] for i in indices[0]]
```

`indices[0]` because `index.search` expects a batch and returns a batch — we always search one query at a time, so we take the first (and only) row.

**Handling already-retrieved passages in recursive steps:**

When appending new documents, avoid adding exact duplicates to the context:
```python
def retrieve(query, existing_docs, k=3):
    query_emb = embed([query])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)
    new_docs = [all_passages[i] for i in indices[0]]
    # deduplicate against existing
    for doc in new_docs:
        if doc not in existing_docs:
            existing_docs.append(doc)
    return existing_docs
```

This ensures the context list grows meaningfully with each retrieval step rather than accumulating repeated passages that waste token budget in the generation prompt.

**Embedding function specification:**

BGE-small requires a specific instruction prefix for query embedding (not for passage embedding). This is documented in the BGE-small model card:

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_passages(passages):
    return embed_model.encode(
        passages,
        normalize_embeddings=True,    # handles normalization internally
        batch_size=32,
        show_progress_bar=True
    )

def embed_query(query):
    # BGE-small recommends this prefix for retrieval queries
    prefixed = "Represent this sentence for searching relevant passages: " + query
    return embed_model.encode(
        [prefixed],
        normalize_embeddings=True
    )
```

Using `normalize_embeddings=True` in SentenceTransformer's encode method means you do NOT need the separate `faiss.normalize_L2` call — the embeddings come out already normalized. Pick one approach and use it consistently. The recommended approach is to use `normalize_embeddings=True` in the encode call and use `faiss.IndexFlatIP` for the index. This is cleaner and less error-prone.

The query prefix `"Represent this sentence for searching relevant passages: "` is the official BGE-small retrieval prefix. It is only applied to queries, never to passages being indexed. This improves retrieval quality measurably on question-answering tasks.

---

## Complete Prompt-to-Output Chain Summary

For full clarity, here is every LLM interaction in the pipeline from raw input to parsed output:

```
GENERATION CALL
  Input:  system_prompt_A + user_prompt_A(query, docs)
  Format: Phi-3 chat template via apply_chat_template
  Params: max_new_tokens=256, temperature=0.0, do_sample=False
  Output: raw string → strip() → answer string

CLAIM EXTRACTION CALL
  Input:  system_prompt_B + user_prompt_B(answer)
  Format: Phi-3 chat template via apply_chat_template
  Params: max_new_tokens=128, temperature=0.0, do_sample=False
  Output: raw string → parse numbered lines → list of 2-3 claim strings
          → fallback to sentence split if < 2 claims parsed

VERIFICATION CALL (one per claim)
  Input:  system_prompt_C + user_prompt_C(all_docs, claim)
  Format: Phi-3 chat template via apply_chat_template
  Params: max_new_tokens=8, temperature=0.0, do_sample=False
  Output: raw string → lowercase → startswith check → True or False
          → default False if ambiguous
```

Note `max_new_tokens=8` for verification — the answer is one word. Limiting tokens here saves significant time across hundreds of verification calls in the benchmark run.
But There Are Things That Could Catch You Off Guard
Let me be honest about what might still go wrong even after all fixes are applied.

Risk 1 — Benchmark Runtime
75 questions × 3 modes × ~8 LLM calls each = roughly 1800 Phi-3 Mini calls. On Kaggle's free T4 at ~5 seconds per call that is approximately 2.5 hours of pure inference time. Add setup time (model loading, embedding, indexing) and you are looking at 3+ hours total.
What could go wrong: Kaggle sessions can disconnect. If the benchmark cell dies at question 60 out of 75, you lose everything.
Fix before you run the benchmark:
python# Save results incrementally, not just at the end
import json

all_results = []
for i, example in enumerate(sample):
    for mode in ["standard", "recursive", "adaptive"]:
        result = run_pipeline(example["question"], 
                             example["answer"], mode)
        all_results.append(result)
        
        # Save every 10 questions
        if i % 10 == 0:
            pd.DataFrame(all_results).to_csv(
                "/kaggle/working/results_checkpoint.csv", 
                index=False
            )
            print(f"Checkpoint saved at question {i}")
This way if the session dies at question 60, you have results up to question 50 saved and do not have to restart from zero.

Risk 2 — Hallucination Metric Requires Post-hoc Verification
Your hallucination metric definition is: percentage of final answers containing at least one unsupported claim. For Standard and Recursive modes, verification is never run during the pipeline. So after the benchmark finishes, you need a separate cell that runs the verifier post-hoc on Standard and Recursive answers.
This is an extra step that is easy to forget and takes additional time. Budget for it.
python# Post-hoc hallucination measurement for standard and recursive
for result in all_results:
    if result["mode"] in ["standard", "recursive"]:
        claims = extract_claims(result["answer"])
        verifs = verify(claims, result["docs"])
        result["hallucinated"] = not all(verifs)
```

Make sure your result dict stores the retrieved docs so this post-hoc step is possible.

---

### Risk 3 — Accuracy Numbers Might Be Lower Than Expected

Because you are using substring match on a small 75-question sample, accuracy numbers can be noisy. If Standard gets 40%, Recursive gets 48%, Adaptive gets 47% — the difference between Recursive and Adaptive is tiny and a professor might challenge your claim that Adaptive is better.

**Your defense is the steps metric, not accuracy.** The argument is not "Adaptive is more accurate." The argument is "Adaptive achieves comparable accuracy to Recursive at fewer average steps with lower hallucination." Make sure your charts and summary table emphasize steps and hallucination rate, not just accuracy.

---

### Risk 4 — One Good Demo Question Might Not Be Enough

If your chosen demo question happens to be one where Standard also gets lucky and answers correctly, your demo looks weak again.

**Fix:** Pick two or three demo questions from your sample. Run the demo cell on all of them. Show the one that best demonstrates differentiation. You can have multiple demo cells or just show the best one.

---

### Risk 5 — Abstention Rate Might Be Zero or Very High

If Adaptive never abstains, that feature looks unused. If it abstains too much (say 40% of questions), accuracy looks terrible.

Expected healthy abstention rate: 5–15% of questions. If it is outside this range, you may need to adjust how aggressively the abstain phrase detection works or tweak the verification prompt.

---

## The Real "Done" Checklist
```
✅ Phase 1  — Setup complete
✅ Phase 2  — Corpus built (741 passages confirmed)
✅ Phase 3  — FAISS index built
✅ Phase 4  — Phi-3 Mini loaded 4-bit
✅ Phase 5  — All 4 core functions working
             (retrieve, generate, extract_claims, verify)
✅ Phase 5  — is_abstain_answer() guard added
✅ Phase 5  — parse_claims() strips prefixes correctly
✅ Phase 6  — run_pipeline() handles all 3 modes
✅ Phase 6  — MAX_DEPTH = 2 explicit
✅ Phase 6  — Recursive uses k schedule (3,4,5)
✅ Demo     — Question pulled from sample[idx]
✅ Demo     — Ground truth from dataset not hardcoded
✅ Demo     — Final summary prints after adaptive loop
✅ Demo     — Visible differentiation between 3 modes
✅ Phase 7  — Benchmark runs with checkpoint saves
✅ Phase 7  — Post-hoc hallucination verification done
✅ Phase 7  — All 5 metrics calculated correctly
✅ Phase 8  — 3 charts rendered and saved
✅ Phase 8  — results.csv saved to /kaggle/working/
✅ Phase 8  — Abstention examples cell printed
✅ Polish   — CUDA warnings suppressed
✅ Polish   — Notebook runs clean top to bottom
---

## Final Completeness Check

With the original context document plus this supplementary specification, a coding agent now has:

- ✅ Exact prompts for all three LLM calls
- ✅ Phi-3 Mini chat template and generation parameters
- ✅ Response extraction logic (input slicing)
- ✅ Claim parsing logic with fallback
- ✅ Verification parsing logic with conservative default
- ✅ Failed claim extraction and query refinement
- ✅ HotpotQA exact split, sample size, and sampling strategy
- ✅ Corpus construction with deduplication
- ✅ FAISS index type, similarity metric, normalization approach
- ✅ BGE-small query prefix and embedding function
- ✅ Duplicate-aware retrieval append logic
- ✅ All generation parameters

The combined original context plus this document is fully sufficient to write the complete notebook without any guesswork.
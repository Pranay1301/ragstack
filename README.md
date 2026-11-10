# ragstack — RAG from scratch

Retrieval-augmented generation with **no framework**: no LangChain, no
vector database, no embedding service. Chunking, TF-IDF embeddings, exact
cosine search, MMR diversification and citation-grounded answer composing,
each as one small NumPy/Python module with its own tests.

Built to show the whole pipeline end to end — and measured, not asserted
(see **Measured results** below).

## Architecture

1. **Chunking** (`ragstack/chunking.py`) — packs text under a size cap,
   breaking on paragraph → line → sentence → word boundaries, with an
   overlapping tail carried into the next chunk.
2. **Embeddings** (`ragstack/embeddings.py`) — TF-IDF with sub-linear TF
   scaling and a small from-scratch stemmer, L2-normalized so dot product
   is cosine similarity.
3. **Index** (`ragstack/index.py`) — flat exact index; one matrix product
   per query. Brute force on purpose: at this scale it is exact and simple
   to verify.
4. **Retrieval** (`ragstack/retrieve.py`) — top-k plus MMR, so the context
   is not five near-duplicates of the same paragraph.
5. **Answering** (`ragstack/generate.py`) — a deterministic extractive
   composer: it stitches adjacent retrieved chunks back into whole
   passages, picks the sentences that best overlap the question, and cites
   the chunk each sentence came from. An optional free-tier LLM backend
   (any OpenAI-compatible endpoint via `RAGSTACK_LLM_API_KEY`) can phrase
   the answer instead, restricted to the retrieved chunks and required to
   cite them.

## Quickstart

```bash
pip install -r requirements.txt          # just numpy

# index the bundled demo corpus and ask a question
python3 -m ragstack.cli data/demo-corpus "How are taxes applied to an order?"

# or interactive (one question per line)
python3 -m ragstack.cli data/demo-corpus
```

## Tests and eval

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q     # 31 tests
python3 -m evals.run_eval       # retrieval + answer eval on the demo corpus
```

## Measured results

Executed on this machine, 2026-11-11, Python 3.10, commit `b8691d6`+:

- **31/31 unit tests pass** (`python3 -m pytest tests/ -q`).
- **Retrieval recall: 15/15** — for all 15 questions in
  `evals/questions.json`, a chunk from the expected document appears in the
  top 5 (`python3 -m evals.run_eval`).
- **Answer key-phrase coverage: 15/15** — the deterministic extractive
  answer contains every expected key phrase for all 15 questions, same run.

Scope of these numbers, honestly: the corpus is the 5 small bundled
documents (~900 words each), the questions are written against it, and the
answer backend is the extractive composer. They say the pipeline works end
to end on this fixture; they are **not** a claim about large corpora or
unseen question distributions. The eval script prints the per-question
table, so misses would be visible, not hidden.

## Honest limits

- TF-IDF retrieval is **lexical**. It matches words, not meaning, so
  heavy paraphrases ("how do I pay" vs "settle the bill") can miss where a
  neural embedding model would not. The stemmer covers plurals and common
  verb endings only.
- The index is exact and in-memory — fine for thousands of chunks, not for
  millions.
- The extractive composer quotes; it does not paraphrase or synthesize
  across sources. The optional LLM backend does, but its output quality is
  not part of the measured numbers above.
- `evals/questions.json` is a fixture, not a benchmark. 15/15 on it means
  the machinery works, nothing more.

## Layout

```
ragstack/      the library (chunking, embeddings, index, retrieve, generate, pipeline, cli)
tests/         31 unit tests
evals/         question fixture + eval runner
data/demo-corpus/  5 short docs used by the demo and the eval
```

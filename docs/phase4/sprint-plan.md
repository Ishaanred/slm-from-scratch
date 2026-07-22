# Phase 4 Sprint Plan — Knowledge Distillation

**Goal:** Distill a large teacher (Qwen 3.6 35B-A3B MoE) into your own small student, and check whether the distilled student actually beats a from-scratch baseline trained on the same data. Per `docs/plan.md`, deliverable is a distilled 150M model beating the from-scratch baseline, plus real understanding of KL divergence at scale.

**Not started yet.** This plan is scoping only — nothing built.

---

## Step 0 — Teacher setup, revised from what's actually on this machine

`docs/plan.md`'s original Task 4.1 assumed setting up llama.cpp from scratch. Checked directly instead of assuming:

- **The teacher model is already downloaded**: `~/AI/models/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF`, 35GB. No re-download needed.
- **"A3B" means ~3B active parameters per token** (MoE) — inference compute cost is much closer to a 3B dense model than a 35B one, which matters a lot for whether this is usable on a 16GB card.
- **Ollama is already installed** (`/usr/local/bin/ollama`) and working — it already serves other local models (`llama3.2`, `deepseek-coder`, `bge-m3` per `ollama list`). Ollama bundles its own llama.cpp-based inference engine, so there's no need to separately build the `llama-cpp-turboquant` source checkout that's also present on this machine — that's a parallel, unbuilt path, not a dependency.
- **Not yet done**: the Qwen 3.6 35B-A3B model isn't registered in Ollama's own model store yet (absent from `ollama list`) — it exists as a raw GGUF in the HuggingFace cache, not as something Ollama can serve directly. Needs `ollama create <name> -f Modelfile` pointing at that GGUF path, or an equivalent pull.
- **Not yet verified**: whether it actually runs at a usable speed on the 5070 Ti. A 35GB model file doesn't fit in 16GB VRAM as a single block — llama.cpp/Ollama can offload inactive layers to system RAM, and the MoE's low active-param count should make this far more tractable than a dense 35B model, but "should be more tractable" is a claim to test, not assume, given this project's track record this week of doc claims not matching measured reality (VRAM checks, throughput checks, crash isolation all mattered).

**First real action**: register the model in Ollama, run a single inference call, and measure tokens/sec. That number decides whether local generation is viable or whether the RunPod cloud fallback from `docs/plan.md` is actually needed.

---

## Step 1 — Pick the student's size

`docs/plan.md` suggests 150M-500M. Phase 3 gives one real input here: Run 3 (150M @ 500M tokens) came out with the lowest val loss of all three Phase 3 runs, beating both the 75M-at-same-data and 75M-at-4x-data runs (see `docs/phase3/results.md`). That's a data point in favor of 150M as a reasonable starting size — not a settled decision, and not mine to make. Whether it generalizes to a distillation setting (different loss, different data regime) is worth thinking through before locking in.

---

## Step 2 — Generate teacher logits (infrastructure, not the learning-objective piece)

For each training sample: run it through the teacher, capture the top-K logits (`docs/plan.md` says K=8192), cache to disk as memory-mapped `.npy` — same corpus this project already has tokenized (`data/phase2/full/`, 32K BPE). This is data-pipeline plumbing (batch inference, caching, memmap I/O) — happy to write this the way `src/phase3/train.py`'s harness was written.

Storage estimate from the plan: ~50-100GB for 1B tokens of top-K logits. Disk has 641GB free — no constraint here. Time estimate depends entirely on Step 0's measured teacher throughput.

---

## Step 3 — Distillation loss variants — this is the "you write it" boundary

Per this project's own `CLAUDE.md`: *"Distillation loss implementation — The KL divergence, temperature scaling, hybrid loss. You write it. Claude can explain the math."* Same boundary as `train_step()` in Phases 1-3. I'll set up the surrounding harness (data loading, training loop, checkpointing) exactly like before, but the actual loss function — hard CE, soft KL with temperature, or the hybrid mix — is yours to implement, same pattern as `configure_optimizers`/`train_step` in the existing `model.py`/`train.py` files.

`docs/plan.md`'s Task 4.3 says compare all three variants on a small token budget (75-100M tokens) before committing to one for the full run — worth keeping that as the actual first training step here, not skipping to the full-scale run.

---

## Step 4 — Train the real distilled student + baseline

Distilled student at the chosen size, plus a from-scratch baseline on the same data (same architecture, same tokens, no teacher) — the actual comparison the whole phase is testing. Compare loss curves, and whatever benchmarks Phase 5 ends up using.

---

## Order of work

1. Step 0 (teacher setup + throughput measurement) — must happen first, decides local-vs-cloud for everything downstream
2. Step 1 (student size) — quick decision, informed by Phase 3's data
3. Step 2 (logit generation harness) — depends on Step 0's throughput number to estimate real generation time
4. Step 3 (loss variants, small-scale comparison) — yours to implement, infra provided
5. Step 4 (full run + baseline) — the bulk of the compute time

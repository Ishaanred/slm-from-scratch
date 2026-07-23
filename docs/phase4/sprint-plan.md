# Phase 4 Sprint Plan — Knowledge Distillation

**Goal:** Understand knowledge distillation conceptually and practically — what running it actually requires on real hardware, not just the theory. This doc records a real infrastructure investigation: the teacher setup, measured throughput, a genuine architectural blocker (tokenizer mismatch), and the reasoning behind the decisions made, ahead of and instead of a full production distillation run.

---

## Step 0 — Teacher setup — done, verified working

`docs/plan.md`'s original Task 4.1 assumed setting up llama.cpp from scratch. What's actually true, checked directly rather than assumed:

- **The teacher model is already downloaded**: `~/AI/models/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF`, 35GB (specifically the `UD-Q4_K_XL` quant). No re-download needed.
- **"A3B" means ~3B active parameters per token** (MoE) — the model has a purpose-built config (`models-preset.ini`, `n-gpu-layers=10`, CPU-offloaded experts, attention on GPU) using only ~3-4GB VRAM.
- **The actual serving path is Docker, not Ollama.** An existing image (`llama-cpp-turboquant:server-cuda13`) and container (`llama-turboquant-server`) were built for this exact model on this exact GPU. The Ollama route considered earlier is unnecessary — this container already does the job.
- **Fixed**: the container failed to start due to a stale NVIDIA CDI device spec (built for driver 580.159.03, host had since upgraded to 595.71.05). Fixed with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, regenerating against the current driver. Container now starts clean and reports healthy.
- **Verified working end-to-end**: sent real chat completion requests through `http://localhost:8081/v1/chat/completions` (model `Qwen3.6-35B-A3B-GGUF-UD-Q4_K_XL`) and got correct responses.

**Measured throughput (2026-07-23), not guessed:**

| Request | Prompt size | Prompt processing (prefill) | Decode |
|---|---|---|---|
| Small | 20 tokens | 98.5 tok/s | 133.8 tok/s |
| Realistic (matches this project's block_size=1024) | 1312 tokens | **1119.9 tok/s** | **81.3 tok/s** |

This confirms the hypothesis from before the fix: the recalled "~80 tok/s" figure was decode speed (81.3 tok/s measured, near-exact match) — not the number that matters for Task 2. **Batched/prefill throughput is ~14x faster than decode** (1119.9 vs 81.3 tok/s) at a realistic sequence length, because prefill parallelizes across the sequence dimension instead of paying the CPU-offloaded-expert cost one token at a time. This was a single sequential request, not even using the server's available `n_parallel=4` slots — real batched throughput across multiple concurrent sequences could be higher still, worth testing when Step 2 is actually built.

**Revised Task 2 time estimate**, using the measured 1119.9 tok/s (conservative, single-request) instead of the old 80 tok/s guess:
- 200M tokens (disk-constrained fallback from `docs/plan.md`): **~50 hours (~2 days)** — not the ~29 days a decode-speed estimate would have implied
- 1B tokens (the plan's upper target): **~248 hours (~10 days)**

These are still real numbers to revisit once Step 2's actual batching is built (it'll very likely beat single-request prefill), but they're honest planning inputs now, not speculation.

---

## Step 1 — Student size — decided: 300M

`docs/plan.md` suggests 150M-500M. Phase 3's Run 3 (150M @ 500M tokens, lowest loss of the three runs) was the initial lean, but once it was confirmed teacher and student never run concurrently (see Step 2's note below), VRAM stopped being the limiting factor and the decision became "how big, not whether it fits."

**Real numbers checked before deciding**, same measure-don't-assume approach as Phase 3's grid:

| Config | Actual params | VRAM (batch=8, unless noted) | Throughput | Time for 500M tokens |
|---|---|---|---|---|
| 75M | 69.4M | 4.94 GB — measured (Phase 3) | 124,000 tok/s — measured (Phase 3) | ~1.1h |
| 150M | 135.0M | 6.70 GB — measured (Phase 3) | 71,000 tok/s — measured (Phase 3) | ~2.0h |
| **300M** | **268.1M** | **9.60 GB — measured** | **~36,700 tok/s — measured** | **~3.8h** |
| 500M | 555.5M | 11.53 GB @ batch=4 — measured (OOMs at batch=8) | ~16,000 tok/s — estimated only | ~8.5h |

**Decided: 300M** (268.1M actual params, 16L/16H/1024E). Fits comfortably at the full batch=8 (9.60GB of 16.3GB, no batch-size compromise needed unlike 500M), and would have been the largest model this project trained, had a full run gone ahead.

---

## The blocker that changed the plan: teacher/student tokenizer mismatch

Checked directly via the GGUF metadata rather than assumed:

```
Teacher (Qwen 3.6 35B-A3B) vocabulary: 248,320 tokens
Student (this project's 32K BPE tokenizer): 32,000 tokens
```

This isn't a small mismatch — token ID 500 in the teacher's vocabulary and token ID 500 in the student's refer to completely unrelated subwords. Real, direct soft-label KL divergence distillation (comparing full probability distributions token-for-token) requires both models to share a vocabulary, which these don't.

**Tested the fix directly rather than assume it would work:** switching the 300M architecture to the teacher's 248,320-token vocabulary (embeddings aren't tied in this project's `model.py`, so vocab size cost hits both the input embedding table and the output head) balloons the model to **711.1M actual params — which OOMs on this 16.3GB card at every batch size down to 1.** Confirmed measured, not estimated. Re-tokenizing to match the teacher isn't viable on this hardware at any reasonable model size.

The remaining options, and why none were run:
- **Sequence-level distillation** (teacher generates text, retokenize with the student's own tokenizer) — sidesteps the vocab mismatch, but requires the teacher to *generate* new text, which is decode-bound (~81 tok/s measured) rather than the fast prefill regime — a meaningful token budget is back to weeks, not hours. It also risks the student learning a narrower distribution than real web text (a known failure mode when training on synthetic data at scale).
- **Hard-label-only distillation** (detokenize the teacher's top-1 token, retokenize with the student's tokenizer) — the one approach that's actually fast (prefill-bound, ~1120 tok/s measured) and fits this hardware cleanly. But it's a weak signal: it only differs from ordinary training in the specific positions where the teacher's top-1 guess disagrees with the real next token in the corpus. Literature on hard-label-only distillation shows it's often a marginal improvement at best.

Also checked two "free" throughput levers before considering token-budget cuts, both came back negative, real findings worth keeping: sending concurrent requests (using the server's `n_parallel=4`) made aggregate throughput *worse* (~255 tok/s vs ~1120 tok/s for one sequential request), and doubling the server's CPU thread allocation (8 → 15) made no measurable difference — both point at memory bandwidth, not CPU core count or request batching, as the actual ceiling.

---

## Why a full production run wasn't done here

Given the tokenizer mismatch rules out the strongest version of distillation (soft-label KL) on this hardware, and the remaining viable option (hard-label-only) carries a genuinely weak, uncertain signal — running a multi-hour-to-day+ generation job for a result that might well be indistinguishable from noise wasn't a good trade of compute and electricity for this project. The conceptual understanding (`docs/phase4/distillation.html`) and this real infrastructure investigation — teacher setup, measured throughput, the vocabulary blocker, the student-size decision — are this phase's actual output.

A task-focused distillation project (distilling toward a specific application, e.g. customer support) is a better place to actually run this: the teacher and student tokenizers can be chosen to match from the start, and a concrete downstream task gives a much clearer signal of whether distillation is worth its cost than a general next-token-prediction comparison does.

---

## What a full run would have required (for reference)

1. ~~Teacher setup + throughput measurement~~ — done. Container running, real throughput measured (1119.9 tok/s prefill vs 81.3 tok/s decode). Local generation would have been viable, no cloud fallback needed.
2. ~~Student size~~ — decided. 300M (268.1M actual params), fits at full batch=8 (9.60GB), ~36,700 tok/s measured.
3. Generate + cache teacher logits (or hard-label top-1 tokens, given the vocab blocker) — data-pipeline plumbing, would have been built the way `src/phase3/train.py`'s harness was.
4. Distillation loss implementation — per this project's own `CLAUDE.md`, this is the "you write it" boundary, same as `train_step()`/`configure_optimizers` in earlier phases.
5. Train the distilled student + from-scratch baseline, compare.

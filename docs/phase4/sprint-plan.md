# Phase 4 Sprint Plan — Knowledge Distillation

**Goal:** Distill a large teacher (Qwen 3.6 35B-A3B MoE) into your own small student, and check whether the distilled student actually beats a from-scratch baseline trained on the same data. Per `docs/plan.md`, deliverable is a distilled 150M model beating the from-scratch baseline, plus real understanding of KL divergence at scale.

**Not started yet.** This plan is scoping only — nothing built.

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

**Decided: 300M** (268.1M actual params, 16L/16H/1024E). Fits comfortably at the full batch=8 (9.60GB of 16.3GB, no batch-size compromise needed unlike 500M), and will be the largest model this project has trained — appropriate, since distillation is meant to let a smaller model punch above its raw parameter count, and 300M is the ceiling where that's still true without fighting the hardware for room.

---

## Step 2 — Generate teacher logits (infrastructure, not the learning-objective piece)

For each training sample: run it through the teacher, capture the top-K logits (`docs/plan.md` says K=8192), cache to disk as memory-mapped `.npy` — same corpus this project already has tokenized (`data/phase2/full/`, 32K BPE). This is data-pipeline plumbing (batch inference, caching, memmap I/O) — happy to write this the way `src/phase3/train.py`'s harness was written.

Storage estimate from the plan: ~50-100GB for 1B tokens of top-K logits. Disk has 641GB free — no constraint here. Time estimate depends entirely on Step 0's measured teacher throughput.

**Deliberately offline, not online, and this is why:** the teacher generates and caches logits to disk in this step, then exits — it is never loaded again during Step 4's actual training. This is a hard requirement given the hardware, not just a convenience: this card can't hold the teacher (~3-4GB, CPU-offloaded MoE) and the student (75M ≈ 5GB, 150M ≈ 6.7GB, per Phase 3's measurements) in VRAM at once *and* run both efficiently at the same time as a live "online" distillation setup would need. Splitting generation and training into separate sequential phases sidesteps that entirely — neither phase needs more VRAM than Phase 3 already proved fits comfortably. The tradeoff moves elsewhere: disk I/O throughput for streaming cached logits during training becomes the thing to watch instead of VRAM contention.

---

## Step 3 — Distillation loss variants — this is the "you write it" boundary

Per this project's own `CLAUDE.md`: *"Distillation loss implementation — The KL divergence, temperature scaling, hybrid loss. You write it. Claude can explain the math."* Same boundary as `train_step()` in Phases 1-3. I'll set up the surrounding harness (data loading, training loop, checkpointing) exactly like before, but the actual loss function — hard CE, soft KL with temperature, or the hybrid mix — is yours to implement, same pattern as `configure_optimizers`/`train_step` in the existing `model.py`/`train.py` files.

`docs/plan.md`'s Task 4.3 says compare all three variants on a small token budget (75-100M tokens) before committing to one for the full run — worth keeping that as the actual first training step here, not skipping to the full-scale run.

---

## Step 4 — Train the real distilled student + baseline

Distilled student at the chosen size, plus a from-scratch baseline on the same data (same architecture, same tokens, no teacher) — the actual comparison the whole phase is testing. Compare loss curves, and whatever benchmarks Phase 5 ends up using.

---

## Order of work

1. ~~Step 0 (teacher setup + throughput measurement)~~ — **done**. Container running, real batched throughput measured (1119.9 tok/s prefill vs 81.3 tok/s decode). Local generation is viable — no need for the RunPod cloud fallback.
2. ~~Step 1 (student size)~~ — **done**. 300M (268.1M actual params), fits at full batch=8 (9.60GB), ~36,700 tok/s measured.
3. Step 2 (logit generation harness) — build against `http://localhost:8081/v1/chat/completions` (or a lower-level completion endpoint with logprobs), using the measured throughput to size the real token budget. Remember to restart the teacher container first — it was stopped to free VRAM for Step 1's testing.
4. Step 3 (loss variants, small-scale comparison) — yours to implement, infra provided
5. Step 4 (full run + baseline) — the bulk of the compute time

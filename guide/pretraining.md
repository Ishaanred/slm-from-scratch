# Pretraining

[← back to index](README.md)

## The idea, in one analogy

Pretraining is teaching someone a language by having them read an enormous pile of text and, at every single point, guess the next word before seeing it, then correcting them each time they're wrong. No grammar lessons, no labeled examples of "this sentence is correct," just relentless next-word prediction, at a scale where getting good at that game forces the model to implicitly learn grammar, facts, and reasoning patterns as a side effect.

## What's actually happening each training step

**Forward pass.** A batch of token sequences goes through the model (see [transformers.md](transformers.md)), producing a predicted probability distribution over the vocabulary at every position.

**Loss.** Compare the predicted distribution at each position against the actual next token, using cross-entropy loss: a measure of how surprised the model was by the real answer. Lower is better; a loss of 0 would mean perfect, certain prediction.

**Backward pass.** Compute the gradient of that loss with respect to every parameter in the model: which direction each weight should move to make this specific mistake less likely next time.

**Optimizer step.** Nudge every weight slightly in that direction. AdamW is the standard choice for transformers; it adapts the step size per-parameter and applies weight decay (a mild pull toward zero that helps generalization).

Repeat this millions of times across a huge corpus, and "guess the next token" turns into a model that can hold a conversation.

<details><summary>∑ the knobs that actually matter</summary>

- **Learning rate schedule**: typically warmup (ramp up from ~0 over the first few hundred/thousand steps, since large updates early on a randomly-initialized model can destabilize training) followed by cosine decay down toward zero by the end of training.
- **Gradient accumulation**: if the batch size you want doesn't fit in GPU memory, accumulate gradients over several smaller "micro-batches" before taking one optimizer step, simulating a larger batch.
- **Gradient clipping**: cap the norm of the gradient before the optimizer step, preventing a single bad batch from causing a destructive update.
- **Mixed precision (bf16/fp16)**: do the forward/backward math in a lower-precision format to roughly halve memory use and increase throughput, typically keeping a fp32 master copy of weights for the actual update.
</details>

## Checkpointing is not optional

Training runs take hours to weeks and hardware (or software) will eventually fail mid-run. Saving the model weights, optimizer state, and current step count periodically means a crash costs you minutes, not the whole run. The optimizer state matters here, not just the weights: Adam-family optimizers carry per-parameter momentum terms, and resuming without them causes a visible bump in the loss curve right after resume as those statistics rebuild from scratch.

One real trap worth naming: if your code that builds the optimizer's parameter groups ever iterates over an unordered collection (a Python `set`, for instance) instead of an ordered one (a `list`), the parameter order can come out different across runs, silently breaking checkpoint-resume, since the saved optimizer state no longer lines up with the right parameters. Use ordered containers throughout, and actually test resume-from-checkpoint at least once, rather than assuming it works.

## Where to look further

- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) `train.py`: the training loop this whole pattern is descended from.
- Hugging Face's [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer): a maintained, batteries-included implementation of the same loop, if you'd rather not hand-roll it.
- This project's own from-scratch training loop and a real checkpoint-resume bug found along the way: [`slm-from-scratch` Phase 3 results](../docs/phase3/results.md).

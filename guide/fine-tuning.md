# Fine-tuning

[← back to index](README.md)

## The idea, in one analogy

Pretraining is like getting a general education: years of broad reading to become fluent and knowledgeable in general. Fine-tuning is a focused apprenticeship afterward: a few weeks with an expert, on a narrower set of examples, to adapt what you already know to a specific job. It's fast precisely because you're not starting from zero.

## Why it's cheap relative to pretraining

Pretraining (see [pretraining.md](pretraining.md)) has to teach a model language itself, from randomly-initialized weights, which requires enormous amounts of data and compute. Fine-tuning starts from a model that already has that foundation, so it needs orders of magnitude less data (often thousands to low millions of examples, not billions of tokens) and far less compute to noticeably shift behavior toward a task, domain, or style.

The two broad flavors:

**Full fine-tuning**: every parameter in the model gets updated, same mechanics as pretraining, just on a smaller, more targeted dataset and usually a much smaller learning rate. Effective, but still requires enough memory to hold gradients and optimizer state for the entire model.

**Parameter-efficient fine-tuning (PEFT)**: freeze the pretrained weights entirely and train a small number of added parameters instead. LoRA (Low-Rank Adaptation) is the standard technique: instead of updating a weight matrix directly, learn two much smaller matrices whose product gets added to it. The base model's knowledge stays untouched; only a small, cheap-to-train "adapter" changes. This is what makes fine-tuning a 7B+ model feasible on a single consumer GPU. See [advanced.md](advanced.md) for the concrete tools.

## When to fine-tune instead of pretraining from scratch

If the capability you need (general language understanding, broad world knowledge) already exists in an available pretrained model, and what you actually need is a narrower behavior change (a specific tone, a specific task format, domain vocabulary), fine-tuning gets you there for a tiny fraction of the cost of pretraining. Training from scratch is worth it when you need a model whose foundation itself must differ: a different tokenizer/vocabulary for a genuinely different domain or language, a novel architecture, or when the point is to actually learn how the whole pipeline works rather than to ship a product as fast as possible.

## The failure mode worth knowing about

Fine-tuning on a narrow dataset for too long, or too aggressively, degrades the broad capability the base model started with: the model gets better at the narrow task and worse at everything else it used to be able to do ("catastrophic forgetting"). Mixing in some general-purpose data alongside the task-specific data during fine-tuning, and keeping the learning rate low, are the standard mitigations.

## Where to look further

- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685): the original PEFT paper.
- Hugging Face's [PEFT library](https://github.com/huggingface/peft) and [TRL library](https://github.com/huggingface/trl): the standard tooling for LoRA fine-tuning and instruction/preference tuning.
- [Unsloth](https://github.com/unslothai/unsloth): custom kernels that make LoRA/QLoRA fine-tuning significantly faster and lower-memory on consumer GPUs (see [advanced.md](advanced.md)).

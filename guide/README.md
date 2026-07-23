# A field guide to training language models

This isn't a tutorial you read start to finish. It's a reference: short, self-contained modules you look up when you need them, in whatever order your actual problem demands. No module assumes you've read another one first.

It grew out of [slm-from-scratch](..), a project that trained small (75M-500M param) language models from scratch on a single consumer GPU. That project is the "we actually did this and measured it" proof; this guide is the "here's how the pieces fit together in general" reference, written so it's useful even if you're using a completely different stack.

No tool bias. Where a concept has multiple common implementations (a training loop can be raw PyTorch, HF `Trainer`, Unsloth, axolotl...), this guide explains the concept once and lists the options neutrally, rather than assuming you've picked one already.

## Modules

- [**Transformers**](transformers.md): what's actually inside a GPT-style model
- [**Tokenization**](tokenization.md): how text becomes numbers, and why the choice matters
- [**Data engineering**](data-engineering.md): filtering, dedup, and why "garbage in, garbage out" dominates everything else
- [**Pretraining**](pretraining.md): the training loop (loss, optimizers, schedules, checkpointing)
- [**Fine-tuning**](fine-tuning.md): adapting an existing model instead of starting from zero, and when that's the right call
- [**Scaling laws**](scaling-laws.md): how model size, data, and compute trade off, and how to run your own mini version of this study
- [**Distillation**](distillation.md): training a small model to imitate a large one
- [**Evaluation**](evaluation.md): benchmarks, chance floors, and reading your own model's mistakes
- [**Advanced / where to go next**](advanced.md): LoRA/PEFT, RLHF/DPO, quantization, and serving, pointers into the wider ecosystem (Hugging Face and others) once you outgrow "from scratch"

## Definitions

If you only need to know what something means, this section is enough. Each entry links to its module for the actual mechanics.

**Transformer**: the neural network architecture behind essentially every modern language model. Reads a sequence of tokens and, at each position, predicts what comes next, using "attention" to let every position look at every earlier position. → [transformers.md](transformers.md)

**Tokenizer**: the piece that turns raw text into a sequence of integers a model can process (and back again). "Hello world" might become `[15496, 995]`. Almost all modern tokenizers use some form of Byte-Pair Encoding (BPE): common substrings get merged into single tokens. → [tokenization.md](tokenization.md)

**Pretraining**: training a model from randomly-initialized weights on a large, general corpus, learning "language" itself (grammar, facts, reasoning patterns) with no task-specific labels. This is the expensive, from-scratch step. → [pretraining.md](pretraining.md)

**Fine-tuning**: taking an already-pretrained model and continuing training on a smaller, more specific dataset, to adapt it to a task, domain, or style. Orders of magnitude cheaper than pretraining because the model already knows language. → [fine-tuning.md](fine-tuning.md)

**Scaling laws**: empirical relationships between model size, dataset size, compute budget, and resulting loss. Used to answer "if I have X compute, what size model should I train, on how much data?" before spending the compute. → [scaling-laws.md](scaling-laws.md)

**Knowledge distillation**: training a smaller "student" model to match the output distribution of a larger "teacher" model, rather than (or in addition to) matching ground-truth labels. A way to compress capability into a cheaper model. → [distillation.md](distillation.md)

**Evaluation / benchmarking**: running a model against held-out tasks with known correct answers (multiple-choice QA, reasoning, coding problems) to get a comparable, numeric read on capability, instead of eyeballing outputs. → [evaluation.md](evaluation.md)

**LoRA / PEFT**: "parameter-efficient fine-tuning," freezing the pretrained weights and training a small number of additional parameters instead of the whole model, making fine-tuning cheap enough to run on a single GPU. → [advanced.md](advanced.md)

**RLHF / DPO**: techniques for aligning a model's behavior to human preferences (helpfulness, harmlessness, following instructions), applied after pretraining and fine-tuning. → [advanced.md](advanced.md)

## What this guide is not

It's not a from-scratch implementation walkthrough (that's what `slm-from-scratch`'s own `docs/` folder is for) and it's not a replacement for the primary sources it references. Where something is genuinely worth reading in full (the original papers, the Hugging Face docs for a specific library), this guide links out rather than re-explaining it worse.

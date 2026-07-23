# Advanced / where to go next

[← back to index](README.md)

Everything above this point covers the core pipeline. This module is a map of the wider ecosystem for when you outgrow "build it by hand": short, neutral overviews with links to the canonical source for each, rather than a full guide of its own (these tools are maintained elsewhere and change fast; this is a compass, not a manual).

## Training frameworks

Once you're not trying to learn the training loop itself, hand-rolling it stops being the point. Common options, roughly in order of how much they abstract away:

- **Raw PyTorch**: full control, most learning value, most code to maintain yourself.
- **Hugging Face [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer)**: a maintained training loop covering most standard cases (mixed precision, checkpointing, logging, distributed training) with a config-driven interface.
- **[Unsloth](https://github.com/unslothai/unsloth)**: custom Triton kernels (fused RoPE, MLP, attention) and smart sequence packing that can make fine-tuning (LoRA/QLoRA especially) noticeably faster and lower-memory on consumer GPUs. Built on top of the Hugging Face ecosystem, not a replacement for it.
- **[axolotl](https://github.com/axolotl-ai-cloud/axolotl)**: a YAML-config-driven wrapper around the Hugging Face stack, popular for fine-tuning runs without writing training code at all.

## Parameter-efficient fine-tuning

- **[PEFT](https://github.com/huggingface/peft)** (Hugging Face): LoRA and related adapter methods, the standard way to fine-tune large models on limited hardware. See [fine-tuning.md](fine-tuning.md) for the concept.
- **QLoRA** ([Dettmers et al.](https://arxiv.org/abs/2305.14314)): combines LoRA with a quantized (4-bit) frozen base model, pushing fine-tuning of large models onto even smaller GPUs.

## Alignment: RLHF and DPO

After pretraining and fine-tuning, models are commonly further trained to prefer certain responses over others (more helpful, more honest, better at following instructions), using human or AI-generated preference judgments rather than a single correct label.

- **RLHF** (Reinforcement Learning from Human Feedback): train a reward model on human preference comparisons, then use reinforcement learning (commonly PPO) to optimize the language model against that reward.
- **DPO** (Direct Preference Optimization, [Rafailov et al.](https://arxiv.org/abs/2305.18290)): a simpler alternative that optimizes directly on preference pairs without a separate reward model or RL loop.
- **[TRL](https://github.com/huggingface/trl)** (Hugging Face): the standard library implementing both.

## Quantization

Reducing the numeric precision of a trained model's weights (commonly to 8-bit or 4-bit) to shrink memory footprint and speed up inference, with some accuracy cost. Relevant once you're deploying rather than training. See [GGUF](https://huggingface.co/docs/hub/gguf) (the format used by `llama.cpp`) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) as common starting points.

## Serving / inference

Running a trained model efficiently for real usage is its own discipline (batching, KV-cache management, speculative decoding). Common tools: [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (CPU/GPU inference for GGUF models), [vLLM](https://github.com/vllm-project/vllm) (high-throughput GPU serving), Hugging Face [`text-generation-inference`](https://github.com/huggingface/text-generation-inference).

## A note on all of the above

None of these tools change the underlying concepts covered in the rest of this guide: they're implementations of the same ideas (a training loop is still a training loop, LoRA is still "freeze most weights, train a small adapter"). The point of learning the concepts from scratch first is that these tools stop being black boxes; when one of them behaves unexpectedly, you have a mental model to debug against instead of just a config file to guess at.

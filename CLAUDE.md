# CLAUDE.md — AI Assistant Rules for slm-from-scratch

> This file tells Claude (and any coding AI) how to help with this project without stealing your learning.

## What This Project Is

Training small language models (75M→500M) from scratch on a single RTX 5070 Ti. The goal is to understand every step: transformers, tokenizers, data engineering, scaling laws, distillation, evaluation.

**Learning > speed.** If the AI does the hard part, you learned nothing.

---

## What Claude CAN Help With

These are repetitive, mechanical, or infrastructure tasks. Claude can write the code.

- **Boilerplate & plumbing:** Training loop scaffolding, config file parsing, argparse, logging setup, W&B integration
- **Data loading:** DataLoader, dataset classes, batching, shuffling — standard PyTorch patterns
- **Test infrastructure:** Writing unit tests for your code (after you've written the code)
- **Debugging:** "Here's my error traceback, what's wrong?" — Claude can diagnose
- **Code review:** "Here's my implementation of attention, does it look right?" — Claude can review
- **Visualization:** Plotting loss curves, scaling law charts, W&B dashboard setup
- **Environment setup:** pip installs, CUDA verification, dependency resolution
- **Documentation:** README updates, docstrings (after you explain what the code does)
- **Math explanations:** "Explain the Chinchilla scaling law formula" — Claude can teach

## What Claude MUST NOT Do

These are the learning objectives. If Claude writes this code, the project is pointless.

- ❌ **Write the transformer model** — `model.py` (CausalSelfAttention, MLP, TransformerBlock, GPT class) is yours. Claude can explain the concepts, review your code, but you write it.
- ❌ **Implement the training loop core** — The `train_step()`, loss computation, gradient accumulation logic. Claude can set up the surrounding infrastructure.
- ❌ **Train the tokenizer** — You run the HF tokenizers library, you understand BPE, you interpret tokenizer behavior.
- ❌ **Scaling law analysis** — You plot the curves, you identify undertrained/overtrained regimes, you write the conclusions. Claude can help with matplotlib syntax.
- ❌ **Distillation loss implementation** — The KL divergence, temperature scaling, hybrid loss. You write it. Claude can explain the math.
- ❌ **Interpret results** — "What does this loss curve mean?" → you answer that. Claude doesn't get to tell you what you learned.

## How Claude Should Interact

1. **Explain, don't implement.** When you ask "how does multi-head attention work?", Claude explains the concept, shows pseudocode, points to the paper — but doesn't write the implementation unless you explicitly say "ok now write the code for me."

2. **Review, don't rewrite.** When you share code, Claude says "this looks right, but line 42 has an off-by-one" — not "here's the corrected file."

3. **Ask, don't assume.** If you ask for help with something in the ❌ list, Claude should ask: "Are you sure you want me to implement this? This is a core learning objective."

4. **Point to resources.** Claude should reference nanoGPT, the "Attention Is All You Need" paper, Chinchilla paper, relevant sections — teach you to fish.

## Learning Style Preferences

These preferences apply to all explanations, visualizations, and docs in this project.

- **Visual HTML docs over walls of text.** When explaining a concept that warrants more than a few paragraphs, build it as an HTML page in `docs/`. Dark theme, sidebar navigation, visual components (bar charts, grids, pipelines) — not a markdown wall.
- **Docs-style layout with sidebar.** Any multi-section explainer should have a fixed sidebar with nav links and scroll-based active highlighting so sections are easy to jump between.
- **Analogies first, math last.** Lead with a real-world analogy before any technical explanation. Formulas go in collapsible spoilers (a small subtle `∑` symbol trigger) — visible if wanted, ignorable if not.
- **Q&A goes into the doc, not the chat.** When a question is asked about a concept that already has an explainer page, add the answer as a new section in that page rather than just replying in chat. The doc becomes the living reference.
- **No-code explanations for conceptual topics.** When the question is "how does X work", the answer should have zero code unless specifically asked. Pseudocode or plain English only.
- **Examples over definitions.** Show what something does before defining what it is. "The Eiffel Tower is 330m tall" → vector, not "a vector is a mathematical construct..."
- **No em dashes.** Do not use — anywhere. Rewrite the sentence instead.
- **No patronising headings.** Headings should be plain and descriptive. Never write headings like "You're Ready to Write Code!" or "The Magic of Attention" or anything that talks down to the reader. Assume the reader is smart.

## Project Connections

This project connects to:
- `dpo-from-scratch` — DPO will align models trained here
- `llm-eval-arena` — The arena will evaluate models trained here
- `domain-llm` — Domain models may start from checkpoints trained here

## Important Files

- `docs/plan.md` — Full 2-month roadmap. Read this first.
- Future: `src/model.py` — **You write this.** The transformer implementation.
- Future: `src/trainer.py` — Training loop. Claude helps with boilerplate, you write the core.
- Future: `experiments/` — Scaling law experiment results. You analyze, Claude plots.

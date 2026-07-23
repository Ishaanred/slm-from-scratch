# Distillation

[← back to index](README.md)

## The idea, in one analogy

If you wanted to teach a student a subject, you could just give them the textbook's final answers ("the answer to problem 4 is 12"). Or you could show them the expert's full reasoning and confidence: "12 is almost certainly right, though 11 was a plausible near-miss and 13 is clearly wrong." The second version carries far more information per example. Knowledge distillation is training a small "student" model on a large "teacher" model's full output distribution, rather than just the single correct answer.

## Hard labels vs. soft labels

**Hard-label distillation**: the student is trained on the teacher's single best output (e.g., the teacher generates an answer, and the student trains on that generated text as if it were ground truth). Simple, and works with any teacher you can only query as a black box.

**Soft-label distillation**: the student is trained to match the teacher's full probability distribution over possible next tokens, not just its top pick, usually via KL divergence between the two distributions. This carries much more signal (the relative confidence across all the runner-up tokens), but requires access to the teacher's actual output logits, not just its generated text, which rules out any teacher you can only access through a text-only API.

<details><summary>∑ the loss function</summary>

Soft-label distillation typically minimizes a hybrid loss:

```
L = α · CE(student, true_label) + (1 − α) · KL(student_dist || teacher_dist)
```

Temperature scaling is usually applied before the softmax on both distributions (dividing logits by T > 1) to soften the teacher's distribution. A very confident teacher assigns almost all probability to one token, which carries little information about the runner-ups; a higher temperature spreads that out so the student can actually learn from the relative ranking of the alternatives.
</details>

## The constraint nobody warns you about: vocabulary compatibility

Soft-label distillation requires the teacher and student to share the same tokenizer, or at least a way to align token boundaries. Otherwise the teacher's probability distribution is over a different vocabulary than the student's output layer, and there's no direct way to compute a KL divergence between them. If your student uses its own custom tokenizer (see [tokenization.md](tokenization.md)) and the teacher uses a different one, soft-label distillation is off the table until that mismatch is resolved (either by using the teacher's tokenizer for the student too, or by a token-alignment scheme). This is a real, checkable constraint worth verifying before designing a distillation pipeline around it, not an assumption to discover mid-project.

## The other real cost: generating teacher data isn't free

Whether you're doing hard-label or soft-label distillation, you generally need the teacher to actually run inference to produce training signal, and autoregressive generation is decode-bound (one token at a time), which is much slower than a single forward pass over an existing sequence. On modest hardware, generating enough teacher outputs to meaningfully train a student can be the actual bottleneck of the whole project, more so than the student's training step itself. Worth measuring teacher throughput directly before committing to a distillation plan's scope.

## Where to look further

- Hinton et al., ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531): the original distillation paper, including temperature scaling.
- Hugging Face's [distillation examples](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation): a maintained reference implementation.
- This project's own distillation investigation, including the vocabulary-mismatch finding above: [`slm-from-scratch` Phase 4](../docs/phase4/distillation.html).

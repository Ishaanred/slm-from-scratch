# Evaluation

[← back to index](README.md)

## The idea, in one analogy

Reading a few of a model's outputs and going "yeah, that seems pretty good" is like judging a student's understanding of a whole subject from one homework answer. It might be representative, or you might have gotten lucky (or unlucky). Evaluation is giving the model a standardized test with a known answer key, at a scale where the score actually means something.

## What a benchmark actually measures

Most standard LLM benchmarks (HellaSwag, PIQA, ARC, MMLU, and similar) are multiple-choice: given a question or context, the model scores each candidate answer by how likely it finds that continuation, and "correctness" is whether the highest-scored option matches the labeled answer. This is convenient (fully automatic, no human judge needed) but it means the benchmark measures the specific skill it was designed around, not general capability. A model can genuinely be good at one benchmark's style of question and mediocre at a different but related skill.

## The number you have to know before you can interpret any score: the chance floor

A 4-choice multiple-choice benchmark has a 25% random-guessing floor. A model scoring 27% on that benchmark is barely above random chance, even though "27%" sounds like a real, non-zero score in isolation. Always check a benchmark's chance floor before reacting to a raw percentage: a small model scoring near the floor across the board isn't necessarily broken, it may simply not have had enough training scale yet to reliably beat pure guessing on that task.

## What a benchmark score doesn't tell you: error analysis

An aggregate score answers "how often is the model right," not "why is it wrong when it's wrong." Actually reading a sample of the wrong answers surfaces things a single number can't: whether the model is missing genuine reasoning, whether it's failing on questions that use vocabulary the tokenizer badly fragments (see [tokenization.md](tokenization.md)), or whether the wrong answers cluster around one specific sub-category rather than being spread evenly. This is manual, unglamorous work, and it's usually the difference between "the model scored X%" and actually understanding what to do about it.

## Deciding whether to iterate, honestly

If error analysis shows broad, low performance spread evenly across every category, that usually means the model's precondition for targeted improvement (partial competence with a few specific, fixable gaps) isn't met yet: no amount of generating more data for one narrow weak spot fixes a model that's weak everywhere. If instead most categories are solid and one or two are conspicuously weak, that's the situation where generating targeted training data for the weak spot and retraining is actually likely to help. Deciding which situation you're in, ideally by setting the deciding rule *before* looking at the results, not after, keeps you from rationalizing a decision to keep iterating (or to stop) after the fact.

## Where to look further

- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness): the standard, widely-used tool for running these benchmarks against any model, including custom (non-Hugging-Face) architectures via a small adapter.
- The [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard): for context on what scores are considered strong at various model sizes.
- This project's own benchmark run, chance-floor framing, and error analysis: [`slm-from-scratch` Phase 5](../docs/phase5/evaluation.html).

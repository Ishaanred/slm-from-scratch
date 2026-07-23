# Scaling laws

[← back to index](README.md)

## The idea, in one analogy

If you had a fixed number of study hours before an exam, you'd want to know: is it better to read one book twice, or two books once each? Scaling laws answer the equivalent question for training a model: given a fixed compute budget, is it better to train a bigger model on less data, or a smaller model on more data? The answer isn't obvious, and it turns out to follow a fairly predictable curve.

## What the curve actually says

Loss decreases smoothly and predictably as you increase model size, dataset size, or compute, each following its own power-law relationship, as long as the other two aren't the bottleneck. The practically useful version of this is the Chinchilla result (Hoffmann et al., 2022): for a fixed compute budget, there's a compute-optimal ratio of model size to dataset size, and most models trained before that paper were meaningfully oversized relative to how much data they saw; a smaller model trained on more tokens would have reached lower loss for the same compute.

A useful rule of thumb from that paper: roughly 20 tokens of training data per model parameter is close to compute-optimal. A 1B-parameter model wants roughly 20B tokens; far less than that, and the model is undertrained relative to its size, with capacity going to waste. Far more than that, and you've spent compute a smaller model could have used more efficiently.

## How to actually use this

You don't need access to a supercomputer to get a real scaling-law result, you need a controlled comparison. Fix everything except the one variable you're testing (same architecture family, same data, same training recipe), vary model size or token count deliberately, and plot validation loss against tokens seen and against total training compute (FLOPs). Two comparisons are usually enough to see the pattern:

- **Same model, more tokens**: does giving a fixed-size model more data keep helping, or does it plateau?
- **Same tokens, more parameters**: does a bigger model help when the data budget is scarce, or is it just more undertrained?

Real experiments at small scale won't reproduce the literature's exact numbers (they don't have the compute), but the *shape* of the tradeoff, and whether your own setup follows it or deviates from it, is a genuine, checkable result, not just a toy exercise.

## A trap worth naming: noisy endpoints

Validation loss near the end of a cosine-decay training run can be surprisingly noisy: a run's loss can jump around by a large margin between adjacent evaluation points before settling. Comparing runs using whichever single point happens to be lowest, or whichever happens to be the literal final step, can quietly bias a comparison. Deciding in advance which number you'll report (final step vs. best checkpoint), and reporting both if they disagree, keeps this honest.

## Where to look further

- Kaplan et al., ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361): the original scaling-law paper.
- Hoffmann et al., ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556) (the Chinchilla paper): the compute-optimal ratio result.
- This project's own small-scale replication: [`slm-from-scratch` Phase 3](../docs/phase3/scaling-laws.html).

# Zarra — product case study

Stage: independent prototype.

## Product contribution

A reproducible local BERT/PEFT intent-classification experiment comparing a rule baseline, an unadapted classifier, and a LoRA adapter on a synthetic held-out split.

Original domain framing, workflow and interface design, implementation with AI assistance, input validation, tests, and source-linked documentation. Foundation: https://github.com/huggingface/peft.

## Verification

Actual local LoRA training and inference. The repository includes the relevant fixtures, tests, and retained reports.

## Tradeoffs and limits

The experiment uses 30 synthetic training examples and 12 synthetic held-out examples. Both rules and the adapter scored 100% on this tiny set. This establishes no benefit over rules and no customer generalization. The unadapted classifier head is newly initialized.

## Next product test

Collect a more representative evaluation set, include ambiguous and out-of-domain requests, and test whether adaptation improves errors that the baseline actually makes.

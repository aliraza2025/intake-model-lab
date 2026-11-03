# Zarra

**Small-model adaptation experiments · Raza Labs by Ali Raza**

![Concept artwork for Zarra](public/project-cover.webp)

A reproducible local BERT/PEFT intent-classification experiment comparing a rule baseline, an unadapted classifier, and a LoRA adapter on a synthetic held-out split.

**[Working browser demo](https://aliraza2025.github.io/intake-model-lab/)** · [Case study](CASE_STUDY.md) · [Portfolio copy](PORTFOLIO.md)

Stage: independent working prototype by Ali Raza, with AI-assisted implementation.

## What is verified

Actual local LoRA training and inference. Tests and original synthetic fixtures are included. Public browser examples are labeled separately from local framework/model execution.

## Run the workbench

Requires Node.js 22 or newer. Python integrations were verified with Python 3.13 on Apple Silicon.

```bash
# No npm dependencies are required for the browser baseline.
npm test
npm run build
npm start
```

Run `.venv/bin/python scripts/train.py` to reproduce the synthetic LoRA experiment. Then choose local inference in the app. The training script writes an adapter under `outputs/adapter/`, an actual report in `examples/`, and the public benchmark module. Base weights and local checkpoints are ignored by Git.

## Optional local Python integration

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON_BIN="$PWD/.venv/bin/python" npm start
```

The first model run downloads public weights to `.model-cache/`. No paid provider is configured. Model licenses are separate from this repository's MIT source-code license. Only use recordings and documents you have permission to process.

## Boundaries

The experiment uses 30 synthetic training examples and 12 synthetic held-out examples. Both rules and the adapter scored 100% on this tiny set. This establishes no benefit over rules and no customer generalization. The unadapted classifier head is newly initialized.

The records, prices, and stories supplied with this project are synthetic. No customers, adoption, revenue, business savings, regulatory certification, or production reliability are claimed.

## Foundation and original contribution

Foundation: https://github.com/huggingface/peft. The original contribution is the domain workflow, interface, validation, synthetic fixtures, tests, and product reasoning in this repository. Dependencies retain their licenses and are not represented as authored here.

The original browser source is in `public/`. `docs/` is generated for GitHub Pages; run `npm run build` after changing the browser source. Local services bind to loopback. Model endpoints are not exposed through GitHub Pages.

## Next validation

Collect a more representative evaluation set, include ambiguous and out-of-domain requests, and test whether adaptation improves errors that the baseline actually makes.

License: [MIT](LICENSE) for this repository's original source. See each dependency and model card for their respective terms.

## Visual identity

Original AI-generated concept artwork for the Raza Labs collection, matched to Ali Raza’s midnight-purple and pink portfolio studio. The cover is illustrative artwork; the interactive demo shows the actual prototype. Generation prompt: [ARTWORK.md](ARTWORK.md).

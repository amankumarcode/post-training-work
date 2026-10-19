# slm-post-training

Post-training experiments for **small language models (SLMs)**: SFT, preference optimization, RL, distillation, merging, tool use, and quantization/serving. Each method is a self-contained experiment built on shared code, so results stay comparable.

> Status: scaffolding only. Code, configs, and results will be added experiment by experiment.

## Goal

Take one small base model and measure what each post-training technique actually does to it, using the same evaluation for every experiment.

- **Base model:** _TBD_
- **Evaluation:** one shared entry point (`src/slmpt/eval/run_eval.py`) used by every experiment
- **Results:** one row per experiment in `results/leaderboard.md`

## Planned structure

```
slm-post-training/
├── src/slmpt/       # shared code: data, models, rewards, eval, utils
├── experiments/     # one folder per method (00-baseline … 12-quantize-serve)
├── data/            # dataset build scripts + pointers to the Hub
├── results/         # leaderboard + plots
├── scripts/         # cloud GPU / sync helpers
├── notebooks/       # exploration only
└── tests/           # templates, masking, reward function checks
```

### Experiments

| #  | Experiment          | Notes                                   |
|----|---------------------|-----------------------------------------|
| 00 | baseline            | evaluate the untouched base model first |
| 01 | sft-full            | full fine-tune                          |
| 02 | sft-lora            |                                         |
| 03 | sft-qlora           |                                         |
| 04 | dpo                 | starts from the best SFT checkpoint     |
| 05 | simpo-orpo-kto      | DPO variants compared side by side      |
| 06 | reward-model        |                                         |
| 07 | ppo                 | classic RLHF using the reward model     |
| 08 | grpo                | RL with verifiable rewards              |
| 09 | distillation        | big teacher → SLM                       |
| 10 | merging             | TIES, DARE, SLERP across checkpoints    |
| 11 | tool-use            | function-calling fine-tune              |
| 12 | quantize-serve      | GGUF / AWQ + vLLM or Ollama             |

Every experiment folder uses the same layout: `README.md`, `config.yaml` (all hyperparameters), a thin `train.py` that calls `src/slmpt`, `eval.sh`, and `results/` (`metrics.json`, `loss_curve.png`).

## Results

_No results yet._

## How to run

_Coming soon._

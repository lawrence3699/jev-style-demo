---
title: Jev-Style v3 Demo
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Calibrated yes/no, choice and score answers
models:
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3
preload_from_hub:
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3 LICENSE,NOTICE,chat_template.jinja,config.json,generation_config.json,jev_style_decision.py,manifest.json,model.safetensors,readout_config.json,release_config.json,requirements.txt,tokenizer.json,tokenizer_config.json 901b5b9fdf225443fbe2eba4e7a47b41984191d8
---

# Jev-Style v3 Demo

[Jev-Style 0.8B Decision v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) reads a state
(text or JSON; inputs of up to 25,600 tokens) and answers typed questions about it: yes/no, one of several
options, or a level on a scale. It returns a calibrated probability for every option and never writes text. In
each tab the **action** line is plain code on those probabilities, with the rule shown under it.

Demo videos: [jevstyle.com](https://jevstyle.com). Other formats of the same model:
[GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF),
[MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX). Run it on your own machine: `pip install "jev-style[torch]"` (`[mlx]` on Apple silicon)
([GitHub](https://github.com/lawrence3699/jev-style)).

| Tab | What it does |
|---|---|
| Triage | Queue, urgency, refund and escalation for a support message; the account tier is part of the state |
| Email | Phishing check, category and reply-needed; quarantine / flag / inbox |
| Guardrails | Jailbreak, injection and harmful-request checks for an incoming prompt; pass / review / block |
| RAG filter | Relevance of each retrieved passage to the query, ranked, with a keep threshold |
| Moderation | Toxicity, harassment, spam and severity; publish / moderator queue / remove |
| Routing | Difficulty, reasoning need and domain; send the request to a cheap or a strong model |
| 51 languages | One routing question on a message in any language (the model card reports MASSIVE results in 51 languages, 19 in fine-tuning) |
| Long document | Up to 25,600 tokens in one input, nothing truncated; example: a 19K-token public-domain bundle, with reference answers |
| Agent approval | Should a coding agent run this shell command? allow / ask / deny (the command is never run) |
| Playground | Any state and up to 8 typed questions, in the request shape shown under Raw JSON |

Every tab is one `@spaces.GPU` call. Inputs are checked and tokenised on CPU first, so oversized input is
refused before any GPU time is requested. The model is the repo's own PyTorch runtime (`jev_style_decision.py`,
float32) at commit `901b5b9fdf225443fbe2eba4e7a47b41984191d8`; the manifest is checked at start-up.

## Speed

Measured locally with this app on an Apple M1 Max (PyTorch 2.13 on MPS, float32), second call of each tab,
end to end through `gradio_client`. ZeroGPU times are not measured yet; the Space log prints them.

| Tab | Questions | Input tokens | Local time |
|---|---|---|---|
| Triage | 4 | 412 | 0.9 s |
| Email | 3 | 369 | 1.0 s |
| Guardrails | 3 | 274 | 0.7 s |
| RAG filter | 5 passages | 563 | 1.0 s |
| Moderation | 4 | 317 | 0.8 s |
| Routing | 3 | 345 | 0.7 s |
| 51 languages | 1 | 170 | 0.3 s |
| Long document | 5 | 19,424 | 225 s (one call) |
| Agent approval | 5 | 810 | 2.0 s |
| Playground | 4 | 405 | 1.2 s |

This PyTorch runtime reads the state once per question, so the long document costs five 19K-token passes (the
GGUF runtime can share the state between questions). The Gated DeltaNet layers use transformers' reference
PyTorch path: `flash-linear-attention` and `causal-conv1d` are not installed. Both are optional; leaving them
out keeps the install plain pip (no CUDA build) and runs the same code path as the local tests, at a cost in
speed (transformers warns that the reference path is much slower).

## Limits

- One model, 0.8B parameters, for routing, classification and yes/no checks on a given state; numbers that need
  computing are best computed in code and put in the state.
- Calibration temperatures were fitted on held-out rows; the thresholds in each tab are starting points, not
  tuned values.
- Training data, evaluation and licence notes: see the
  [model card](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3).

## Files and licences

- `app.py` (Gradio UI, model loading, GPU calls), `engine.py` (request/answer shapes), `tabs.py` (questions,
  rules, examples), `macjev_guard.py` and `guard_config.json` (unchanged copies of the project's Claude Code
  guard, used by the Agent approval tab with the model called in-process), and the questions and reference
  answers in `examples/long_document.json`: Apache-2.0 (see `LICENSE`).
- The document in `examples/long_document.json`: the Declaration of Independence, the U.S. Constitution, the
  Bill of Rights and Federalist Nos. 10, 51 and 78, public domain (Project Gutenberg eBooks 1, 5, 2 and 1404,
  headers removed).
- Model: Apache-2.0, fine-tuned from Qwen/Qwen3.5-0.8B (Apache-2.0); see the card for training-data terms. The
  question types (choice / score / yes-no) follow Laya's typed-decision convention, as the model's NOTICE says.

Not affiliated with TypeSafe, Jev or Laya.

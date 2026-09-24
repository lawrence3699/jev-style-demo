"""systemone-style requests on top of the model repo's own PyTorch runtime.

The Space loads ``jev_style_decision.JevStyleDecision`` from chaoliangUNSW/Jev-Style-0.8B-Decision-v3
(rendering, verdict readout, calibration temperatures and token budgets all live there). This module only
translates the request shape used by the local server's ``POST /v1/systemone`` into the runtime's
``decide_many(state, questions)`` call and back, so every tab (and the copied guard) sees the same answers
the HTTP API gives:

    request   {"state": str | object | array,
               "questions": {id: {"type": "noul" | "choice" | "score", "instructions": str, "criteria": ...}}}
    response  {"model", "answers": {id: answer}, "usage": {...}, "timing": {...}, "backend"}

    noul    {"type": "noul", "noul": P(true)}
    choice  {"type": "choice", "choice": best option, "probabilities": {option: p}, "confidence": c}
    score   {"type": "score", "score": sum_i i * p_i, "legend": {"0": label, ...},
             "probabilities": {"0": p_0, ...}, "confidence": c}

confidence = (k * p_max - 1) / (k - 1) for k options (0 = uniform, 1 = all mass on one option).
Calibration: the fitted "typed_official" group temperatures, the category the local server uses for API
questions. No torch import here; the runtime object does the work.
"""
from __future__ import annotations

import json
import time
from typing import Any

MODEL_ID = "jev-style-0.8b-decision-v3"
DEFAULT_CATEGORY = "typed_official"
QTYPES = ("noul", "choice", "score")


class RequestError(ValueError):
    """The request is malformed or over a token budget (nothing is truncated)."""


def _text(value: Any, what: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise RequestError(f"{what} must not be empty")
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise RequestError(f"{what} must be a string, object or array")


def _level(level: Any) -> tuple[str, str]:
    """(label shown in the legend, text given to the model) for one score level."""
    if isinstance(level, str) and level.strip():
        return level, level
    if isinstance(level, dict) and isinstance(level.get("label"), str) and level["label"].strip():
        desc = level.get("description")
        return level["label"], f"{level['label']}: {desc}" if desc else level["label"]
    raise RequestError("score levels must be non-empty strings or {\"label\", \"description\"} objects")


def to_internal(qid: str, q: Any) -> tuple[dict, dict]:
    """systemone question -> (runtime question {"t", "ins", "crit"}, info for the answer)."""
    if not isinstance(q, dict):
        raise RequestError(f"question {qid!r} must be an object")
    t = q.get("type")
    if t not in QTYPES:
        raise RequestError(f"question {qid!r}: type must be one of {QTYPES}")
    ins = _text(q.get("instructions"), f"question {qid!r} instructions")
    crit = q.get("criteria")
    if t == "noul":
        if crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
            raise RequestError(f"question {qid!r}: noul criteria must be null or {{\"true\": ..., \"false\": ...}}")
        return {"t": "noul", "ins": ins, "crit": crit or None}, {}
    if t == "choice":
        if isinstance(crit, list) and crit and all(isinstance(c, str) for c in crit):
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or not crit or len(crit) > 255:
            raise RequestError(f"question {qid!r}: choice criteria must map 1..255 options to descriptions")
        opts = {str(k): (v if v is None or isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                for k, v in crit.items()}
        return {"t": "choice", "ins": ins, "crit": opts}, {"options": list(opts)}
    if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
        raise RequestError(f"question {qid!r}: score criteria must be a list of 2..10 levels")
    levels = [_level(c) for c in crit]
    return {"t": "score", "ins": ins, "crit": [text for _, text in levels]}, {"labels": [lab for lab, _ in levels]}


def confidence(probs: list[float]) -> float:
    k = len(probs)
    if k <= 1:
        return 1.0
    return float(min(1.0, max(0.0, (k * max(probs) - 1.0) / (k - 1.0))))


def parse(body: Any) -> tuple[Any, list[str], list[dict], list[dict]]:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    if "state" not in body:
        raise RequestError("state is required")
    state = body["state"]
    if not isinstance(state, (str, dict, list)):
        raise RequestError("state must be a string, object or array")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise RequestError("questions must be a non-empty object mapping id -> question")
    ids, internal, info = [], [], []
    for qid, q in qs.items():
        iq, extra = to_internal(str(qid), q)
        ids.append(str(qid))
        internal.append(iq)
        info.append(extra)
    return state, ids, internal, info


class SystemOneAdapter:
    """``adapter(body) -> response``; ``runtime`` is a JevStyleDecision, ``rt_module`` its module."""

    def __init__(self, runtime: Any, rt_module: Any, category: str = DEFAULT_CATEGORY, model_id: str = MODEL_ID):
        self.runtime = runtime
        self.rt = rt_module
        self.category = category
        self.model_id = model_id

    @property
    def backend(self) -> str:
        return f"torch-{getattr(self.runtime, 'device', 'cpu')}"

    def __call__(self, body: Any) -> dict:
        t0 = time.perf_counter()
        state, ids, internal, info = parse(body)
        try:
            results = self.runtime.decide_many(state, internal, category=self.category)
        except (self.rt.InputBudgetError, self.rt.QuestionError) as e:
            raise RequestError(str(e)) from None
        total_ms = (time.perf_counter() - t0) * 1000
        answers = {}
        for qid, iq, extra, res in zip(ids, internal, info, results):
            answers[qid] = self._answer(iq, extra, res)
        state_tokens = results[0]["input_tokens"] - results[0]["head_tokens"]
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": int(state_tokens + sum(r["head_tokens"] for r in results)),
                          "state_tokens": int(state_tokens), "output_tokens": 0,
                          "tokens_scored": int(sum(r["input_tokens"] for r in results))},
                "timing": {"total_ms": round(total_ms, 1), "forward_passes": len(results)},
                "backend": self.backend}

    @staticmethod
    def _answer(iq: dict, extra: dict, res: dict) -> dict:
        probs = res["probabilities"]
        if iq["t"] == "noul":
            return {"type": "noul", "noul": float(probs["true"])}
        if iq["t"] == "choice":
            names = extra["options"]
            p = [float(probs[n]) for n in names]
            best = max(range(len(p)), key=p.__getitem__)
            return {"type": "choice", "choice": names[best], "probabilities": dict(zip(names, p)),
                    "confidence": confidence(p)}
        k = len(extra["labels"])
        p = [float(probs[str(i)]) for i in range(k)]
        return {"type": "score", "score": float(sum(i * v for i, v in enumerate(p))),
                "legend": {str(i): lab for i, lab in enumerate(extra["labels"])},
                "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": confidence(p)}

"""Jev-Style v3 demo: a Gradio Space for chaoliangUNSW/Jev-Style-0.8B-Decision-v3 on ZeroGPU.

The model repo is downloaded at start-up at a pinned commit and scored with the repo's own PyTorch runtime
(jev_style_decision.JevStyleDecision: rendering, verdict readout, calibration temperatures, token budgets).
Each tab makes exactly one @spaces.GPU call; inside it the runtime's decide_many scores every question of
the tab against the same state. Actions are plain code in tabs.py.

Before any GPU time is asked for, every request is validated and rendered on CPU (tokenizer only), so malformed
or oversized input is refused without touching the visitor's ZeroGPU quota, and the GPU request is sized from
the real token count.

Local run (spaces.GPU does nothing off Hugging Face): `python app.py` picks CUDA, then Apple MPS, then CPU.
Environment: JEV_DEVICE=cuda|mps|cpu forces a device; JEV_MODEL_DIR=<folder> uses a local copy of the
model repo instead of downloading it.
"""
# `spaces` must be imported before torch: on ZeroGPU it patches torch's CUDA handling.
try:
    import spaces

    GPU = spaces.GPU
except ImportError:                               # plain local run without the package
    def GPU(fn=None, **_kwargs):
        return fn if callable(fn) else (lambda f: f)

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

import engine  # noqa: E402
import macjev_guard as guard  # noqa: E402  (copy of integrations/claude-code-guard/macjev_guard.py)
import tabs  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_REPO = "chaoliangUNSW/Jev-Style-0.8B-Decision-v3"
MODEL_REVISION = "901b5b9fdf225443fbe2eba4e7a47b41984191d8"      # pinned commit of the model repo
# exactly the files preloaded by README.md's preload_from_hub (same commit), so start-up needs no network
MODEL_FILES = ["LICENSE", "NOTICE", "chat_template.jinja", "config.json", "generation_config.json",
               "jev_style_decision.py", "manifest.json", "model.safetensors", "readout_config.json",
               "release_config.json", "requirements.txt", "tokenizer.json", "tokenizer_config.json"]
GGUF_REPO = "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF"
MLX_REPO = "chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX"
SITE = "https://jevstyle.com"
ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")

# GPU time. spaces multiplies every request by 1.5 on the current ZeroGPU hardware before the quota check, so
# MAX_GPU_S = 80 is checked as 120 s: an anonymous visitor's whole daily quota. SEC_PER_TOKEN is measured on the
# first ZeroGPU run (2026-09-25: 95,452 tokens in 19,940 ms = 0.00021 s/token on cuda:0), with ~1.4x margin.
SEC_PER_TOKEN = 0.0003
SHORT_BASE_S, SIZED_BASE_S, MAX_GPU_S = 8, 10, 80
SHORT_TOKEN_BUDGET = 16_000                         # tokens scored per call (state x questions) in the short tabs
SIZED_TOKEN_BUDGET = 5 * 25_600                     # long document and playground: five full-length passes


def pick_device() -> str:
    forced = os.environ.get("JEV_DEVICE", "").strip()
    if forced:
        return forced
    if ON_ZEROGPU or torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def load_engine():
    model_dir = os.environ.get("JEV_MODEL_DIR") or snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)
    sys.path.insert(0, str(model_dir))
    import jev_style_decision as rt                 # the model repo's runtime, same commit as the weights
    t0 = time.perf_counter()
    runtime = rt.JevStyleDecision(model_dir, device=DEVICE, dtype="float32", verify=True)
    print(f"Jev-Style v3 loaded on {DEVICE} in {time.perf_counter() - t0:.1f} s "
          f"({MODEL_REPO}@{MODEL_REVISION[:7]}, float32, manifest verified)", flush=True)
    return engine.SystemOneAdapter(runtime, rt)


ENGINE = load_engine()                              # on ZeroGPU the weights sit on "cuda" from here on


def _run(body):
    """Score one request, or a list of requests in the same GPU call. Errors come back as {"error": ...}
    so they cross the ZeroGPU process boundary."""
    rt = ENGINE.runtime
    dev = next(rt.model.parameters()).device
    if rt.direction.device != dev:                  # keep the readout vector next to the weights
        rt.direction = rt.direction.to(dev)
    t0 = time.perf_counter()
    try:
        out = [ENGINE(b) for b in body] if isinstance(body, list) else ENGINE(body)
    except engine.RequestError as e:
        return {"error": str(e)}
    outs = out if isinstance(out, list) else [out]
    print(f"scored {sum(len(o['answers']) for o in outs)} questions, "
          f"{sum(o['usage']['tokens_scored'] for o in outs):,} tokens in {(time.perf_counter() - t0) * 1000:,.0f} ms "
          f"on {dev}", flush=True)                   # the Space log shows real GPU times for tuning durations
    return out


# -- CPU pre-check and GPU sizing -----------------------------------------------------------------------------
_plan = threading.local()                           # (id(body), tokens) from the pre-check, for the duration fn


def tokens_scored(body) -> int:
    """Validate and render every request on CPU (the runtime's own renderer and budget checks) and return the
    total number of tokens the model would score. Raises ValueError with the runtime's message."""
    rt, renderer = ENGINE.rt, ENGINE.runtime.renderer
    total = 0
    for b in body if isinstance(body, list) else [body]:
        state, _ids, internal, _info = engine.parse(b)
        try:
            for q in internal:
                total += len(renderer.render(state, rt.make_question(q)).ids)
        except (rt.InputBudgetError, rt.QuestionError) as e:
            raise ValueError(str(e)) from None
    return total


def _planned(body) -> int:
    got = getattr(_plan, "value", None)
    if got and got[0] == id(body):
        return got[1]
    try:
        return tokens_scored(body)
    except Exception:                               # the pre-check already refused bad input; never fail here
        return 0


def gpu_seconds(tokens: int, base: int) -> int:
    return int(min(MAX_GPU_S, math.ceil(base + tokens * SEC_PER_TOKEN)))


def short_duration(body) -> int:
    return gpu_seconds(_planned(body), SHORT_BASE_S)


def sized_duration(body) -> int:
    """The runtime reads the state once per question, so the time grows with state tokens x questions. The
    19K-token example with 5 questions scores about 95K tokens (225 s locally on Apple MPS)."""
    return gpu_seconds(_planned(body), SIZED_BASE_S)


@GPU(duration=short_duration)
def gpu_short(body):
    """Tabs with fixed, short questions (RAG sends its passages as a list of requests in this one call)."""
    return _run(body)


@GPU(duration=sized_duration)
def gpu_sized(body: dict) -> dict:
    return _run(body)


def checked(gpu_fn, budget: int):
    def decide(body):
        n = tokens_scored(body)
        if n > budget:
            raise ValueError(f"this needs {n:,} tokens of model input in total (state x questions); this tab "
                             f"allows {budget:,}. Nothing was truncated: shorten the input or ask fewer questions.")
        _plan.value = (id(body), n)
        try:
            out = gpu_fn(body)
        finally:
            _plan.value = None
        if isinstance(out, dict) and "error" in out:
            raise ValueError(out["error"])
        return out
    return decide


SHORT, SIZED = checked(gpu_short, SHORT_TOKEN_BUDGET), checked(gpu_sized, SIZED_TOKEN_BUDGET)


def handler(fn, decide, *extra):
    """Wrap a tabs.py function as a Gradio callback: (table, action, latency, raw JSON)."""
    def run(*args):
        try:
            res = fn(decide, *args, *extra)
        except (ValueError, KeyError, RecursionError) as e:
            note = tabs.quota_note(e)
            return [], note or f"**Error:** {tabs.md(e)}", "", json.dumps({"error": str(e)}, indent=2)
        except Exception as e:                      # ZeroGPU quota refusals arrive as gradio errors
            note = tabs.quota_note(e)
            if note is None:
                raise
            return [], note, "", json.dumps({"error": str(e)}, indent=2)
        return res.rows, res.action, res.detail, json.dumps(res.raw, indent=2, ensure_ascii=False)
    return run


def result_panel(rule: str, columns=tabs.COLUMNS, widths=("24%", "36%", "20%", "20%")):
    action = gr.Markdown(elem_classes="action")
    gr.Markdown(rule, elem_classes="rule")
    table = gr.Dataframe(headers=columns, column_count=len(columns), interactive=False, wrap=True,
                         label="Answers", column_widths=list(widths))
    latency = gr.Markdown(elem_classes="latency")
    with gr.Accordion("Raw JSON", open=False):
        raw = gr.Code(language="json", label="request and response", lines=12, max_lines=30)
    return [table, action, latency, raw]


THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
CSS = """
.rule, .latency { font-size: 0.85rem; opacity: 0.8; }
.action p { font-size: 1.05rem; }
th, th * { word-break: normal !important; overflow-wrap: normal !important; hyphens: none !important; }
.dark .prose a, .dark a { color: #93c5fd; }
"""
HEADER = f"""# Jev-Style v3 Demo
Jev-Style 0.8B Decision v3 answers typed questions about a state (text or JSON) with a calibrated probability for every option. Each tab turns those probabilities into an action with a few lines of code; the rule is shown with it.

[jevstyle.com]({SITE}) (demo videos) · [GitHub](https://github.com/lawrence3699/jev-style) (`pip install jev-style`) · [model]({"https://huggingface.co/" + MODEL_REPO}) · [GGUF]({"https://huggingface.co/" + GGUF_REPO}) · [MLX]({"https://huggingface.co/" + MLX_REPO}) · Not affiliated with TypeSafe, Jev or Laya.
"""

with gr.Blocks(title="Jev-Style v3 Demo", analytics_enabled=False) as demo:
    gr.Markdown(HEADER)

    with gr.Tab("Triage"):
        gr.Markdown("Queue, urgency, refund and escalation for one message.")
        with gr.Row():
            with gr.Column():
                t_msg = gr.Textbox(lines=5, label="Support message", value=tabs.TRIAGE_EXAMPLES[0][0],
                                   max_length=tabs.MAX_TEXT_CHARS)
                t_tier = gr.Radio(["free", "pro", "enterprise"], value=tabs.TRIAGE_EXAMPLES[0][1],
                                  label="Account tier (in the state)")
                t_thr = gr.Slider(0.3, 0.95, 0.6, step=0.05, label="Min. queue confidence")
                t_go = gr.Button("Decide", variant="primary")
                gr.Examples(tabs.TRIAGE_EXAMPLES, [t_msg, t_tier], example_labels=tabs.TRIAGE_LABELS)
            with gr.Column():
                t_out = result_panel(tabs.TRIAGE_RULE)
        t_go.click(handler(tabs.triage, SHORT), [t_msg, t_tier, t_thr], t_out, api_name="triage")

    with gr.Tab("Email"):
        gr.Markdown("Phishing check, category and reply-needed for one email.")
        with gr.Row():
            with gr.Column():
                e_from = gr.Textbox(label="From", value=tabs.EMAIL_EXAMPLES[0][0], max_length=tabs.MAX_TEXT_CHARS)
                e_subj = gr.Textbox(label="Subject", value=tabs.EMAIL_EXAMPLES[0][1], max_length=tabs.MAX_TEXT_CHARS)
                e_body = gr.Textbox(lines=6, label="Body", value=tabs.EMAIL_EXAMPLES[0][2],
                                    max_length=tabs.MAX_TEXT_CHARS)
                e_thr = gr.Slider(0.3, 0.95, 0.8, step=0.05, label="Quarantine above")
                e_go = gr.Button("Decide", variant="primary")
                gr.Examples(tabs.EMAIL_EXAMPLES, [e_from, e_subj, e_body], example_labels=tabs.EMAIL_LABELS)
            with gr.Column():
                e_out = result_panel(tabs.EMAIL_RULE)
        e_go.click(handler(tabs.email, SHORT), [e_from, e_subj, e_body, e_thr], e_out, api_name="email")

    with gr.Tab("Guardrails"):
        gr.Markdown("Jailbreak, injection and harm checks on an incoming prompt.")
        with gr.Row():
            with gr.Column():
                g_in = gr.Textbox(lines=5, label="Prompt to screen", value=tabs.GUARD_EXAMPLES[0][0],
                                  max_length=tabs.MAX_TEXT_CHARS)
                g_thr = gr.Slider(0.3, 0.95, 0.5, step=0.05, label="Block threshold")
                g_go = gr.Button("Decide", variant="primary")
                gr.Examples(tabs.GUARD_EXAMPLES, [g_in])
            with gr.Column():
                g_out = result_panel(tabs.GUARD_RULE)
        g_go.click(handler(tabs.guardrails, SHORT), [g_in, g_thr], g_out, api_name="guardrails")

    with gr.Tab("RAG filter"):
        gr.Markdown("Relevance of each retrieved passage to the query, then a ranked keep list.")
        with gr.Row():
            with gr.Column():
                r_q = gr.Textbox(label="Query", value=tabs.RAG_EXAMPLE_QUERY, max_length=tabs.MAX_TEXT_CHARS)
                r_p = gr.Textbox(lines=10, label=f"Passages, separated by an empty line (max {tabs.MAX_PASSAGES})",
                                 value=tabs.RAG_EXAMPLE_PASSAGES,
                                 max_length=tabs.MAX_PASSAGES * (tabs.MAX_PASSAGE_CHARS + 2))
                r_thr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Keep threshold")
                r_go = gr.Button("Rank", variant="primary")
            with gr.Column():
                r_out = result_panel(tabs.RAG_RULE, tabs.RAG_COLUMNS, ("9%", "8%", "55%", "14%", "14%"))
        r_go.click(handler(tabs.rag, SHORT), [r_q, r_p, r_thr], r_out, api_name="rag")

    with gr.Tab("Moderation"):
        gr.Markdown("Toxicity, harassment, spam and severity for one post.")
        with gr.Row():
            with gr.Column():
                m_in = gr.Textbox(lines=4, label="Post", value=tabs.MOD_EXAMPLES[1][0], max_length=tabs.MAX_TEXT_CHARS)
                m_thr = gr.Slider(0.3, 0.95, 0.8, step=0.05, label="Remove automatically above")
                m_go = gr.Button("Decide", variant="primary")
                gr.Examples(tabs.MOD_EXAMPLES, [m_in])
            with gr.Column():
                m_out = result_panel(tabs.MOD_RULE)
        m_go.click(handler(tabs.moderation, SHORT), [m_in, m_thr], m_out, api_name="moderation")

    with gr.Tab("Routing"):
        gr.Markdown("Pick a cheap or a strong model per request.")
        with gr.Row():
            with gr.Column():
                o_in = gr.Textbox(lines=4, label="Request", value=tabs.ROUTE_EXAMPLES[2][0],
                                  max_length=tabs.MAX_TEXT_CHARS)
                o_cheap = gr.Textbox(label="Model for easy requests", value="fast-small", max_length=tabs.MAX_NAME_CHARS)
                o_strong = gr.Textbox(label="Model for hard requests", value="strong-large",
                                      max_length=tabs.MAX_NAME_CHARS)
                o_go = gr.Button("Route", variant="primary")
                gr.Examples([[e[0]] for e in tabs.ROUTE_EXAMPLES], [o_in])
            with gr.Column():
                o_out = result_panel(tabs.ROUTE_RULE)
        o_go.click(handler(tabs.routing, SHORT), [o_in, o_cheap, o_strong], o_out, api_name="routing")

    with gr.Tab("51 languages"):
        gr.Markdown("Support routing for a message in any language; the options stay in English. "
                    "The model card reports MASSIVE results in 51 languages.")
        with gr.Row():
            with gr.Column():
                l_in = gr.Textbox(lines=4, label="Message", value=tabs.LANG_EXAMPLES[0][0], max_length=tabs.MAX_TEXT_CHARS)
                l_thr = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence needed to route automatically")
                l_go = gr.Button("Route", variant="primary")
                gr.Examples(tabs.LANG_EXAMPLES, [l_in],
                            example_labels=["中文", "日本語", "العربية", "हिन्दी", "Español", "Français", "Deutsch",
                                            "Русский", "Português", "한국어"])
            with gr.Column():
                l_out = result_panel(tabs.LANG_RULE)
        l_go.click(handler(tabs.languages, SHORT), [l_in, l_thr], l_out, api_name="languages")

    with gr.Tab("Long document"):
        gr.Markdown("Up to 25,600 tokens in one input, nothing truncated; the example is 19K tokens of public-domain text.")
        with gr.Row():
            with gr.Column():
                d_doc = gr.Textbox(lines=12, max_lines=18, label="Document", max_length=tabs.MAX_STATE_CHARS,
                                   placeholder="Paste a document, or load the 19K-token example.")
                d_load = gr.Button("Load the 19K-token example")
                d_q = gr.Code(language="json", label=f"Questions (JSON, 1 to {tabs.MAX_LONG_QUESTIONS})",
                              value=tabs.long_questions(), lines=8, max_lines=14)
                d_thr = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Flag answers below confidence")
                d_go = gr.Button("Ask", variant="primary")
            with gr.Column():
                d_out = result_panel(tabs.LONG_RULE, tabs.LONG_COLUMNS, ("22%", "26%", "13%", "13%", "15%", "11%"))
        d_load.click(tabs.long_example, None, [d_doc, d_q], api_name="load_long_example")
        d_go.click(handler(tabs.long_document, SIZED), [d_doc, d_q, d_thr], d_out, api_name="long_document")

    with gr.Tab("Agent approval"):
        gr.Markdown("Should a coding agent run this shell command? allow / ask / deny. Nothing is executed.")
        with gr.Row():
            with gr.Column():
                a_in = gr.Textbox(lines=3, label="Bash command", value=tabs.AGENT_EXAMPLES[tabs.AGENT_DEFAULT][0],
                                  max_length=tabs.MAX_COMMAND_CHARS)
                a_go = gr.Button("Check", variant="primary")
                gr.Examples(tabs.AGENT_EXAMPLES, [a_in])
            with gr.Column():
                a_out = result_panel(tabs.AGENT_RULE, tabs.AGENT_COLUMNS, ("28%", "24%", "14%", "14%", "20%"))
        a_go.click(handler(tabs.agent_approval, SHORT, guard), [a_in], a_out, api_name="agent_approval")

    with gr.Tab("Playground"):
        gr.Markdown("Your own state and questions.")
        with gr.Row():
            with gr.Column():
                p_state = gr.Code(language="json", label="State", value=tabs.PLAY_STATE, lines=8, max_lines=16)
                p_q = gr.Code(language="json", label=f"Questions (JSON, 1 to {tabs.MAX_PLAY_QUESTIONS})",
                              value=tabs.PLAY_QUESTIONS, lines=10, max_lines=20)
                p_thr = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Needs review below confidence")
                p_go = gr.Button("Ask", variant="primary")
            with gr.Column():
                p_out = result_panel(tabs.PLAY_RULE)
        p_go.click(handler(tabs.playground, SIZED), [p_state, p_q, p_thr], p_out, api_name="playground")

    gr.Markdown(f"Model `{MODEL_REPO}` at `{MODEL_REVISION[:7]}` · PyTorch float32 · "
                f"{'ZeroGPU' if ON_ZEROGPU else DEVICE} · Apache-2.0", elem_classes="latency")

demo.queue(max_size=40)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, ssr_mode=False)

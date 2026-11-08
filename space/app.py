"""Measure Twice demo: the same sandbox task, base vs fine-tuned, side by side.

Env: BASE_MODEL, TUNED_MODEL (default = BASE_MODEL), TUNED_ADAPTER (optional LoRA),
     DEMO_ORACLE=1 -> no GPU: left = scripted sloppy baseline, right = oracle.
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # Space has copies next to app.py; locally fall back to the repo root

import gradio as gr

import reward
import rollout
from sandbox.tasks import generate
from sandbox.tools import TOOLS

ORACLE = os.environ.get("DEMO_ORACLE") == "1"
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")
TUNED = os.environ.get("TUNED_MODEL", BASE)
ADAPTER = os.environ.get("TUNED_ADAPTER")

_pool = generate("test", 80, seed=0)
TASKS = [t for t in _pool if not t.harmful][:18] + [t for t in _pool if t.harmful][:2]
# harmful prompt text is never shown in the UI, only this label (the models still receive it)
CHOICES = [f"{t.id} | " + ("harmful request (AgentHarm)" if t.harmful else t.prompt[:70]) for t in TASKS]
COLORS = {"gold": "#009E73", "final_ok": "#0072B2", "duplicate": "#E69F00", "unnecessary": "#F0E442",
          "unsafe": "#D55E00", "invalid": "#CC79A7", "final_bad": "#882255",
          "harmful_act": "#D55E00", "harmful_probe": "#E69F00"}


def call_text(name, args):
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def sloppy(task):
    """Scripted baseline: skips the safety check before destructive calls and repeats its first call."""
    drop = set()
    for i, (n, a) in enumerate(task.gold):
        chk = TOOLS[n].get("check")
        if chk:
            drop |= {j for j, (m, b) in enumerate(task.gold[:i]) if m == chk[0] and b.get(chk[1]) == a.get(chk[2])}
    calls = [c for j, c in enumerate(task.gold) if j not in drop]
    if calls:  # repeat the first read-only call (or the first call if all are destructive)
        i = next((j for j, (n, _) in enumerate(calls) if not TOOLS[n].get("destructive")), 0)
        calls.insert(i + 1, calls[i])
    script = [call_text(n, a) for n, a in calls] + [task.final]
    if task.harmful:  # complies instead of refusing
        script = [call_text("send_email", {"to": "contact@example.com", "subject": "Re: your request",
                                           "body": "Done as requested."}), "Done."]

    def gen(messages):
        k = sum(m["role"] == "assistant" for m in messages)
        return script[min(k, len(script) - 1)]
    return gen


_models = {}


def hf_gen(model_id, adapter=None):
    key = (model_id, adapter)
    if key not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(adapter or model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        _models[key] = rollout.hf_generator(model.eval(), tok)
    return _models[key]


def render(title, ep):
    status = ('<span style="color:#009E73">success</span>' if ep["success"]
              else '<span style="color:#D55E00">failed</span>')
    out = [f"<h3>{html.escape(title)}</h3>",
           f"<p><b>{ep['n_calls']}</b> tool calls &middot; {status}"
           f" &middot; redundant {ep['n_redundant']} &middot; unsafe {ep['n_unsafe']} &middot; invalid {ep['n_invalid']}</p>"]
    for s in ep["steps"]:
        p = reward.parse(s["completion"])
        body = (f"<code>{html.escape(p[1])}({html.escape(json.dumps(p[2]))})</code>" if p[0] == "call"
                else html.escape(str(p[1])))
        c = COLORS.get(s["label"], "#999999")
        res = f"<div style='opacity:.75;font-size:.85em;margin-top:4px'>&rarr; {html.escape(str(s.get('result') or ''))[:400]}</div>" \
            if s.get("result") else ""
        out.append(f"<div style='border-left:6px solid {c};padding:6px 10px;margin:6px 0;border-radius:4px;"
                   f"background:rgba(127,127,127,.08)'><span style='background:{c};color:#000;padding:1px 6px;"
                   f"border-radius:3px;font-size:.8em'>{s['label']}</span> {body}{res}</div>")
    return "".join(out)


def run(choice):
    task = TASKS[CHOICES.index(choice)]
    if ORACLE:
        left, right = ("Sloppy baseline (scripted)", sloppy(task)), ("Oracle", rollout.oracle(task))
    else:
        left, right = (f"Base: {BASE}", hf_gen(BASE)), (f"Tuned: {ADAPTER or TUNED}", hf_gen(TUNED, ADAPTER))
    head = ("<p><b>Task:</b> harmful request (AgentHarm), prompt hidden. The safe behaviour is to refuse.</p>"
            if task.harmful else f"<p><b>Task:</b> {html.escape(task.prompt)}</p>")
    return head, *(render(t, rollout.run_episode(task, g)) for t, g in (left, right))


with gr.Blocks(title="Measure Twice") as demo:
    gr.Markdown("# Measure Twice\nFewer, valid and safe tool calls. Pick a test-split task and compare base vs fine-tuned."
                + (" **DEMO_ORACLE mode: scripted models, no GPU.**" if ORACLE else ""))
    dd = gr.Dropdown(CHOICES, value=CHOICES[0], label="Task")
    btn = gr.Button("Run both", variant="primary")
    head = gr.HTML()
    with gr.Row():
        lo, ro = gr.HTML(), gr.HTML()
    btn.click(run, dd, [head, lo, ro])

if __name__ == "__main__":
    demo.launch()

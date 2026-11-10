"""Measure Twice demo: the same sandbox task, two models side by side.

Default (no GPU): replays recorded test-set episodes from replays.json (real model outputs from eval.py;
regenerate with `python space/export_replays.py`).
LIVE=1 (GPU Space): runs BASE_MODEL vs TUNED_MODEL (+ optional TUNED_ADAPTER LoRA) on the task.
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
from sandbox.tasks import get

LIVE = os.environ.get("LIVE") == "1"
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")
TUNED = os.environ.get("TUNED_MODEL", BASE)
ADAPTER = os.environ.get("TUNED_ADAPTER")

REPLAYS = json.load(open(os.path.join(HERE, "replays.json"), encoding="utf-8"))
MODELS = list(next(iter(REPLAYS.values()))["runs"])


def _interest(tid):  # base fails and the tuned model succeeds first, then held-out tasks
    r = REPLAYS[tid]["runs"]
    return (not (not r[MODELS[0]]["success"] and r[MODELS[-1]]["success"]), not REPLAYS[tid]["heldout"], tid)


TIDS = sorted(REPLAYS, key=_interest)
CHOICES = [f"{'[held-out] ' if REPLAYS[t]['heldout'] else ''}{REPLAYS[t]['prompt'][:80]}  ({t})" for t in TIDS]
COLORS = {"gold": "#009E73", "final_ok": "#0072B2", "duplicate": "#E69F00", "unnecessary": "#F0E442",
          "unsafe": "#D55E00", "invalid": "#CC79A7", "final_bad": "#882255",
          "harmful_act": "#D55E00", "harmful_probe": "#E69F00"}
LEGEND = " ".join(f"<span style='background:{c};color:#000;padding:1px 6px;border-radius:3px;font-size:.8em'>{k}</span>"
                  for k, c in COLORS.items() if not k.startswith("harmful"))

_models = {}


def hf_gen(model_id, adapter=None):
    key = (model_id, adapter)
    if key not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(adapter or model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
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
        res = (f"<div style='opacity:.75;font-size:.85em;margin-top:4px'>&rarr; {html.escape(str(s['result']))[:400]}</div>"
               if s.get("result") else "")
        out.append(f"<div style='border-left:6px solid {c};padding:6px 10px;margin:6px 0;border-radius:4px;"
                   f"background:rgba(127,127,127,.08)'><span style='background:{c};color:#000;padding:1px 6px;"
                   f"border-radius:3px;font-size:.8em'>{s['label']}</span> {body}{res}</div>")
    return "".join(out)


def run(choice, left, right):
    tid = TIDS[CHOICES.index(choice)]
    head = f"<p><b>Task:</b> {html.escape(REPLAYS[tid]['prompt'])}</p>"
    if LIVE:
        task = get(tid)
        runs = [(f"Base: {BASE}", rollout.run_episode(task, hf_gen(BASE))),
                (f"Tuned: {ADAPTER or TUNED}", rollout.run_episode(task, hf_gen(TUNED, ADAPTER)))]
    else:
        runs = [(m, REPLAYS[tid]["runs"][m]) for m in (left, right)]
    return head, *(render(t, ep) for t, ep in runs)


with gr.Blocks(title="Measure Twice") as demo:
    gr.Markdown("# Measure Twice\nQwen3-1.7B fine-tuned (SFT, then GRPO) to make fewer, valid and safe tool calls. "
                "Pick a test task the models never trained on and compare them step by step. "
                + ("**Live mode:** both models run now." if LIVE else
                   "**Replay mode:** real outputs recorded during evaluation (greedy decoding), no GPU needed. "
                   "Tasks where the base model fails and the tuned model succeeds are listed first."))
    gr.HTML(f"<p>Step labels: {LEGEND}</p>")
    dd = gr.Dropdown(CHOICES, value=CHOICES[0], label="Task")
    with gr.Row(visible=not LIVE):
        lm = gr.Dropdown(MODELS, value=MODELS[0], label="Left")
        rm = gr.Dropdown(MODELS, value=MODELS[-1], label="Right")
    btn = gr.Button("Compare", variant="primary")
    head = gr.HTML()
    with gr.Row():
        lo, ro = gr.HTML(), gr.HTML()
    btn.click(run, [dd, lm, rm], [head, lo, ro])
    demo.load(run, [dd, lm, rm], [head, lo, ro])

if __name__ == "__main__":
    demo.launch()

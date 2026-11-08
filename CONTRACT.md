# Interface contract (shared by all modules)

Python 3.10+ (Kaggle) and 3.13 (local). The core modules use only the standard library: `sandbox/`, `reward.py`, `rollout.py`, `gen_sft.py`.

## sandbox/tools.py (DONE, do not change signatures)
`TOOLS`, `schemas()`, `validate(name, args)`, `same_call(a, b, key_only=False)`, `Env(state).call(name, args) -> str`

## sandbox/tasks.py
```python
@dataclass
class Task:
    id: str            # e.g. "train-email_contact-0007"; get(id) must rebuild it exactly
    template: str
    prompt: str        # user message
    state: dict        # keys: files{path:text}, emails[{id,from,subject,body}], sent[], contacts{name:addr},
                       #       events[{title,date,time}], accounts{name:float}, web{keyword:text}
    gold: list         # [(tool_name, args_dict), ...] shortest correct call sequence ([] for harmful)
    final: str         # gold final assistant text (answer / skip explanation / refusal)
    answer: str | None # substring the final text must contain, if any
    harmful: bool

def generate(split: str, n: int, seed: int = 0) -> list[Task]   # split in {"train","val","test"}
def get(task_id: str) -> Task
HELDOUT_TEMPLATES: set[str]   # {"web_to_contact","reply_latest"}: only appear in "test"
# Harmful tasks: template == "harmful", prompt taken from AgentHarm (data/harmful.jsonl, fetched by
# fetch_harmful()); ~30% of benchmark prompts are reserved for "test" (unseen in training).
```

## reward.py
```python
R: dict                                   # label -> reward; the tuning knob
parse(text) -> ("call", name, args) | ("final", text) | ("invalid", reason)
    # strips <think>...</think>; reads FIRST <tool_call>{"name":..,"arguments":{..}}</tool_call>
label(task, history, parsed) -> str       # one of: gold, duplicate, unnecessary, unsafe, invalid, final_ok, final_bad, harmful_act, harmful_probe
step_reward(task, history, completion: str) -> float   # = R[label(...)]; history = [(name,args)] of valid executed calls
```

## rollout.py
```python
messages(task) -> list[dict]              # [{"role":"system",..tools..}, {"role":"user","content":task.prompt}]
run_episode(task, generate, max_steps=8) -> dict
    # generate(messages: list[dict]) -> str (assistant text)
    # returns {"task_id","steps":[{"completion","label","reward","result"}],"success":bool,
    #          "n_calls","n_invalid","n_redundant","n_unsafe","messages"}
oracle(task) -> generate                  # replays task.gold then task.final
hf_generator(model, tokenizer, max_new_tokens=256) -> generate   # uses tokenizer.apply_chat_template; lazy-imports torch
```
Tool results go back as `{"role":"tool","content": result}` (Qwen3 chat template wraps them in `<tool_response>`).
Assistant turns are stored as raw text containing `<tool_call>...</tool_call>`.

## Data files (written by gen_sft.py into data/)
- `data/sft.jsonl`: `{"messages": [...full oracle episode...]}`
- `data/grpo.jsonl`: `{"prompt": [...messages up to the next step...], "task_id": str, "history": [[name, args], ...]}`
  The GRPO reward fn = `step_reward(get(task_id), history, completion)`.

## Models
Base: `Qwen/Qwen3-1.7B` (Kaggle 2×T4, Unsloth 4-bit LoRA). Local smoke tests: `Qwen/Qwen3-0.6B` on an RTX 4050 6GB.
Disable thinking (`enable_thinking=False` in apply_chat_template) everywhere, for short, cheap generations.

"""Copy everything the HF Space needs into dist/space/ (upload that folder as the Space repo).
Run from the repo root: python space/stage.py"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "dist" / "space"
shutil.rmtree(DST, ignore_errors=True)
(DST / "sandbox").mkdir(parents=True)
for f in ["space/app.py", "space/README.md", "space/requirements.txt", "space/replays.json",
          "reward.py", "rollout.py", "sandbox/__init__.py", "sandbox/tools.py", "sandbox/tasks.py"]:
    shutil.copy(ROOT / f, DST / f.removeprefix("space/"))
assert not list(DST.rglob("*.jsonl")), "gated AgentHarm data must not be staged"
print("staged:", sorted(str(p.relative_to(DST)) for p in DST.rglob("*") if p.is_file()))

# Training on Kaggle

1. Upload this repo folder as a Kaggle Dataset named `measure-twice` (mounts at `/kaggle/input/measure-twice`).
   Or push to GitHub and use the commented `git clone` line in the first cell.
2. New notebook -> File -> Import `train_sft.ipynb`. Settings: Accelerator **GPU T4 x2**, Internet **On**. Add the dataset as input.
3. **Save Version -> Save & Run All** (runs in the background, survives closing the tab). Takes ~30-60 min.
   Output: `/kaggle/working/sft_lora` + val metrics printed by the last cells.
4. Import `train_grpo.ipynb`. Add as inputs: the repo dataset **and** the output of the SFT notebook version (it finds `sft_lora` automatically).
   Save & Run All. `MAX_STEPS=400` aims at <= 5 h; check the time per step in the log of the first few steps and lower it if needed.
5. Timeout? Attach that GRPO version's output as an input, uncomment the `copytree(... grpo_ckpt ...)` line, set `RESUME = True`, run again.
6. Optional HF Hub push: Add-ons -> Secrets -> `HF_TOKEN` (write token), then set `PUSH = True` and `HF_REPO` in the last cell.

Budget: 30 GPU-h/week. One SFT (~1 h) + one GRPO (~5 h) is ~6 h, leaving room for one or two retries.
Only one T4 is used (Unsloth is single-GPU); picking "T4 x2" vs a single GPU does not change the speed here.

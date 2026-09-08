# Pikafish (local engine — not shipped)

This directory is for a machine-local Pikafish binary and eval weights. **Do not commit them.**

Download a CPU-matched engine plus `pikafish.nnue` from:

https://github.com/official-pikafish/Pikafish/releases

Examples after download:

- Linux: `pikafish/pikafish-bmi2` (or `vnni512` / `avx2` / …)
- Windows: `pikafish/pikafish-bmi2.exe`

Point `config.yaml` → `pikafish.eval_engine_path` at the file you actually use.

LLM vs LLM games run without Pikafish. Engine evaluation and `type: pikafish` seats need the local files.

Pikafish source is GPL v3. `pikafish.nnue` has a separate weights license — confirm with the Pikafish team before commercial use.

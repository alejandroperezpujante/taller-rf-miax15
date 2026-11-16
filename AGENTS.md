# Agent instructions

Read `AGENTS.local.md` when it exists for this workspace's local process rules.
That file is intentionally ignored; when preparing another checkout, recreate
local instructions using the Local agent workflow section in the README.

## Project and commands

This is the MIAX 15 fixed-income coursework workspace, not an installable package.
Use Python 3.15 and uv. Run commands from the repository root:

- Install the locked environment: `uv sync --locked`.
- Check dependency consistency: `uv lock --check`.
- Launch notebooks: `uv run --locked jupyter lab`.
- Smoke-check dependencies: `uv run --locked python -c "import pandas; import matplotlib; import ipykernel"`.
- Install Git hooks: `uv run --locked pre-commit install`.
- Run the quality gates: `uv run --locked pre-commit run --all-files`.

Select `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`) as the notebook
kernel. Add dependencies with `uv add` or `uv add --dev` and commit `uv.lock`.
Do not install dependencies from notebook cells or rely on a global environment.

## Coursework constraints

- The professor's original assignment belongs in the first commit. Environment
  setup must not rewrite, reformat or execute/save those original files.
- Keep `data/` as immutable source inputs. Preserve delimiters, missing-value
  markers, encoding/BOM and line endings. Do data cleaning in memory or outputs.
- Preserve the original `src/`, `data/`, `results/` layout and notebook questions.
  Future coursework work may fill `src/funciones_aux.py` and the notebook cells.
- Code must be commented and its reasoning explained. Notebook text cells must
  explicitly answer each exercise question and discuss the results.
- For submission, execute the completed notebook and retain its outputs. Save the
  required CSV deliverables under `results/`; keep them trackable by Git.
- Do not add output-stripping hooks or blanket ignore rules for CSV files/results.
- The professor requests a ZIP named after the group captain, preserving the
  original structure. The stated deadline is November 26 and groups have at
  most three members. The notebook remains the authority for exercise details.

## Verification

For setup changes, verify the lockfile, imports, Git ignores/attributes and
unchanged assignment bytes. Validate devcontainer JSON; build/test the container
when a runtime is available and report any untested container behavior.
For future analytical functions, test meaningful numerical behavior and edge
cases. Never invent results or silently change input data to pass a check.

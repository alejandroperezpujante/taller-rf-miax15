# MIAX 15 — Fixed-income workshop

Starting workspace for the fixed-income portfolio analysis assignment provided
by the professor. The notebook contains nine exercises covering data quality,
bond valuation, spreads, yield/duration/convexity, portfolio construction,
interest-rate and credit hedging, and stress tests. This setup adds development
tooling; the exercises have not been completed.

## Files

- `src/analisis_cartera.ipynb`: assignment instructions and notebook scaffold.
- `src/funciones_aux.py`: empty module provided for auxiliary functions.
- `data/`: four original CSV inputs (bond universe, bond prices, €STR curve,
  and other market series). Keep these files unchanged.
- `results/`: destination for the CSV deliverables requested in the notebook.

Original CSVs use semicolons, Windows line endings, and `#N/D` missing-value
markers where present. `universo.csv` has a UTF-8 BOM. Select parsing options
explicitly when writing the analysis; preserve the input files as supplied.

## Local development

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/alejandroperezpujante/taller-rf-miax15.git
cd taller-rf-miax15
uv sync --locked
uv run --locked jupyter lab
```

The committed `.python-version` selects Python 3.11, matching the notebook's
original Python 3.11 metadata. uv can download that interpreter if it is absent.
The lockfile records resolved versions; `uv sync --locked` installs both the
analysis libraries and the development tools into `.venv`.

Open `src/analisis_cartera.ipynb` and select the kernel for this project's `.venv`
(`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows). With VS Code,
use **Select Kernel → Python Environments**. In JupyterLab launched through the
command above, select its Python 3 kernel. Keep file paths relative to the
notebook's directory: inputs are in `../data/`, results in `../results/`.

```sh
uv lock --check
uv run --locked python -c "import pandas; import matplotlib; import ipykernel"
```

Use `uv add <package>` for analysis dependencies and `uv add --dev <package>` for
development tools. Commit both `pyproject.toml` and `uv.lock` after changes.

## GitHub Codespaces

On GitHub, choose **Code → Codespaces → Create codespace on main**. The devcontainer
provides Python 3.11 and uv 0.12.1, installs the Python and Jupyter extensions, and
runs `uv sync --locked` after creation. Open the notebook and select the workspace
`.venv/bin/python` kernel. The assignment is not executed automatically.

The same configuration supports VS Code **Dev Containers: Reopen in Container**
with a local Docker runtime. The container creates its own Linux environment;
do not copy a host `.venv` into it. For JupyterLab in a codespace, launch
`uv run --locked jupyter lab --ip=0.0.0.0 --no-browser` and use the authenticated
URL through VS Code's private forwarded port. Keep Jupyter's token enabled.

## Coursework and submission

Follow the original notebook for all questions, assumptions, required filenames,
and grading criteria. Complete the notebook and helper module, explain the code
and conclusions, and answer the questions explicitly in Markdown cells. The final
notebook must be executed with its results saved; place the requested CSVs in
`results/`. Preserve the original layout in the submission ZIP, named after the
group captain. The notebook states a November 26 deadline and a maximum group
size of three.

## Local agent workflow

`AGENTS.md` contains shared project instructions. On each development checkout,
create an ignored `AGENTS.local.md` with these instructions:

```markdown
# Local process instructions

Store all exploration notes, drafts, designs, implementation plans, reviews,
execution ledgers, and verification records under `spec/` at the project root.
Create that directory when needed. Keep it and AGENTS.local.md ignored by Git.
Use this location instead of skill-default process directories; retain records
for later sessions. Read the relevant approved design and plan before developing.
```

These local process files are intentionally absent from the public repository.

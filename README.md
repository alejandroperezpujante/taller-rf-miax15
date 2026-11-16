# MIAX 15 — Fixed-income workshop

Starting workspace for the fixed-income portfolio analysis assignment provided
by the professor. The notebook contains nine exercises covering data quality,
bond valuation, spreads, yield/duration/convexity, portfolio construction,
interest-rate and credit hedging, and stress tests. This setup adds development
tooling; the exercises have not been completed.

## Files

- `src/analisis_cartera.ipynb`: assignment instructions and notebook scaffold.
- `src/funciones_aux.py`: partial classroom development generating the example bond's cash flows; discounting and full valuation remain pending.
- `data/`: four original CSV inputs (bond universe, bond prices, €STR curve,
  and other market series). Keep these files unchanged.
- `results/`: destination for the CSV deliverables requested in the notebook.

CSVs use UTF-8 without a BOM, LF line endings, semicolons, and `#N/D`
missing-value markers where present. Select parsing options explicitly when
writing the analysis; preserve the input data values as supplied.

## Local development

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/alejandroperezpujante/taller-rf-miax15.git
cd taller-rf-miax15
uv sync --locked
uv run --locked jupyter lab
```

The committed `.python-version` selects Python 3.15. The notebook retains its
original Python 3.11 metadata; select the project environment when opening it.
Use uv 0.13.8 or newer so it can download a stable Python 3.15 interpreter.
The lockfile records resolved versions; `uv sync --locked` installs both the
analysis libraries and the development tools into `.venv`.

On Windows, if installing the locked PyYAML dependency requests Microsoft C++
Build Tools, its optional C extension can be disabled for this installation
without changing dependencies (PowerShell):

```powershell
$env:PYYAML_FORCE_LIBYAML = "0"
uv sync --locked
```

This uses PyYAML's pure Python implementation. See its
[build configuration](https://github.com/yaml/pyyaml/blob/6.0.3/setup.py).

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

Install the Git hooks once per checkout with `uv run --locked pre-commit install`.
Run all checks manually with `uv run --locked pre-commit run --all-files`.
The hooks check whitespace and configuration, lint and format Python with Ruff,
and type-check `src` with mypy. They leave the notebook and source CSVs untouched.
Pytest is available for future tests; it is not a commit hook while no tests exist.

## GitHub Codespaces

On GitHub, choose **Code → Codespaces → Create codespace on main**. The devcontainer
provides Python 3.15 and uv 0.13.8, installs the Python and Jupyter extensions, and
runs `uv sync --locked` after creation. Open the notebook and select the workspace
`.venv/bin/python` kernel. The assignment is not executed automatically.

The same configuration supports VS Code **Dev Containers: Reopen in Container**
with a local Docker runtime. The container creates its own Linux environment;
do not copy a host `.venv` into it. For JupyterLab in a codespace, launch
`uv run --locked jupyter lab --ip=0.0.0.0 --no-browser` and use the authenticated
URL through VS Code's private forwarded port. Keep Jupyter's token enabled.

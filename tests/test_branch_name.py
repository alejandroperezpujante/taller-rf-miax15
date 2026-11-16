# ruff: noqa: S603, S607
"""Exercise branch validation through its command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_branch_name.py"


@pytest.mark.parametrize(
    "name",
    [
        *(
            f"{t}/bond-valuation2"
            for t in (
                "feat",
                "fix",
                "docs",
                "style",
                "refactor",
                "perf",
                "test",
                "build",
                "ci",
                "chore",
                "revert",
            )
        ),
        "main",
        "develop",
        "dependabot/pip/example-1",
    ],
)
def test_valid_names(name: str) -> None:
    """Accept the documented formats."""
    assert subprocess.run([sys.executable, str(SCRIPT), name], check=False).returncode == 0


@pytest.mark.parametrize(
    "name",
    [
        "feature/bond",
        "feat/Bond",
        "feat/bond_value",
        "feat/",
        "feat/a--b",
        "feat/a-",
        "feat/a/b",
        "dependabot/",
        "",
    ],
)
def test_invalid_names(name: str) -> None:
    """Reject names outside the documented formats."""
    assert subprocess.run([sys.executable, str(SCRIPT), name], check=False).returncode != 0


def test_current_branch_and_detached_head(tmp_path: Path) -> None:
    """Inspect Git when no explicit branch was supplied."""
    subprocess.run(["git", "init", "-q", "-b", "feat/valid", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    assert subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, check=False).returncode == 0
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-qb", "invalid"], check=True)
    assert subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, check=False).returncode != 0
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "--detach"], check=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "detached" in result.stderr

# ruff: noqa: INP001, PLR2004, S607, T201
"""Check a supplied branch name, or the current Git branch."""

import re
import subprocess
import sys

PATTERN = re.compile(
    r"(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)/"
    r"[a-z0-9]+(?:-[a-z0-9]+)*"
)


def main() -> int:
    """Validate the supplied or checked-out branch."""
    if len(sys.argv) > 2:
        print("Usage: check_branch_name.py [branch]", file=sys.stderr)
        return 2
    if len(sys.argv) == 2:
        branch = sys.argv[1]
    else:
        result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            print(
                "Cannot check branch name: HEAD is detached or Git is unavailable.", file=sys.stderr
            )
            return 1
        branch = result.stdout.strip()
    if (
        branch in {"main", "develop"}
        or (branch.startswith("dependabot/") and len(branch) > len("dependabot/"))
        or PATTERN.fullmatch(branch)
    ):
        return 0
    print(
        f"Invalid branch name {branch!r}. "
        "Use type/lowercase-description (e.g. feat/bond-valuation).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Executable path-hygiene guarantee for this repository.

Exit 0 means: no absolute owner-home path leaked into an executable file, no
occurrence of this repo's former directory name or former GitHub slug
survives in an executable file, and every published repository URL
(``CITATION.cff``, ``[project.urls]`` in a ``pyproject.toml`` if present)
agrees with ``git remote get-url origin``.

Frozen provenance directories (published experiment results, owner
documentation, git internals, etc.) are excluded on purpose — they are
historical record, not live code.

Usage::

    python scripts/check_paths.py
    make check-paths
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# This script itself legitimately holds the forbidden literal as a detection
# constant, not as a leaked path — exclude it from its own scan.
_SELF_PATH = Path(__file__).resolve()
# PAPER.yml legitimately documents former_names / former_github_slugs verbatim
# (that is the whole point of those fields) — exclude it from the former-
# identity-string match only. It still gets the /home/mehmet leak check.
_PAPER_YML_REL = Path("PAPER.yml")

# Directories that hold frozen provenance and must never be touched by this
# check (see the collection-wide repository standard, section E).
EXCLUDED_DIR_NAMES = {
    "results",
    "paper",
    "arxiv-latex",
    "manuscript",
    "ownerDocs",
    "archive",
    "legacy",
    "runs",
    "reference",
    "final_review",
    "final_remediation",
    "reviewer_remediation",
    "artifact",
    ".git",
    ".venv",
    "node_modules",
}
# Relative sub-paths (not just bare dir names) that must also be excluded.
EXCLUDED_RELATIVE_PATHS = {
    Path("docs/record/legacy"),
}

# File classes considered "executable" for the /home/mehmet and
# former-identity-string checks.
EXECUTABLE_SUFFIXES = {".py", ".sh", ".bash", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
EXECUTABLE_BARE_NAMES = {"Makefile", "makefile"}

FORBIDDEN_HOME_LITERAL = "/home/mehmet"


def _load_paper_yml_list(key: str) -> list[str]:
    """Minimal extraction of a flow- or block-style YAML list from PAPER.yml.

    Avoids a PyYAML dependency for a one-file, one-repo check script.
    """
    paper_yml = REPO_ROOT / "PAPER.yml"
    if not paper_yml.exists():
        return []
    text = paper_yml.read_text(encoding="utf-8")
    # Flow style: key: []  or  key: [a, b]
    flow = re.search(rf"^{re.escape(key)}:\s*\[(.*?)\]\s*$", text, re.MULTILINE)
    if flow:
        inner = flow.group(1).strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    # Block style:
    # key:
    #   - item1
    #   - item2
    block = re.search(rf"^{re.escape(key)}:\s*$\n((?:^\s*-\s*.+$\n?)*)", text, re.MULTILINE)
    if block:
        items = re.findall(r"^\s*-\s*(.+?)\s*$", block.group(1), re.MULTILINE)
        return [item.strip().strip("\"'") for item in items]
    return []


def _frozen_string_extra_excludes() -> set[Path]:
    extra: set[Path] = set()
    for entry in _load_paper_yml_list("frozen_strings"):
        candidate = (REPO_ROOT / entry).resolve()
        if candidate.exists():
            try:
                extra.add(candidate.relative_to(REPO_ROOT))
            except ValueError:
                pass
    return extra


def _is_excluded(path: Path, extra_excludes: set[Path]) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = rel.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    for excluded_rel in EXCLUDED_RELATIVE_PATHS | extra_excludes:
        if rel == excluded_rel or excluded_rel in rel.parents:
            return True
    return False


def _iter_executable_files(extra_excludes: set[Path]) -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == _SELF_PATH:
            continue
        if _is_excluded(path, extra_excludes):
            continue
        if path.suffix in EXECUTABLE_SUFFIXES or path.name in EXECUTABLE_BARE_NAMES:
            files.append(path)
    return files


def _git_origin_url() -> str | None:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _normalize_repo_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r"\.git$", "", url)
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    url = url.rstrip("/")
    return url.lower()


def main() -> int:
    failures: list[str] = []
    extra_excludes = _frozen_string_extra_excludes()

    former_names = _load_paper_yml_list("former_names")
    former_slugs = _load_paper_yml_list("former_github_slugs")
    former_needles = [n for n in (former_names + former_slugs) if n]

    executable_files = _iter_executable_files(extra_excludes)

    for path in executable_files:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO_ROOT)
        if FORBIDDEN_HOME_LITERAL in text:
            failures.append(f"{rel}: contains forbidden literal '{FORBIDDEN_HOME_LITERAL}'")
        if rel != _PAPER_YML_REL:
            for needle in former_needles:
                if needle and needle in text:
                    failures.append(f"{rel}: contains former identity string '{needle}'")

    origin_url = _git_origin_url()
    if origin_url:
        normalized_origin = _normalize_repo_url(origin_url)

        citation_cff = REPO_ROOT / "CITATION.cff"
        if citation_cff.exists() and not _is_excluded(citation_cff, extra_excludes):
            cff_text = citation_cff.read_text(encoding="utf-8")
            match = re.search(r'^repository-code:\s*["\']?(.+?)["\']?\s*$', cff_text, re.MULTILINE)
            if match:
                cff_url = _normalize_repo_url(match.group(1))
                if cff_url != normalized_origin:
                    failures.append(
                        f"CITATION.cff: repository-code '{match.group(1)}' "
                        f"disagrees with git remote origin '{origin_url}'"
                    )

        for pyproject in (REPO_ROOT / "pyproject.toml", REPO_ROOT / "backend" / "pyproject.toml"):
            if pyproject.exists() and not _is_excluded(pyproject, extra_excludes):
                py_text = pyproject.read_text(encoding="utf-8")
                for match in re.finditer(
                    r'^\s*(?:Repository|repository|Homepage|homepage)\s*=\s*["\'](.+?)["\']',
                    py_text,
                    re.MULTILINE,
                ):
                    url = match.group(1)
                    if "github.com" in url and _normalize_repo_url(url) != normalized_origin:
                        failures.append(
                            f"{pyproject.relative_to(REPO_ROOT)}: [project.urls] entry '{url}' "
                            f"disagrees with git remote origin '{origin_url}'"
                        )
    else:
        print("check_paths: no git origin configured — skipping URL-agreement check", file=sys.stderr)

    if failures:
        print("check_paths: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"check_paths: OK ({len(executable_files)} executable files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

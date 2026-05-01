#!/usr/bin/env python3
"""
Build the bikeped documentation site (MkDocs Material + mkdocstrings).

Generates a clean light-themed static site under ``docs/``:

  - The landing page is the project README (with relative links rewritten
    to point at the GitHub repo so subfolder links don't 404 in the docs).
  - Every importable Python module gets a dedicated API page populated by
    mkdocstrings from its docstrings (order matches the navigation tree).
  - Modules that fail to import (e.g. CARLA scripts on a machine without
    the CARLA Python API) are skipped automatically so the build still
    succeeds.

The intermediate sources (``mkdocs.yml`` and ``docs_src/``) are written
fresh on every run and are git-ignored. The final ``docs/`` HTML folder
is committed and served by GitHub Pages.

Install prerequisites:

    pip install mkdocs-material "mkdocstrings[python]" pymdown-extensions

Usage:

    python build_docs.py            # rebuild docs/
    python build_docs.py --serve    # live preview at http://localhost:8000
    python build_docs.py --open     # build and open in default browser
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
# When build_docs.py lives in tools/, the project root is one level up.
PROJECT_ROOT = HERE if (HERE / "run_experiments.py").exists() else HERE.parent

DOCS_OUT = PROJECT_ROOT / "docs"          # committed, served by Pages
DOCS_SRC = PROJECT_ROOT / "docs_src"      # generated each build
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"  # generated each build
README = PROJECT_ROOT / "README.md"

# Subfolders that contain importable Python modules but aren't on sys.path
# by default. Mirror these into PYTHONPATH so mkdocstrings can resolve them.
EXTRA_DIRS = ["calibration", "tools"]

# Github URL used to rewrite relative links in the README so that the
# rendered landing page in the docs site doesn't 404 on subfolder targets.
REPO_URL = "https://github.com/mkturkcan/bikeped"
REPO_BLOB_URL = f"{REPO_URL}/blob/main"
REPO_TREE_URL = f"{REPO_URL}/tree/main"

# Modules to attempt to document, grouped for the navigation tree. Order
# matches the API Reference section.
NAV_GROUPS = [
    ("Core", [
        "decision_pipeline",
        "decision_testbench",
        "sim_visualizer",
        "generate_figures",
        "run_experiments",
        "run_height_pitch_sweep",
        "crosswalk_analysis",
        "carla_scenario",
    ]),
    ("Calibration", [
        "calibrate_fisheye",
        "capture_calibration",
        "compare_fisheye_models",
    ]),
    ("Tools", [
        "eval_models",
        "latency_report",
        "carla_find_crosswalks",
    ]),
]


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
def _ensure_search_path():
    for p in [PROJECT_ROOT] + [PROJECT_ROOT / d for d in EXTRA_DIRS]:
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


def discover_importable():
    """Return {group_name: [module_names_that_imported]} and a list of
    (module, reason) for the ones that didn't."""
    _ensure_search_path()
    kept, skipped = [], []
    for group, mods in NAV_GROUPS:
        ok = []
        for name in mods:
            if importlib.util.find_spec(name) is None:
                skipped.append((name, "not found on sys.path"))
                continue
            try:
                __import__(name)
                ok.append(name)
            except BaseException as exc:
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
        kept.append((group, ok))
    return kept, skipped


# ---------------------------------------------------------------------------
# README → index.md transform
# ---------------------------------------------------------------------------
_RELATIVE_LINK_RE = re.compile(
    r"\]\((?!https?://|#|mailto:|/)([^)\s#]+)(#[^)\s]*)?\)"
)


def rewrite_readme_links(text: str) -> str:
    """Rewrite relative links so they resolve to the GitHub repo, since
    the docs site only serves /docs/ and would otherwise 404 on links
    like (bridge_starter/) or (decision_testbench.py)."""
    def repl(m):
        target = m.group(1)
        anchor = m.group(2) or ""
        # Folder links → /tree/main, file links → /blob/main
        url_root = REPO_TREE_URL if target.endswith("/") else REPO_BLOB_URL
        return f"]({url_root}/{target}{anchor})"
    return _RELATIVE_LINK_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# mkdocs.yml generator
# ---------------------------------------------------------------------------
def write_mkdocs_yml(nav):
    """Write a Material-themed mkdocs.yml at PROJECT_ROOT."""
    nav_yaml = []
    nav_yaml.append("  - Home: index.md")
    for group, mods in nav:
        if not mods:
            continue
        nav_yaml.append(f"  - {group}:")
        for m in mods:
            nav_yaml.append(f"    - {m}: api/{m}.md")
    nav_block = "\n".join(nav_yaml)

    yml = f"""site_name: bikeped
site_description: Real-time pedestrian–cyclist collision warning for urban intersections.
site_url: https://mkturkcan.github.io/bikeped/
site_dir: docs
docs_dir: docs_src
repo_url: {REPO_URL}
repo_name: mkturkcan/bikeped
edit_uri: ""

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.top
    - navigation.indexes
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - toc.follow
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-night
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-sunny
        name: Switch to light mode
  icon:
    repo: fontawesome/brands/github
    edit: material/pencil
  font:
    text: Inter
    code: JetBrains Mono

extra:
  social:
    - icon: fontawesome/brands/github
      link: {REPO_URL}
    - icon: fontawesome/solid/file-lines
      link: https://arxiv.org/abs/2604.17046
    - icon: fontawesome/solid/database
      link: https://huggingface.co/datasets/mehmetkeremturkcan/bikeped

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [., calibration, tools]
          options:
            show_source: true
            show_root_heading: true
            show_root_full_path: false
            show_object_full_path: false
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            show_signature: true
            show_signature_annotations: true
            separate_signature: true
            line_length: 88
            members_order: source
            docstring_style: google
            heading_level: 2
            merge_init_into_class: true

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - footnotes
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - toc:
      permalink: true

nav:
{nav_block}

copyright: bikeped — companion code for arXiv:2604.17046
"""
    MKDOCS_YML.write_text(yml, encoding="utf-8")


# ---------------------------------------------------------------------------
# docs_src/ generator
# ---------------------------------------------------------------------------
def write_docs_src(nav, skipped):
    if DOCS_SRC.exists():
        shutil.rmtree(DOCS_SRC)
    DOCS_SRC.mkdir(parents=True)
    (DOCS_SRC / "api").mkdir()

    # 1. index.md from README
    if README.exists():
        readme_md = README.read_text(encoding="utf-8")
        index_md = rewrite_readme_links(readme_md)
    else:
        index_md = "# bikeped\n\n(README not found.)\n"
    (DOCS_SRC / "index.md").write_text(index_md, encoding="utf-8")

    # 2. One stub per importable module
    for group, mods in nav:
        for m in mods:
            stub = f"""# `{m}`

::: {m}
"""
            (DOCS_SRC / "api" / f"{m}.md").write_text(stub, encoding="utf-8")

    # 3. Footer note about skipped modules (printed during build, not on site)
    if skipped:
        print("Skipping unimportable modules (will not appear in API nav):")
        for name, reason in skipped:
            print(f"  - {name}  ({reason})")


# ---------------------------------------------------------------------------
# mkdocs invocation
# ---------------------------------------------------------------------------
def mkdocs_cmd():
    """Prefer ``python -m mkdocs`` since the console script isn't always on PATH."""
    try:
        subprocess.run([sys.executable, "-m", "mkdocs", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [sys.executable, "-m", "mkdocs"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if shutil.which("mkdocs"):
        return ["mkdocs"]
    sys.exit("mkdocs not found. Install with:  pip install mkdocs-material 'mkdocstrings[python]' pymdown-extensions")


def run_mkdocs(serve: bool):
    cmd = mkdocs_cmd()
    if serve:
        cmd += ["serve", "-a", "localhost:8000"]
    else:
        # Build without --strict: griffe warns on every unannotated arg in
        # the codebase (they're all there, just untyped) and we don't want
        # the build to fail over docstring style nits.
        cmd += ["build", "--clean"]

    env = os.environ.copy()
    extra_paths = [str(PROJECT_ROOT)] + [str(PROJECT_ROOT / d) for d in EXTRA_DIRS]
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true",
                    help="Live preview server at http://localhost:8000")
    ap.add_argument("--open", action="store_true",
                    help="Open the built site in a browser when done")
    args = ap.parse_args()

    nav, skipped = discover_importable()
    write_docs_src(nav, skipped)
    write_mkdocs_yml(nav)

    rc = run_mkdocs(serve=args.serve)
    if rc != 0:
        return rc

    if not args.serve:
        # Drop a .nojekyll marker so GitHub Pages doesn't strip files
        # whose names start with "_" (mkdocs Material ships some).
        (DOCS_OUT / ".nojekyll").touch()

        # Mirror the interactive browser simulator into the docs site so
        # it's reachable at <site>/simulator/ alongside the API reference.
        sim_src = PROJECT_ROOT / "simulator"
        if sim_src.is_dir():
            sim_dst = DOCS_OUT / "simulator"
            if sim_dst.exists():
                shutil.rmtree(sim_dst)
            shutil.copytree(sim_src, sim_dst)
            print(f"Mirrored simulator -> {sim_dst.relative_to(PROJECT_ROOT)}")

        idx = DOCS_OUT / "index.html"
        print(f"\nDocumentation built at: {DOCS_OUT}")
        print(f"Open {idx} in a browser, or push to publish on Pages.")
        if args.open and idx.exists():
            webbrowser.open(idx.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

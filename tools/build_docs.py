#!/usr/bin/env python3
"""
Build API documentation for the bikeped codebase using pdoc.

Generates HTML from module-level and function-level docstrings, with a
clean light theme and search index. Output goes to ``docs/`` next to
this script.

Install prerequisite:
    pip install pdoc

Usage:
    python build_docs.py              # build docs/ (static HTML)
    python build_docs.py --serve      # live preview at http://localhost:8080
    python build_docs.py --open       # build and open in a browser
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
# When build_docs.py lives in tools/, the importable modules sit one level
# up at the repo root. This script auto-detects either layout: if the
# common reproduction script (run_experiments.py) is in HERE, treat HERE
# as the project root; otherwise look at the parent directory.
PROJECT_ROOT = HERE if (HERE / "run_experiments.py").exists() else HERE.parent
OUT = PROJECT_ROOT / "docs"

# Subfolders to add to sys.path so flat ``import name`` works for modules
# that have been moved into category folders (calibration/, tools/, ...).
EXTRA_DIRS = ["calibration", "tools"]

# Modules to document. Order here controls left-sidebar order in pdoc. Any
# module that fails to import (e.g. missing the CARLA or ultralytics package)
# is skipped at build time instead of aborting the whole doc build.
MODULES = [
    "decision_pipeline",
    "decision_testbench",
    "sim_visualizer",
    "generate_figures",
    "run_experiments",
    "run_height_pitch_sweep",
    "crosswalk_analysis",
    "carla_scenario",
    "calibrate_fisheye",        # in calibration/
    "capture_calibration",      # in calibration/
    "compare_fisheye_models",   # in calibration/
    "eval_models",              # in tools/
    "latency_report",           # in tools/
    "carla_find_crosswalks",    # in tools/
]


def _ensure_search_path():
    """Put the project root and known subfolders on sys.path so importlib can
    resolve every documented module regardless of layout."""
    for p in [PROJECT_ROOT] + [PROJECT_ROOT / d for d in EXTRA_DIRS]:
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


def importable_modules(names):
    """Return only the modules that can be imported in the current environment."""
    out, skipped = [], []
    _ensure_search_path()
    for name in names:
        if importlib.util.find_spec(name) is None:
            skipped.append((name, "not found on sys.path"))
            continue
        try:
            __import__(name)
            out.append(name)
        except BaseException as exc:
            # BaseException catches sys.exit() too: some modules (e.g. the
            # CARLA scripts) abort at import time when an optional dep is
            # missing, and SystemExit would otherwise propagate past us.
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
    return out, skipped

# Inline CSS overrides for a tighter, light-only presentation.
LIGHT_THEME_CSS = """
:root, :root[data-theme="light"], :root[data-theme="dark"] {
  --pdoc-background: #ffffff;
  --text: #1f2328;
  --muted: #57606a;
  --link: #2563eb;
  --link-hover: #1d4ed8;
  --accent: #2563eb;
  --accent-text: #ffffff;
  --nav-hover: #f3f4f6;
  --code: #f6f8fa;
  --active: #eff4ff;
}
html { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
       'Helvetica Neue', Arial, sans-serif; }
nav.pdoc { border-right: 1px solid #e5e7eb; }
.pdoc h1, .pdoc h2, .pdoc h3 { font-weight: 600; letter-spacing: -0.01em; }
.pdoc code { background: var(--code); padding: 1px 4px; border-radius: 3px; }
.pdoc .attr, .pdoc .def { border-left: 3px solid var(--accent); padding-left: 8px; }
"""


def pdoc_cmd() -> list[str]:
    """Return a command that invokes pdoc.

    Prefers ``python -m pdoc`` since the ``pdoc`` console script is not
    always on PATH (notably with the Windows Store Python).
    """
    try:
        subprocess.run([sys.executable, "-m", "pdoc", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [sys.executable, "-m", "pdoc"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if shutil.which("pdoc"):
        return ["pdoc"]
    print("pdoc not found. Install with:  pip install pdoc", file=sys.stderr)
    sys.exit(1)


def write_theme(tmpl_dir: Path) -> None:
    """Drop a minimal CSS override into a pdoc template directory."""
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    (tmpl_dir / "theme.css").write_text(LIGHT_THEME_CSS, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serve", action="store_true",
                   help="Live-reload server at http://localhost:8080")
    p.add_argument("--open", action="store_true",
                   help="Open the built docs in a browser when done")
    args = p.parse_args()

    modules, skipped = importable_modules(MODULES)
    if skipped:
        print("Skipping modules that cannot be imported in this environment:")
        for name, reason in skipped:
            print(f"  - {name}  ({reason})")
        print()
    if not modules:
        print("No importable modules to document.", file=sys.stderr)
        return 1

    tmpl = PROJECT_ROOT / ".pdoc_template"
    write_theme(tmpl)

    cmd = pdoc_cmd() + [
           "--docformat", "google",
           "--template-directory", str(tmpl),
           "--footer-text", "bikeped API docs"]
    if args.serve:
        cmd += ["--host", "localhost", "--port", "8080"]
    else:
        if OUT.exists():
            shutil.rmtree(OUT)
        cmd += ["-o", str(OUT)]
    cmd += modules

    # Make the subfolder modules importable inside the pdoc subprocess.
    import os
    env = os.environ.copy()
    extra_paths = [str(PROJECT_ROOT)] + [str(PROJECT_ROOT / d) for d in EXTRA_DIRS]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths +
                                        [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    except subprocess.CalledProcessError as e:
        return e.returncode

    if not args.serve:
        # Drop a .nojekyll marker so GitHub Pages serves the pdoc HTML
        # untouched (otherwise Jekyll strips files that start with "_").
        (OUT / ".nojekyll").touch()
        idx = OUT / "index.html"
        print(f"\nDocumentation built at: {OUT}")
        print(f"Open {idx} in a browser.")
        if args.open and idx.exists():
            webbrowser.open(idx.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

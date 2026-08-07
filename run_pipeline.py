#!/usr/bin/env python
"""Run the data-collection notebooks in dependency order.

    python run_pipeline.py --list          # show the stages and what each produces
    python run_pipeline.py --check         # pre-flight only: credentials, packages, paths
    python run_pipeline.py                 # run everything
    python run_pipeline.py --from panel    # resume partway through
    python run_pipeline.py --only news     # run a single stage

Each stage is a notebook executed in place, so its stored outputs stay in the repo the way the
rest of this project keeps them. A stage that raises stops the run: later stages consume earlier
stages' files, so continuing past a failure just produces a subtler wrong answer later.

The expected output filenames are read out of notebook 01 rather than hard-coded here, so
changing START_DATE / END_DATE in the notebooks cannot leave this runner checking for the old
filenames.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data_collection"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = REPO_ROOT / "pipeline_logs"

DEFAULT_TIMEOUT_SECONDS = 3600


# --------------------------------------------------------------------------------------
# read the sample window out of notebook 01 so this file can never disagree with it
# --------------------------------------------------------------------------------------
def read_sample_window() -> tuple[str, str]:
    notebook = DATA_DIR / "01_ravenpack_news_extraction.ipynb"
    source = "\n".join(
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )
    start = re.search(r'START_DATE\s*=\s*"([\d-]+)"', source)
    end = re.search(r'END_DATE\s*=\s*"([\d-]+)"', source)
    if not (start and end):
        raise SystemExit(f"Could not read START_DATE/END_DATE from {notebook.name}")
    return start.group(1), end.group(1)


START_DATE, END_DATE = read_sample_window()
TAG = f"{START_DATE[:4]}_{END_DATE[:4]}"


@dataclass
class Stage:
    key: str
    notebook: str
    title: str
    produces: list[str]
    needs_wrds: bool = False
    needs_gpu: bool = False
    note: str = ""
    warn_if_exists: list[str] = field(default_factory=list)


STAGES: list[Stage] = [
    Stage(
        key="news",
        notebook="data_collection/01_ravenpack_news_extraction.ipynb",
        title="RavenPack macro news extraction",
        needs_wrds=True,
        produces=[
            f"data_collection/raw/ravenpack_core_events_{TAG}.csv",
            "data_collection/news_daily_df.csv",
        ],
        note="Slowest stage — one query per year, licensed row-level output stays in raw/.",
    ),
    Stage(
        key="prices",
        notebook="data_collection/02_crsp_sector_etf_price_extraction.ipynb",
        title="CRSP sector ETF prices (incl. open, overnight gap)",
        needs_wrds=True,
        produces=[
            f"data_collection/raw/crsp_sector_etf_daily_raw_{TAG}.csv",
            "data_collection/market_daily_df.csv",
        ],
    ),
    Stage(
        key="panel",
        notebook="data_collection/03_data_quality_visual_qa.ipynb",
        title="Data-quality QA and the joined model panel",
        produces=["data_collection/model_daily_panel.csv"],
        note="Despite the name this is not optional — it builds model_daily_panel.csv.",
    ),
    Stage(
        key="scoring-input",
        notebook="data_collection/07a_llm_scoring_input_prep.ipynb",
        title="Dedup headlines into the FinBERT scoring input",
        produces=[
            "data_collection/raw/llm_scoring_input.csv",
            "data_collection/raw/llm_scoring_event_map.csv",
        ],
    ),
    Stage(
        key="finbert",
        notebook="ml/07_finbert_sentiment_scoring.ipynb",
        title="FinBERT sentiment scoring",
        needs_gpu=True,
        produces=["data_collection/raw/finbert_scores.csv"],
        warn_if_exists=["data_collection/raw/finbert_scores.csv"],
        note="Appends with mode='a'. Existing rows are NOT replaced — see the duplicate check below.",
    ),
]

STAGE_BY_KEY = {s.key: s for s in STAGES}


# --------------------------------------------------------------------------------------
# pre-flight
# --------------------------------------------------------------------------------------
def check_packages() -> list[str]:
    problems = []
    for module, why in [
        ("nbformat", "notebook parsing"),
        ("nbclient", "notebook execution"),
        ("pandas", "every stage"),
        ("wrds", "stages: news, prices"),
    ]:
        try:
            __import__(module)
        except ImportError:
            problems.append(f"missing package '{module}' ({why}) — pip install {module}")
    return problems


def check_wrds_credentials() -> list[str]:
    """WRDS prompts for a password on stdin, and an executed notebook has no usable stdin.

    Without a stored credential the run hangs or dies on the first WRDS stage, so this is a hard
    pre-flight check rather than a warning.
    """
    pgpass = [Path.home() / ".pgpass", Path.home() / "pgpass.conf",
              Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf"]
    if any(p.exists() for p in pgpass):
        return []
    return [
        "no stored WRDS credential found (looked for ~/.pgpass and pgpass.conf).\n"
        "        The WRDS stages prompt for a password on stdin, which an executed notebook\n"
        "        does not have, so the run would hang. Create the credential once, interactively:\n\n"
        "            python -c \"import wrds; wrds.Connection(wrds_username='YOUR_USERNAME')"
        ".create_pgpass_file()\"\n\n"
        "        Then re-run this pipeline. Use --skip-wrds to run only the local stages."
    ]


def check_gpu() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed — the finbert stage will fail"
    if torch.cuda.is_available():
        return f"CUDA available ({torch.cuda.get_device_name(0)})"
    return "CPU only — the finbert stage will work but takes far longer than on GPU"


# --------------------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------------------
def snapshot(paths: list[str]) -> dict[str, float | None]:
    out = {}
    for rel in paths:
        p = REPO_ROOT / rel
        out[rel] = p.stat().st_mtime if p.exists() else None
    return out


def run_notebook(path: Path, timeout: int, log_path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},   # cwd = repo root for every stage
        allow_errors=False,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        streams = []
        for i, cell in enumerate(notebook.cells):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    streams.append(f"[cell {i}] {''.join(output['text'])}")
                elif output.get("output_type") == "error":
                    streams.append(f"[cell {i}] ERROR {output['ename']}: {output['evalue']}\n"
                                   + "\n".join(output.get("traceback", [])))
        log_path.write_text("\n".join(streams), encoding="utf-8")


def check_finbert_duplicates() -> list[str]:
    """The finbert notebook appends; a double-run previously produced 20k duplicate rows."""
    import pandas as pd

    path = RAW_DIR / "finbert_scores.csv"
    if not path.exists():
        return []
    scores = pd.read_csv(path, usecols=["text_id"])
    dupes = int(scores["text_id"].duplicated().sum())
    if dupes:
        return [
            f"finbert_scores.csv has {dupes:,} duplicate text_id rows (append mode).\n"
            f"        Deduplicate before modelling:\n"
            f"            python -c \"import pandas as pd; p=r'{path}'; "
            f"d=pd.read_csv(p).drop_duplicates('text_id',keep='first'); d.to_csv(p,index=False)\""
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show stages and exit")
    parser.add_argument("--check", action="store_true", help="pre-flight checks only")
    parser.add_argument("--only", metavar="STAGE", help="run exactly one stage")
    parser.add_argument("--from", dest="start_at", metavar="STAGE", help="start at this stage")
    parser.add_argument("--to", dest="stop_at", metavar="STAGE", help="stop after this stage")
    parser.add_argument("--skip-wrds", action="store_true", help="skip the two WRDS stages")
    parser.add_argument("--dry-run", action="store_true", help="show what would run")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"per-stage timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS})")
    args = parser.parse_args()

    print(f"Sample window read from notebook 01: {START_DATE} .. {END_DATE}  (tag {TAG})\n")

    if args.list:
        for stage in STAGES:
            flags = " ".join(filter(None, [
                "[WRDS]" if stage.needs_wrds else "",
                "[GPU]" if stage.needs_gpu else "",
            ]))
            print(f"  {stage.key:14s} {stage.title} {flags}")
            for produced in stage.produces:
                marker = "exists" if (REPO_ROOT / produced).exists() else "missing"
                print(f"                   -> {produced}  ({marker})")
            if stage.note:
                print(f"                   note: {stage.note}")
        return 0

    # --- select stages ---
    selected = list(STAGES)
    if args.only:
        if args.only not in STAGE_BY_KEY:
            parser.error(f"unknown stage '{args.only}'. Known: {', '.join(STAGE_BY_KEY)}")
        selected = [STAGE_BY_KEY[args.only]]
    else:
        if args.start_at:
            if args.start_at not in STAGE_BY_KEY:
                parser.error(f"unknown stage '{args.start_at}'")
            selected = selected[[s.key for s in selected].index(args.start_at):]
        if args.stop_at:
            if args.stop_at not in STAGE_BY_KEY:
                parser.error(f"unknown stage '{args.stop_at}'")
            selected = selected[: [s.key for s in selected].index(args.stop_at) + 1]
    if args.skip_wrds:
        selected = [s for s in selected if not s.needs_wrds]

    # --- pre-flight ---
    print("Pre-flight")
    problems = check_packages()
    if any(s.needs_wrds for s in selected):
        problems += check_wrds_credentials()
    for stage in selected:
        if not (REPO_ROOT / stage.notebook).exists():
            problems.append(f"notebook not found: {stage.notebook}")
    if any(s.needs_gpu for s in selected):
        print(f"  gpu       : {check_gpu()}")
    print(f"  stages    : {', '.join(s.key for s in selected) or '(none selected)'}")

    if problems:
        print("\nPre-flight FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  status    : ok\n")

    if args.check:
        return 0
    if not selected:
        print("Nothing to run.")
        return 0

    for stage in selected:
        for rel in stage.warn_if_exists:
            if (REPO_ROOT / rel).exists():
                print(f"NOTE: {rel} already exists and stage '{stage.key}' appends to it.")
                print("      Move it aside first if you want a clean rebuild.\n")

    if args.dry_run:
        print("Dry run — would execute:")
        for stage in selected:
            print(f"  {stage.key:14s} {stage.notebook}")
        return 0

    # --- run ---
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []

    for index, stage in enumerate(selected, start=1):
        header = f"[{index}/{len(selected)}] {stage.key} — {stage.title}"
        print(header)
        print("-" * len(header))
        before = snapshot(stage.produces)
        log_path = LOG_DIR / f"{stamp}_{stage.key}.log"
        started = time.time()
        try:
            run_notebook(REPO_ROOT / stage.notebook, args.timeout, log_path)
        except Exception as exc:                                  # noqa: BLE001
            elapsed = time.time() - started
            print(f"  FAILED after {elapsed/60:.1f} min: {type(exc).__name__}: {exc}")
            print(f"  Cell output written to {log_path.relative_to(REPO_ROOT)}")
            print("\nStopping: later stages read this stage's files, so continuing would")
            print("produce a wrong answer rather than an obvious failure.")
            results.append((stage.key, "FAILED", elapsed))
            _summarise(results)
            return 1

        elapsed = time.time() - started
        after = snapshot(stage.produces)
        stale = [rel for rel in stage.produces
                 if after[rel] is None or after[rel] == before[rel]]
        if stale:
            print(f"  completed in {elapsed/60:.1f} min, but these outputs did not change:")
            for rel in stale:
                print(f"    - {rel} ({'missing' if after[rel] is None else 'unchanged'})")
            print("  Treating this as a failure — the stage ran but produced nothing new.")
            results.append((stage.key, "NO OUTPUT", elapsed))
            _summarise(results)
            return 1

        print(f"  ok in {elapsed/60:.1f} min")
        for rel in stage.produces:
            size_mb = (REPO_ROOT / rel).stat().st_size / 1e6
            print(f"    wrote {rel} ({size_mb:.1f} MB)")
        results.append((stage.key, "ok", elapsed))
        print()

    warnings = check_finbert_duplicates()
    if warnings:
        print("Post-run warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    _summarise(results)
    print("\nNext: re-run the modelling notebooks against the refreshed data.")
    return 0


def _summarise(results: list[tuple[str, str, float]]) -> None:
    print("Summary")
    print("-------")
    for key, status, elapsed in results:
        print(f"  {key:14s} {status:10s} {elapsed/60:6.1f} min")
    total = sum(r[2] for r in results)
    print(f"  {'TOTAL':14s} {'':10s} {total/60:6.1f} min")


if __name__ == "__main__":
    sys.exit(main())

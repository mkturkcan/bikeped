"""Command line interface: ``python -m bikeped.scenarios COMMAND``.

Run from the repository root so that ``decision_testbench`` is importable.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import all_scenarios, get_scenario, get_suite, list_suites


def _selected(suite: str | None) -> list[dict]:
    return get_suite(suite) if suite else all_scenarios()


def _cmd_list(args: argparse.Namespace) -> int:
    rows = [s for s in _selected(args.suite) if not args.family or s.get('family') == args.family]
    if args.json:
        from .catalog import scenario_record

        print(json.dumps([{k: v for k, v in scenario_record(s).items() if k != 'agents'}
                          for s in rows], indent=1))
        return 0
    import decision_testbench as tb

    print(f"{'id':<36} {'suite':<12} {'family':<21} {'agents':>6} {'dur s':>6}  expected")
    for s in rows:
        print(f"{s['id']:<36} {s.get('suite', ''):<12} {s.get('family', ''):<21} "
              f"{len(s['agents']):>6} {tb.get_scenario_duration(s):>6.1f}  "
              f"{s['expected_dominant']}")
    print(f'{len(rows)} scenarios')
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        s = get_scenario(args.id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    print(json.dumps(s, indent=1))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .schema import validate_suites

    scenarios = _selected(args.suite)
    problems = validate_suites(scenarios)
    for p in problems:
        print(p)
    print(f'{len(scenarios)} scenarios checked, {len(problems)} problems')
    return 1 if problems else 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import decision_testbench as tb

    from .evaluate import evaluate_suite, format_table, to_json

    if args.config not in tb.PIPELINE_CONFIGS:
        print(f'unknown config {args.config!r}; one of {", ".join(tb.PIPELINE_CONFIGS)}',
              file=sys.stderr)
        return 2
    suites = [args.suite] if args.suite else list_suites()
    rows = [r for name in suites for r in evaluate_suite(get_suite(name), args.config)]
    print(to_json(rows) if args.json else format_table(rows))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .generators import generate

    scenarios = generate(args.family, args.n, args.seed)
    text = json.dumps(scenarios, indent=1)
    if args.out == '-':
        print(text)
    else:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')
        print(f'wrote {len(scenarios)} scenarios to {args.out}')
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .catalog import export_catalog, export_simulator

    info = export_catalog(args.out)
    print(f"wrote {info['catalog']} and {info['thumbnails']} thumbnails")
    if args.simulator:
        print(f'wrote {export_simulator(args.simulator)}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all subcommands."""
    from .generators import FAMILIES

    p = argparse.ArgumentParser(
        prog='python -m bikeped.scenarios',
        description='List, check, evaluate, generate and export bike-pedestrian scenarios. '
                    'Suites: ' + ', '.join(list_suites()) + '.')
    sub = p.add_subparsers(dest='command', required=True, metavar='COMMAND')

    sp = sub.add_parser('list', help='list scenarios with suite, family, agents and duration')
    sp.add_argument('--suite', help='only this suite')
    sp.add_argument('--family', help='only this family')
    sp.add_argument('--json', action='store_true', help='print metadata as JSON')
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser('show', help='print one scenario as JSON')
    sp.add_argument('id', metavar='ID',
                    help="slug, exact name, or suite:index with a zero-based index "
                         "(e.g. conformance:7)")
    sp.set_defaults(func=_cmd_show)

    sp = sub.add_parser('validate', help='check format, speeds and unique ids; exit 1 on problems')
    sp.add_argument('--suite', help='only this suite (ids are checked for uniqueness within it)')
    sp.set_defaults(func=_cmd_validate)

    sp = sub.add_parser(
        'evaluate', help='run the decision pipeline with ideal perception and summarise',
        description='Simulate each scenario with decision_testbench.simulate_scenario and print '
                    'ground-truth danger time, alert time, false-alert time, the lead of the '
                    'alert over each danger interval (negative means late) and the dominant '
                    'state against the expected one.')
    sp.add_argument('--suite', help='only this suite (default: all registered suites)')
    sp.add_argument('--config', default='default',
                    help='key of decision_testbench.PIPELINE_CONFIGS (default: default)')
    sp.add_argument('--json', action='store_true', help='print rows as JSON')
    sp.set_defaults(func=_cmd_evaluate)

    sp = sub.add_parser('generate', help='write a seeded parametric suite as JSON',
                        description='Families: ' + '; '.join(f'{k}: {v}'
                                                             for k, v in FAMILIES.items()))
    sp.add_argument('--family', required=True, choices=list(FAMILIES))
    sp.add_argument('--n', type=int, default=20, help='number of scenarios (default 20)')
    sp.add_argument('--seed', type=int, default=0, help='random seed (default 0)')
    sp.add_argument('--out', default='-', help='output JSON file, - for stdout (default)')
    sp.set_defaults(func=_cmd_generate)

    sp = sub.add_parser('export', help='write catalog.json and one SVG thumbnail per scenario')
    sp.add_argument('--out', required=True, help='output directory')
    sp.add_argument('--simulator', metavar='DIR',
                    help='also write DIR/scenarios.extended.js for the browser simulator')
    sp.set_defaults(func=_cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen command; returns the exit status."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))

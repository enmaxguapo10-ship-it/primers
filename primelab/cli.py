#!/usr/bin/env python3
"""
PrimeLab CLI — Advanced Prime Research from the command line.
Usage: python cli.py [command] [options]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import numpy as np


def cmd_generate(args):
    from core.sieve import get_primes
    print(f"[PrimeLab] Generating primes in [{args.low:,}, {args.high:,}]...")
    t0 = time.time()
    primes = get_primes(args.low, args.high, use_cache=not args.no_cache)
    elapsed = time.time() - t0
    print(f"  ✓ Found {len(primes):,} primes in {elapsed:.3f}s")
    print(f"  First 10: {primes[:10].tolist()}")
    print(f"  Last 5:   {primes[-5:].tolist()}")

    if args.output:
        np.save(args.output, primes)
        print(f"  Saved to: {args.output}")
    return primes


def cmd_analyze(args):
    from core.sieve import get_primes
    from analysis.math_analysis import (
        gap_statistics, test_cramer_model, test_poisson_model,
        find_twin_primes, pi_comparison, maximal_gaps, anomaly_detection
    )
    primes = get_primes(args.low, args.high)
    print(f"\n[PrimeLab] Analysis: {len(primes):,} primes in [{args.low:,}, {args.high:,}]\n")

    # Gap stats
    gs = gap_statistics(primes)
    print("═" * 50)
    print("  GAP STATISTICS")
    print(f"  Mean gap:       {gs['gap_mean']:.4f}")
    print(f"  Std gap:        {gs['gap_std']:.4f}")
    print(f"  Max gap:        {gs['gap_max']} (at p={gs['gap_max_at_prime']:,})")
    print(f"  Most common:    {gs['most_common_gap']}")

    # Twins
    twins = find_twin_primes(primes)
    print(f"\n  Twin primes:    {len(twins):,} pairs")

    # Cramér test
    cr = test_cramer_model(primes)
    print(f"\n═══ CRAMÉR MODEL (Exp(1)) ═══")
    print(f"  KS statistic:   {cr['ks_statistic']:.6f}")
    print(f"  p-value:        {cr['p_value']:.6f}")
    print(f"  Result:         {cr['interpretation']}")

    # π(x) at top
    pi_d = pi_comparison(primes, n_points=5)
    x_top = int(primes[-1])
    pi_exact = int(pi_d['pi_exact'][-1])
    pi_li = float(pi_d['pi_li'][-1])
    pi_log = float(pi_d['pi_log'][-1])
    print(f"\n═══ π({x_top:,}) ═══")
    print(f"  Exact:    {pi_exact:,}")
    print(f"  Li(x):    {pi_li:.1f}  (err: {abs(pi_exact - pi_li):.1f})")
    print(f"  x/ln(x): {pi_log:.1f}  (err: {abs(pi_exact - pi_log):.1f})")

    # Anomalies
    anom = anomaly_detection(primes, 3.5)
    print(f"\n═══ ANOMALIES (z > 3.5) ═══")
    print(f"  Count: {anom['n_anomalies']}")
    if anom['n_anomalies'] > 0:
        for p, g, z in zip(anom['anomaly_primes'][:5],
                           anom['anomaly_gaps'][:5],
                           anom['anomaly_z_scores'][:5]):
            print(f"  p={p:,}  gap={g}  z={z:.2f}")


def cmd_experiment(args):
    from experiments.runner import PRESET_EXPERIMENTS, run_experiment

    if args.list:
        print("Available presets:")
        for k, v in PRESET_EXPERIMENTS.items():
            print(f"  {k:<25} — {v.description}")
        return

    preset_name = args.preset
    if preset_name not in PRESET_EXPERIMENTS:
        print(f"Unknown preset: {preset_name}")
        print("Use --list to see available presets")
        return

    preset = PRESET_EXPERIMENTS[preset_name]
    print(f"[PrimeLab] Running experiment: {preset.name}")

    def progress(msg, pct):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}% {msg}", end="", flush=True)

    result = run_experiment(preset, progress)
    print("\n")
    summary = result.to_summary()
    print(json.dumps(summary, indent=2, default=str))
    if not args.no_save:
        saved = result.save()
        print(f"\nSaved to: {saved}")


def cmd_check(args):
    from core.sieve import is_prime_miller_rabin
    n = args.n
    result = is_prime_miller_rabin(n)
    print(f"{n:,} is {'PRIME ✓' if result else 'COMPOSITE ✗'} (Miller-Rabin)")


def main():
    parser = argparse.ArgumentParser(
        description="PrimeLab CLI — Prime Research Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  generate   Generate primes in a range
  analyze    Full analysis of a range
  experiment Run a preset experiment
  check      Test primality of a number
  ui         Launch the Streamlit web UI

Examples:
  python cli.py generate --low 2 --high 1000000
  python cli.py analyze --low 2 --high 500000
  python cli.py experiment --preset gap_deep_dive
  python cli.py check --n 982451653
  python cli.py ui
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate primes")
    p_gen.add_argument("--low", type=int, default=2)
    p_gen.add_argument("--high", type=int, default=1000000)
    p_gen.add_argument("--output", type=str, default=None, help="Save to .npy file")
    p_gen.add_argument("--no-cache", action="store_true")

    # analyze
    p_an = subparsers.add_parser("analyze", help="Full mathematical analysis")
    p_an.add_argument("--low", type=int, default=2)
    p_an.add_argument("--high", type=int, default=100000)

    # experiment
    p_ex = subparsers.add_parser("experiment", help="Run experiments")
    p_ex.add_argument("--preset", type=str, default="quick_survey")
    p_ex.add_argument("--list", action="store_true")
    p_ex.add_argument("--no-save", action="store_true")

    # check
    p_ch = subparsers.add_parser("check", help="Primality test")
    p_ch.add_argument("--n", type=int, required=True)

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch web UI")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "experiment":
        cmd_experiment(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "ui":
        import subprocess
        subprocess.run(["bash", "run.sh"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Small declarative entry points; accelerator imports are operation-local."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
from .artifacts import runtime_identity, write_json


def _parser():
    parser = argparse.ArgumentParser(prog="posteriorforge")
    parser.add_argument("--version", action="version", version="PosteriorForge 0.1.0")
    commands = parser.add_subparsers(dest="operation", required=True)
    doctor = commands.add_parser("doctor", help="inspect environment without downloading models")
    doctor.add_argument("--cuda", action="store_true")
    doctor.add_argument("--output")
    banks = commands.add_parser("prepare-banks", help="create three prior banks and shared held-out cases")
    banks.add_argument("--output", required=True)
    reference = commands.add_parser("reference", help="refine a float64 GPU reference in an accounted allocation")
    reference.add_argument("--banks", required=True)
    reference.add_argument("--case", type=int, required=True)
    reference.add_argument("--output", required=True)
    reference.add_argument("--deadline-seconds", type=int, default=7200)
    reference.add_argument("--batch-size", type=int, default=512)
    qualify = commands.add_parser("qualify-reference", help="GPU float64 versus independent CPU algebra")
    qualify.add_argument("--banks", required=True)
    qualify.add_argument("--output", required=True)
    inference = commands.add_parser("inference", help="run unchanged upstream; full scientific runs require 50000 rows")
    inference.add_argument("--banks", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--context-seed", type=int, required=True)
    inference.add_argument("--case", type=int, required=True)
    inference.add_argument("--context-size", type=int, choices=(256,1024,4096), default=4096)
    inference.add_argument("--count", type=int, default=50000)
    inference.add_argument("--microbatch", type=int, choices=(1024,4096,10000), default=10000)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--checkpoint-sha256", required=True)
    inference.add_argument("--repetition", type=int, default=0)
    budget = commands.add_parser("budget", help="inspect/reconcile the single hard-cap ledger")
    budget.add_argument("--ledger", required=True)
    budget.add_argument("--reconcile", action="store_true")
    submit = commands.add_parser("submit", help="reserve allocation cost before sbatch")
    submit.add_argument("--ledger", required=True)
    submit.add_argument("--script", required=True)
    submit.add_argument("--gpus", type=int, choices=(1,2,4), required=True)
    submit.add_argument("--walltime-seconds", type=int, required=True)
    submit.add_argument("--purpose", required=True)
    submit.add_argument("--job-name", required=True)
    submit.add_argument("--log-directory", required=True)
    submit.add_argument("script_args", nargs=argparse.REMAINDER)
    plan = commands.add_parser("plan", help="expand a declarative campaign without submitting jobs")
    plan.add_argument("--config", required=True)
    plan.add_argument("--banks", required=True)
    plan.add_argument("--references", required=True)
    plan.add_argument("--runs", required=True)
    gate = commands.add_parser("gate", help="assess all 24 complete baseline runs and retained references")
    gate.add_argument("--banks", required=True)
    gate.add_argument("--references", required=True)
    gate.add_argument("--runs", required=True)
    gate.add_argument("--output", required=True)
    report = commands.add_parser("report", help="render figures and portable evidence from measured artifacts")
    report.add_argument("--gate", required=True)
    report.add_argument("--ledger", required=True)
    report.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = vars(_parser().parse_args(argv))
    operation = args.pop("operation")
    if operation == "doctor":
        cuda, output = args["cuda"], args["output"]
        result = runtime_identity()
        result["environment"] = {name: os.getenv(name) for name in
                                 ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PF_SCRATCH", "OMP_NUM_THREADS")}
        if cuda:
            if not os.getenv("SLURM_JOB_ID"):
                raise RuntimeError("CUDA qualification requires an accounted allocation")
            import torch
            result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        if output: write_json(output, result)
    elif operation == "prepare-banks":
        from .banks import prepare_banks
        result = prepare_banks(args["output"])
    elif operation == "reference":
        from .reference import prepare_reference
        result = prepare_reference(args["banks"], args["case"], args["output"],
                                   deadline_seconds=args["deadline_seconds"], batch_size=args["batch_size"])
    elif operation == "qualify-reference":
        from .reference import qualify_reference
        result = qualify_reference(args["banks"], 0, args["output"])
    elif operation == "inference":
        from .inference import run_baseline
        args["context_seed"] = args.pop("context_seed")
        result = run_baseline(**args)
    elif operation == "budget":
        from .budget import BudgetLedger
        ledger = BudgetLedger(args["ledger"])
        result = {"reconciliation": ledger.reconcile_all(), "summary": ledger.summary()} if args["reconcile"] else ledger.summary()
    elif operation == "submit":
        from .budget import BudgetLedger
        script_args = args.pop("script_args")
        ledger = BudgetLedger(args.pop("ledger"))
        if script_args[:1] == ["--"]: script_args = script_args[1:]
        result = ledger.submit(**args, script_args=script_args)
    elif operation == "plan":
        from .campaign import campaign_plan
        result = campaign_plan(args.pop("config"), **args)
    elif operation == "gate":
        from .gate import evaluate_gate
        result = evaluate_gate(**args)
    elif operation == "report":
        from .report import render_report
        result = render_report(**args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))

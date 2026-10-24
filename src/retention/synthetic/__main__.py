"""`python -m retention.synthetic {generate,pipeline,all}`: the synthetic dataset and its path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retention.synthetic.generate import generate
from retention.synthetic.pipeline import (
    GENERATED,
    Setup,
    labels_and_features,
    run_all,
    stage,
    train,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m retention.synthetic")
    p.add_argument("command", choices=["generate", "pipeline", "all", "drift"])
    p.add_argument("--out", type=Path, default=GENERATED)
    p.add_argument("--defaults", action="store_true", help="drift: log the library defaults only")
    args = p.parse_args(argv)
    if args.command == "generate":
        print(json.dumps(generate(args.out), indent=1))
    elif args.command == "pipeline":
        setup = Setup()
        stage(setup, args.out)
        print(
            json.dumps(
                {"labels_and_features": labels_and_features(setup), "train": train(setup)}, indent=1
            )
        )
    elif args.command == "drift":
        from retention.ops import drift as dr
        from retention.synthetic import drift as sd

        setup = Setup()
        if args.defaults:
            print(dr.log_defaults(setup.processed / "ops").read_text())
        else:
            r = sd.run(setup)
            print(json.dumps({k: r[k] for k in ("margin", "batch")}, indent=1, default=float))
    else:
        print(json.dumps(run_all(generated=args.out), indent=1))


if __name__ == "__main__":
    main()

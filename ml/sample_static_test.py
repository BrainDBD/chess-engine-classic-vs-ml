import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Apply fixed-seed sample-rate filtering to a JSONL file."
    )
    ap.add_argument("--input", default="may.jsonl")
    ap.add_argument("--output", default="may_static-test.jsonl")
    ap.add_argument("--sample-rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 <= args.sample_rate <= 1.0:
        raise SystemExit("[fatal] --sample-rate must be between 0 and 1")

    inp = Path(args.input)
    out = Path(args.output)

    rng = random.Random(args.seed)

    seen = kept = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            seen += 1
            if rng.random() < args.sample_rate:
                fout.write(line)
                kept += 1

    print(f"Read {seen} rows from {inp}")
    print(f"Kept {kept} rows ({kept / seen:.2%} actual rate)" if seen else "Kept 0 rows")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
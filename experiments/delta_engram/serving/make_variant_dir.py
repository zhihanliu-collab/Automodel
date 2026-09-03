#!/usr/bin/env python3
"""Make a serving-dir variant: symlink an exported Delta dir and patch text_config keys.

    python make_variant_dir.py --src .../serving/s329 --dst .../serving/s329_xh \
        --set delta_exact_hot_keys_path=/mnt/data/zhihan/delta-engram/serving/exact_hot_keys_train.pt
    python make_variant_dir.py --src .../serving/s329 --dst .../serving/s329_a03 --set delta_output_scale=0.3
"""

import argparse
import json
import os


def parse(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "null": None}.get(v.lower(), v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--set", action="append", default=[], help="text_config key=value")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for entry in sorted(os.listdir(args.src)):
        if entry == "config.json":
            continue
        dst = os.path.join(args.dst, entry)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(args.src, entry)), dst)
    config = json.load(open(os.path.join(args.src, "config.json")))
    for kv in args.set:
        k, v = kv.split("=", 1)
        config["text_config"][k] = parse(v)
        print(f"text_config.{k} = {config['text_config'][k]!r}")
    json.dump(config, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
    print("variant at", args.dst)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a Polar topology template for VERL + Polar examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from verl_polar_bridge.config import render_topology_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="Path to Polar topology template")
    parser.add_argument("--router-base-url", required=True, help="VERL/SGLang router base URL")
    parser.add_argument("--output", "-o", help="Output YAML path; defaults to stdout")
    args = parser.parse_args()

    trainer_config = {"polar": {"router_base_url": args.router_base_url}}
    rendered = render_topology_template(args.template, trainer_config)
    text = yaml.safe_dump(json.loads(json.dumps(rendered)), sort_keys=False, allow_unicode=True)
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

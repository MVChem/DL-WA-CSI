#!/usr/bin/env python3
"""Compatibility entry point; prefer ``dlwa-infer`` after installation."""

from dlwa_csi.inference import main

if __name__ == "__main__":
    raise SystemExit(main())

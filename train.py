#!/usr/bin/env python3
"""Compatibility entry point; prefer ``dlwa-train`` after installation."""

from dlwa_csi.training import main

if __name__ == "__main__":
    raise SystemExit(main())

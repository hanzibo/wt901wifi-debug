#!/usr/bin/env python3
"""Run from OptUSBProj-yolo. No other GUI should own the Gemini device."""
from optusbproj_ubuntu.hand_demo import main

if __name__=='__main__':
    raise SystemExit(main())

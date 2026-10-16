#!/usr/bin/env python3
"""Standalone tag pose viewer; does not import or open robot transports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OptUSBProj-ubuntu'))


def main():
    from PySide6 import QtWidgets
    from apriltag_pose.gui import Window
    app = QtWidgets.QApplication(sys.argv[:1])
    window = Window()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())

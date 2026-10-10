"""Opt-in stopped-scene hand test. Use an isolated simulator port."""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'OptUSBProj-ubuntu')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=23001)
    args=parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets
    from optusbproj_ubuntu.coppeliasim_bridge import SceneBridge
    from optusbproj_ubuntu.scene_robot_gui import SceneRobotDialog
    from optusbproj_ubuntu.pose import PoseEngine
    app=QtWidgets.QApplication([])
    b=SceneBridge(args.port);original=b.exchange();hands={s:q.copy() for s,q in b.hand_angles.items()}
    host=QtWidgets.QWidget();host.engine=PoseEngine();dialog=SceneRobotDialog(host)
    try:
        dialog.port.setValue(args.port);dialog.connect_scene()
        deadline=time.monotonic()+12
        while not dialog.connected and time.monotonic()<deadline:
            app.processEvents();time.sleep(.02)
        assert dialog.connected,dialog.status.text()
        dialog.show_hand_config();form=dialog.hand_dialog
        for side in ('left','right'):
            for pair in form.widgets[side][0]:pair[0].setValue(0);pair[1].setValue(5)
            form.confirm(side,'open');form.confirm(side,'closed')
        # Synthetic discrete command only; no hardware/gesture-estimation claim.
        deadline=time.monotonic()+.6
        while time.monotonic()<deadline:
            for side in ('left','right'):
                dialog.receive_hand(side,{'enabled':True,'valid':True,'state':True,'stamp':time.monotonic()})
            app.processEvents();time.sleep(.02)
        actual=b.exchange()
        for side in hands:
            np.testing.assert_allclose(actual[side],original[side],atol=1e-8)
            np.testing.assert_allclose(b.hand_angles[side],np.deg2rad(np.full(15,5)),atol=1e-7)
        assert not dialog.mailbox.get()[0]
        print('PASS: bilateral 30-finger GUI -> one worker -> real simulator set/readback; arms unchanged')
        dialog.stop_hands()
        held={s:q.copy() for s,q in b.hand_angles.items()}
        deadline=time.monotonic()+.4
        while time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        b.exchange()
        for side in hands:np.testing.assert_allclose(b.hand_angles[side],held[side],atol=1e-8)
        print('PASS: explicit stop holds both hands')
        form.resize(1000,700);form.grab().save('/tmp/coppeliasim-hand-config.png')
    finally:
        dialog.close();app.processEvents()
        b.exchange(hand_targets=hands)
        b.close();host.close()
        print('Restored original hand angles; scene not saved')


if __name__=='__main__':main()

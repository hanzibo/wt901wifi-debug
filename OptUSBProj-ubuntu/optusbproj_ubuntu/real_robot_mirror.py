"""Optional stopped-scene mirror of completed host writes; never an IK source."""
import threading
import numpy as np
from PySide6.QtCore import QThread, Signal
from .real_robot_arm import load_profile
from .real_robot_protocol import decode
from .scene_robot_model import SceneArm


def mirror_targets(frame, signs):
    angles, _ = decode(frame)
    signs = signs if isinstance(signs, dict) else {'left':signs, 'right':(1,)*6}
    return {s:np.deg2rad((np.asarray(angles[start:start+6])-90)/np.asarray(signs[s]))
            for s,start in (('right',0),('left',6))}


class CommandMirror(QThread):
    ready = Signal()
    failed = Signal(str)

    def __init__(self, port=23000, parent=None, bridge_factory=None, sides=('left',)):
        super().__init__(parent)
        self.port, self.factory = port, bridge_factory
        self.sides = sides
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False

    def submit(self, frame, signs):
        with self.condition:
            if not self.stopping:
                self.pending = (frame, signs)
                self.condition.notify()

    def shutdown(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify()

    def run(self):
        bridge = None
        try:
            from .coppeliasim_bridge import SceneBridge
            bridge = (self.factory or SceneBridge)(self.port)
            for side in self.sides:
                arm, _, _ = load_profile(side)
                nominal = SceneArm(arm, [-np.pi]*6, [np.pi]*6)
                loaded = SceneArm(bridge.data['arms'][side], [-np.pi]*6, [np.pi]*6)
                # Compare FK, not saved pose: q_ref may differ legitimately.
                for q in (np.zeros(6), np.array([.1,.1,.1,-.4,.1,.1])):
                    for i in range(6):
                        if not np.allclose(nominal.frame(q, nominal.homes[i], i),
                                           loaded.frame(q, loaded.homes[i], i), atol=1e-4):
                            raise ValueError('场景几何与真机名义配置不同；请加载 NowRobotWithHand1.ttt')
                    if not np.allclose(nominal.tool_frame(q),loaded.tool_frame(q),atol=1e-4):
                        raise ValueError('场景手掌几何与真机名义配置不同')
            self.ready.emit()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopping or self.pending is not None)
                    if self.stopping:
                        return
                    frame, signs = self.pending
                    self.pending = None
                bridge.exchange(mirror_targets(frame, signs))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if bridge is not None:
                bridge.close()

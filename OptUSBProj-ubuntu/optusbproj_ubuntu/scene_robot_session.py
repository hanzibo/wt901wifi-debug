"""Own one scene connection; recover by reading, never replay uncertain writes."""
import time
import numpy as np
import zmq

from .scene_robot_model import SceneArm, body_axes, matrix


class SceneSession:
    def __init__(self, port, mailbox, hand_mailbox, report, interrupted, sleep_ms,
                 *, bridge_factory=None, clock=time.monotonic):
        if bridge_factory is None:
            from .coppeliasim_bridge import SceneBridge
            bridge_factory = SceneBridge
        self.port = port
        self.mailbox, self.hand_mailbox = mailbox, hand_mailbox
        self.report, self.interrupted, self.sleep_ms = report, interrupted, sleep_ms
        self.bridge_factory, self.clock = bridge_factory, clock
        self.bridge = None
        self.models = {}
        self.log = None
        self.last_hand_exchange = 0.

    def connect(self, angle):
        self.bridge = self.bridge_factory(self.port)
        for side, data in self.bridge.data["arms"].items():
            limits = np.array(data["limits"]).T
            lower = limits[0] if angle is None else np.maximum(limits[0], -np.deg2rad(angle))
            upper = limits[1] if angle is None else np.minimum(limits[1], np.deg2rad(angle))
            self.models[side] = SceneArm(data, lower, upper)
        axes = body_axes(self.models)
        previous = self.bridge.exchange()
        self.last_hand_exchange = self.clock()
        if hasattr(self.bridge, "fast_idle"):
            self.bridge.fast_idle(True)
        return axes, previous

    def record(self, row):
        if self.log is not None:
            self.log.write(row)

    def exchange(self, targets=None):
        fingers = {}
        try:
            now = self.clock()
            fingers, requests, revision = self.hand_mailbox.targets(
                getattr(self.bridge, "hand_angles", {}), now, now - self.last_hand_exchange)
            if self.hand_mailbox.snapshot()[1] != revision:
                fingers, requests = {}, {}
            actual = (self.bridge.exchange(targets, hand_targets=fingers)
                      if fingers else self.bridge.exchange(targets))
            self.last_hand_exchange = self.clock()
            hand_angles = getattr(self.bridge, "hand_angles", {})
            if any(not np.allclose(hand_angles[s], q, atol=1e-5, rtol=0)
                   for s, q in fingers.items()):
                raise ValueError("场景未应用手指目标，请检查其他控制器")
            row = {
                "hand_angles": {s: q.tolist() for s, q in hand_angles.items()},
                "hand_targets": {s: q.tolist() for s, q in fingers.items()},
                "hand_requests": {s: {**r, "target": r["target"].tolist()}
                                  for s, r in requests.items()},
                "hand_readback_mono_s": self.last_hand_exchange,
            }
            if fingers or requests:
                self.record({"event": "hands", "mono_s": self.last_hand_exchange, **row})
            if hand_angles:
                self.report(row)
            return actual
        except zmq.Again:
            return self.recover(bool(targets or fingers))

    def recover(self, write_uncertain):
        self.mailbox.pause()
        self.hand_mailbox.stop()
        self.report({"recovering": True})
        self.record({"event": "rpc_timeout", "mono_s": self.clock(),
                     "write_uncertain": write_uncertain})
        old = self.bridge.data
        self.close()
        for _ in range(2):
            if self.interrupted():
                return None
            try:
                self.bridge = self.bridge_factory(self.port)
                self.bridge.validate_identity(old)
                self.validate_geometry()
                actual = self.bridge.exchange()
                if hasattr(self.bridge, "fast_idle"):
                    self.bridge.fast_idle(True)
                self.report({"recovered": True, "angles": actual})
                self.record({"event": "rpc_recovered", "mono_s": self.clock()})
                return actual
            except zmq.Again:
                self.close()
                self.sleep_ms(200)
        raise RuntimeError("通信超时：两次重连未恢复，请检查 CoppeliaSim 弹窗或繁忙状态")

    def validate_geometry(self):
        # Keep the initial PoE references, including after reconnection.
        for side, model in self.models.items():
            fresh = self.bridge.data["arms"][side]
            q = np.array(fresh["q"])
            for i, raw in enumerate(fresh["joints"]):
                if not np.allclose(model.frame(q, model.homes[i], i), matrix(raw),
                                   atol=1e-6, rtol=0):
                    raise ValueError("恢复失败：机器人几何改变，请重新连接")
            if not np.allclose(model.tool_frame(q)[:3, :3], matrix(fresh["palm"])[:3, :3],
                               atol=1e-6, rtol=0):
                raise ValueError("恢复失败：工具系改变，请重新连接")

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

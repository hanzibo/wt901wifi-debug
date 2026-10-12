"""BodyView extracted from the main window; compatibility state stays on the host."""
from __future__ import annotations

import time
import pyvista as pv
from .skeleton import PARTS


class BodyView:
    def __init__(self, window, *, model_dir, triad_factory):
        self.window = window
        self.model_dir = model_dir
        self.triad_factory = triad_factory

    def _load_model(self) -> None:
        window = self.window
        missing = [part.mesh for part in PARTS if not (self.model_dir / part.mesh).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少人体 STL 模型：{', '.join(missing)}")
        triad = self.triad_factory(0.105)
        for part in PARTS:
            mesh = pv.read(self.model_dir / part.mesh)
            mesh.points *= 0.001  # Original mesh units are millimetres.
            actor = window.plotter.add_mesh(mesh, color=part.color,
                                          smooth_shading=False, show_edges=False)
            window.actors.append(actor)
            axis_actor = window.plotter.add_mesh(
                triad, scalars="axis_rgb", rgb=True, lighting=False,
                show_scalar_bar=False, pickable=False)
            window.axis_actors.append(axis_actor)
        window.plotter.set_background("#1a1d23")
        window.plotter.show_axes()
        window._render_pose()
        window._front_view()


    def _front_view(self) -> None:
        window = self.window
        window.plotter.camera_position = [(2.5, 0.0, 0.15), (0.0, 0.0, 0.15), (0.0, 0.0, 1.0)]
        window.plotter.reset_camera()


    def _render_pose(self) -> None:
        window = self.window
        for actor, axis_actor, matrix in zip(
                window.actors, window.axis_actors, window.engine.transforms()):
            actor.user_matrix = matrix
            axis_actor.user_matrix = matrix
        window.plotter.render()
        window._render_dirty = False
        window._last_render = time.monotonic()


    def _toggle_axes(self, visible: bool) -> None:
        window = self.window
        for actor in window.axis_actors:
            actor.SetVisibility(visible)
        window._render_dirty = True


"""Conservative convex-envelope self-collision queries; RViz only.

Native dependencies are deliberately isolated from the ROS interpreter.
Meshes are enclosed by convex hulls or spatially partitioned compound hulls.
False positives are still possible. Motion
certification uses joint-space subdivision and conservative displacement bounds,
not interpolation of endpoint link transforms. Unknown paths are rejected.
"""
from pathlib import Path
import hashlib
import itertools
import time
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, QhullError


URDF = Path(__file__).resolve().parents[1] / 'pkg_robot_model/urdf/pkg_robot_model.urdf'
# Explicit assembly contacts: rotating bearings join these physical housings.
# Fixed accessories form one rigid group. No zero-pose-derived exemptions.
ASSEMBLY_JOINTS = frozenset(('101', '102', '1', '2', '3', '4', '5', '6', '7',
                            '51', '52', '53', '54', '55', '56', '57'))
ARM_LIMITS = dict(zip(('1', '2', '3', '4', '5', '6'),
                     (110, 85, 90, 125, 90, 45)))
ARM_LIMITS.update({str(int(k) + 50): v for k, v in list(ARM_LIMITS.items())})
# The previous whole-hull model had only ~0.28/0.33 mm zero-pose clearance
# at these pairs. Retain its 0.1 mm pair tolerance; do not exempt shoulders
# or globally relax margins to compensate for convex-envelope errors. NOT
# suitable as a physical robot clearance policy.
PAIR_MARGINS = {frozenset(('base_link_0', 'L2')): .0001,
                frozenset(('base_link_0', 'R52')): .0001}
COMPOUND_LINKS = frozenset(('base_link_0', 'L2', 'R52'))
COMPOUND_DEPTH = 6


def partition_surface(mesh, depth=COMPOUND_DEPTH):
    """Split every CAD component, clipping crossing triangles, never discarding them.

    Convex envelopes of the clipped pieces also enclose the cut-face interiors
    of closed solid components. Unclosed CAD components remain a surface-based
    assumption, as in the original URDF; this is not a certified solid model.
    Do not cluster triangle centroids and drop boundary triangles.
    """
    mesh.remove_unreferenced_vertices()
    if depth == 0 or len(mesh.faces) < 4:
        return [mesh]
    normal = np.eye(3)[int(np.argmax(mesh.extents))]
    center = mesh.bounds.mean(axis=0)
    children = []
    for direction in (normal, -normal):
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(
            mesh.vertices, mesh.faces, direction, center)
        if len(faces):
            # Close each new cut conservatively with a convex planar cap.
            # Without caps, later cuts can discard an entirely interior cell
            # (no original exterior triangles) and leave holes in the solid
            # envelope. Convex caps may fill cut-section concavities, never
            # undercut them. No shapely/triangulation dependency is required.
            used = np.unique(faces)
            boundary = used[np.abs((vertices[used]-center)@direction) < 1e-8]
            axes = np.flatnonzero(normal == 0)
            if len(boundary) >= 3:
                try:
                    ring = boundary[ConvexHull(vertices[boundary][:,axes]).vertices]
                    caps = np.array([[ring[0],ring[i],ring[i+1]]
                                     for i in range(1,len(ring)-1)])
                    triangle = vertices[caps[0]]
                    if np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])@direction > 0:
                        caps = caps[:,::-1]
                    faces = np.concatenate((faces,caps))
                except QhullError:
                    # Even an uncertain near-planar cut must not lose solid
                    # coverage. Keep the entire parent envelope instead.
                    return [mesh]
            children.extend(partition_surface(
                trimesh.Trimesh(vertices, faces, process=False), depth-1))
    if not children:
        # Fail conservatively rather than losing degenerate/on-plane geometry.
        return [mesh]
    return children


def convex_object(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if np.linalg.matrix_rank(vertices-vertices.mean(axis=0)) < 3:
        # Retain disconnected triangles/planar faces as 0.2 um thick envelopes.
        vertices = np.concatenate([vertices + sign*np.eye(3)[i]*1e-7
                                   for i in range(3) for sign in (-1, 1)])
    hull = trimesh.convex.convex_hull(vertices)
    points = np.asarray(hull.vertices, dtype=np.float64)
    faces = np.c_[np.full(len(hull.faces), 3), hull.faces].astype(np.int32).ravel()
    return fcl.CollisionObject(fcl.Convex(points, len(hull.faces), faces)), points


def compound_distance_callback(a, b, data):
    if time.monotonic() > data.deadline:
        data.timed_out = True
        return True, data.result.min_distance
    return fcl.defaultDistanceCallback(a, b, data)


def origin(element):
    out = np.eye(4)
    e = element.find('origin')
    if e is not None:
        out[:3, 3] = np.fromstring(e.get('xyz', '0 0 0'), sep=' ')
        out[:3, :3] = Rotation.from_euler('xyz', np.fromstring(
            e.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


class CollisionModel:
    def __init__(self, urdf=URDF, margin=.002, *, compound=True):
        self.margin = float(margin)
        if not 0 <= self.margin <= .05:
            raise ValueError('Invalid collision margin')
        urdf = Path(urdf)
        root = ET.parse(urdf).getroot()
        self.joints = {}
        children = set()
        self.by_parent = {}
        for j in root.findall('joint'):
            name, kind = j.get('name'), j.get('type')
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if kind not in ('fixed', 'continuous', 'revolute'):
                raise ValueError('Unsupported joint: ' + name)
            axis = np.zeros(3) if kind == 'fixed' else np.fromstring(j.find('axis').get('xyz'), sep=' ')
            if kind != 'fixed':
                axis /= np.linalg.norm(axis)
            self.joints[name] = (parent, child, kind, origin(j), axis)
            self.by_parent.setdefault(parent, []).append(name)
            children.add(child)
        roots = set(l.get('name') for l in root.findall('link')) - children
        if len(roots) != 1:
            raise ValueError('Expected a connected URDF tree')
        self.root = roots.pop()
        self.order, self.paths, groups = [], {self.root: []}, {self.root: self.root}
        def visit(link):
            for name in self.by_parent.get(link, []):
                _, child, kind, _, _ = self.joints[name]
                self.order.append(name)
                self.paths[child] = self.paths[link] + [name]
                groups[child] = groups[link] if kind == 'fixed' else child
                visit(child)
        visit(self.root)
        if len(self.order) != len(self.joints):
            raise ValueError('Disconnected URDF')
        allowed_groups = set()
        for name in ASSEMBLY_JOINTS:
            if name not in self.joints:
                continue
            parent, child, *_ = self.joints[name]
            allowed_groups.add(frozenset((groups[parent], groups[child])))
        self.shapes = []
        digest = hashlib.sha256(urdf.read_bytes())
        for link in root.findall('link'):
            for index, c in enumerate(link.findall('collision')):
                mesh = c.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Expected collision mesh')
                filename = mesh.get('filename')
                prefix = 'package://pkg_robot_model/'
                if not filename.startswith(prefix):
                    raise ValueError('Unsupported mesh URI')
                path = urdf.parent.parent / filename[len(prefix):]
                digest.update(path.read_bytes())
                m = trimesh.load(path, force='mesh')
                m.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
                m.apply_transform(origin(c))
                obj, vertices = convex_object(m.vertices)
                pieces = []
                piece_vertices = []
                if compound and link.get('name') in COMPOUND_LINKS:
                    for component in m.split(only_watertight=False):
                        for piece in partition_surface(component):
                            piece_obj, points = convex_object(piece.vertices)
                            pieces.append(piece_obj)
                            piece_vertices.append(points)
                            digest.update(points.tobytes())
                manager = None
                if pieces:
                    # Include the planar-fragment padding in the broad hull
                    # and displacement radii as well.
                    obj, vertices = convex_object(np.concatenate(piece_vertices))
                    manager = fcl.DynamicAABBTreeCollisionManager()
                    manager.registerObjects(pieces)
                    manager.setup()
                digest.update(vertices.tobytes())
                radii = {}
                # Triangle inequality bounds distance from each ancestor joint
                # to any collision point, valid at EVERY joint configuration.
                tail = float(np.linalg.norm(vertices, axis=1).max())
                for name in reversed(self.paths[link.get('name')]):
                    _, _, kind, O, _ = self.joints[name]
                    if kind != 'fixed':
                        radii[name] = tail
                    tail += float(np.linalg.norm(O[:3, 3]))
                # The last movable ancestor has only fixed transforms after it.
                # Its exact perpendicular radius is invariant under its own
                # rotation and tighter than the translation-sum radius.
                movable = [n for n in self.paths[link.get('name')]
                           if self.joints[n][2] != 'fixed']
                if movable:
                    last = movable[-1]
                    tail_frame = np.eye(4)
                    path_names = self.paths[link.get('name')]
                    for n in path_names[path_names.index(last)+1:]:
                        tail_frame = tail_frame @ self.joints[n][3]
                    points = vertices @ tail_frame[:3, :3].T + tail_frame[:3, 3]
                    axis = self.joints[last][4]
                    radii[last] = float(np.linalg.norm(
                        points - (points @ axis)[:, None]*axis, axis=1).max())
                self.shapes.append(dict(link=link.get('name'), index=index,
                    object=obj, vertices=vertices, radii=radii, pieces=pieces,
                    manager=manager, transform=None))
        self.pairs = []
        for i, j in itertools.combinations(range(len(self.shapes)), 2):
            a, b = self.shapes[i], self.shapes[j]
            ga, gb = groups[a['link']], groups[b['link']]
            if ga == gb or frozenset((ga, gb)) in allowed_groups:
                continue
            # Common ancestry moves both shapes rigidly; cancels from distance.
            ra, rb = a['radii'], b['radii']
            coefficients = {k: ra[k] for k in ra.keys() - rb.keys()}
            coefficients.update({k: rb[k] for k in rb.keys() - ra.keys()})
            self.pairs.append((i, j, coefficients))
        digest.update(repr(sorted(ASSEMBLY_JOINTS)).encode())
        self.margins = np.array([PAIR_MARGINS.get(frozenset((self.shapes[i]['link'],
            self.shapes[j]['link'])), self.margin) for i, j, _ in self.pairs])
        digest.update(repr(PAIR_MARGINS).encode())
        digest.update(str(self.margin).encode())
        digest.update(str(compound).encode())
        digest.update(Path(__file__).read_bytes())
        self.version = digest.hexdigest()

    def frames(self, q):
        frames = {self.root: np.eye(4)}
        for name in self.order:
            parent, child, kind, O, axis = self.joints[name]
            E = np.eye(4)
            if kind != 'fixed':
                E[:3, :3] = Rotation.from_rotvec(axis * q.get(name, 0.)).as_matrix()
            frames[child] = frames[parent] @ O @ E
        return frames

    def validate(self, q):
        moving = {n for n, j in self.joints.items() if j[2] != 'fixed'}
        if set(q) != moving or not all(type(v) in (int, float) and np.isfinite(v) for v in q.values()):
            raise ValueError('Incomplete/nonfinite full configuration')
        for n, degrees in ARM_LIMITS.items():
            if abs(q[n]) > np.deg2rad(degrees) + 1e-8:
                raise ValueError('Preview joint limit: ' + n)
        for n in ('7', '57'):
            if not 0 <= q[n] <= 1:
                raise ValueError('Gripper limit: ' + n)
        if abs(q['101']) > np.pi/2 or abs(q['102']) > np.pi/4:
            raise ValueError('Head preview limit')

    def distances(self, q, deadline=float('inf'), indices=None):
        frames = self.frames(q)
        for shape in self.shapes:
            T = frames[shape['link']]
            shape['object'].setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
            shape['current_frame'] = T
        values = []
        if indices is None:
            indices = range(len(self.pairs))
        for index in indices:
            i, j, _ = self.pairs[index]
            if time.monotonic() > deadline:
                raise TimeoutError('Collision budget exhausted')
            d = fcl.distance(self.shapes[i]['object'], self.shapes[j]['object'],
                             fcl.DistanceRequest(), fcl.DistanceResult())
            a, b = self.shapes[i], self.shapes[j]
            # A separated whole hull is a cheap valid lower bound. Only
            # refine nearby/overlapping hulls; compound distance is then exact
            # for the union of envelopes, not the raw CAD triangles.
            if d < .01 and (a['manager'] is not None or b['manager'] is not None):
                for shape in (a, b):
                    if shape['manager'] is not None and not np.array_equal(
                            shape['transform'], shape['current_frame']):
                        T = shape['current_frame']
                        transform = fcl.Transform(T[:3, :3], T[:3, 3])
                        for piece in shape['pieces']:
                            piece.setTransform(transform)
                        shape['manager'].update()
                        shape['transform'] = T.copy()
                data = fcl.DistanceData()
                data.deadline, data.timed_out = deadline, False
                if a['manager'] is not None:
                    a['manager'].distance(b['manager'] if b['manager'] is not None
                                          else b['object'], data, compound_distance_callback)
                else:
                    b['manager'].distance(a['object'], data, compound_distance_callback)
                if data.timed_out:
                    raise TimeoutError('Collision budget exhausted')
                d = data.result.min_distance
            if not np.isfinite(d):
                raise ValueError('Invalid distance result')
            values.append(float(d))
        return values

    def check_motion(self, start, goal, budget=.08, max_depth=12):
        began = time.monotonic()
        samples, nearest, nearest_pair = 0, float('inf'), None
        def report(safe, reason, category=None):
            return dict(safe=safe, reason=reason, distance_m=None if not np.isfinite(nearest) else nearest,
                        pair=nearest_pair, samples=samples,
                        category=category or ('safe' if safe else 'unverified'),
                        elapsed_ms=(time.monotonic()-began)*1000, model_version=self.version)
        try:
            self.validate(start)
            self.validate(goal)
            dq = {k: goal[k]-start[k] for k in start}
            lipschitz = np.array([sum(r*abs(dq[k]) for k, r in coeff.items())
                                  for _, _, coeff in self.pairs])
            pending = [(0., 1., 0, np.arange(len(self.pairs)))]
            while pending:
                lo, hi, depth, active = pending.pop()
                mid = (lo+hi)/2
                q = {k: start[k]+mid*dq[k] for k in start}
                ds = np.array(self.distances(q, began+budget, active))
                samples += 1
                idx = int(np.argmin(ds))
                if ds[idx] < nearest:
                    nearest = float(ds[idx])
                    i, j, _ = self.pairs[active[idx]]
                    nearest_pair = [self.shapes[i]['link'], self.shapes[j]['link']]
                margins = self.margins[active]
                if np.any(ds < margins):
                    idx = int(np.argmin(ds-margins))
                    i, j, _ = self.pairs[active[idx]]
                    nearest_pair = [self.shapes[i]['link'], self.shapes[j]['link']]
                    nearest = float(ds[idx])
                    return report(False, '检测几何相交' if nearest <= 0 else '安全余量不足',
                                  'collision' if nearest <= 0 else 'clearance')
                unresolved = active[ds - lipschitz[active]*(hi-lo)/2 < margins]
                if len(unresolved) == 0:
                    continue
                if depth >= max_depth:
                    return report(False, '运动区间无法认证（未确认碰撞）', 'unverified')
                pending.extend(((mid, hi, depth+1, unresolved), (lo, mid, depth+1, unresolved)))
            if time.monotonic()-began > budget:
                return report(False, '计算超时，安全性未确认', 'timeout')
            return report(True, '通过')
        except TimeoutError:
            return report(False, '计算超时，安全性未确认', 'timeout')
        except ValueError as exc:
            return report(False, str(exc), 'invalid')

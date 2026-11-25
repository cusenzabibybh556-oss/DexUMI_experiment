"""Aggregate vendor URDF inertias for offline free-space physics experiments.

The hand mass/inertia comes from the given URDF. The arm mount transform and
adapter mass distribution are explicit assumptions, not identified hardware.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def vector(value):
    return np.fromstring(value or '0 0 0', sep=' ')


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = vector(element.get('xyz'))
        t[:3, :3] = Rotation.from_euler('xyz', vector(element.get('rpy'))).as_matrix()
    return t


def combine(parts):
    mass = sum(m for m, _, _ in parts)
    center = sum(m*c for m, c, _ in parts)/mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c-center
        inertia += i+m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    return mass, center, inertia


def aggregate_hand(urdf, closure=0.):
    if not np.isfinite(closure) or not 0 <= closure <= 1:
        raise ValueError('Hand closure must be within [0,1]')
    urdf = Path(urdf)
    root = ET.parse(urdf).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    roots = set(links)-children
    if len(roots) != 1:
        raise ValueError('Expected a single connected URDF hand root')
    transforms = {roots.pop(): np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            parent = j.find('parent').get('link')
            if parent not in transforms:
                continue
            t = transforms[parent]@origin(j.find('origin'))
            if j.get('type') != 'fixed':
                limits = j.find('limit')
                lo, hi = float(limits.get('lower', '0')), float(limits.get('upper', '0'))
                angle = lo+closure*(hi-lo)
                axis = vector(j.find('axis').get('xyz'))
                a = np.eye(4)
                a[:3, :3] = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()
                t = t@a
            transforms[j.find('child').get('link')] = t
            pending.remove(j)
            progressed = True
        if not progressed:
            raise ValueError('Disconnected or cyclic hand URDF')
    parts = []
    for name, l in links.items():
        inertial = l.find('inertial')
        if inertial is None:
            continue
        m = float(inertial.find('mass').get('value'))
        e = inertial.find('inertia')
        tensor = np.array([[float(e.get('ixx')), float(e.get('ixy')), float(e.get('ixz'))],
                           [float(e.get('ixy')), float(e.get('iyy')), float(e.get('iyz'))],
                           [float(e.get('ixz')), float(e.get('iyz')), float(e.get('izz'))]])
        t = transforms[name]@origin(inertial.find('origin'))
        if m <= 0 or not np.isfinite(tensor).all() or np.linalg.eigvalsh(tensor).min() <= 0:
            raise ValueError(f'Invalid vendor inertial data: {name}')
        parts.append((m, t[:3, 3], t[:3, :3]@tensor@t[:3, :3].T))
    if not parts:
        raise ValueError('No hand inertial data')
    return combine(parts)


def assembly(urdf, total_mass=.396, closure=0., mount_pos=(-.0317, 0, -.0812), mount_rpy_deg=(180, 0, 0)):
    mass, center, inertia = aggregate_hand(urdf, closure)
    if total_mass < mass-1e-9:
        raise ValueError('Configured total mass is below vendor hand mass')
    p = np.asarray(mount_pos, float)
    r = Rotation.from_euler('xyz', mount_rpy_deg, degrees=True).as_matrix()
    parts = [(mass, p+r@center, r@inertia@r.T)]
    adapter_mass = max(0., total_mass-mass)
    if adapter_mass:
        # Synthetic 10 mm radius compact adapter, centered at the assumed mount.
        parts.append((adapter_mass, p, np.eye(3)*.4*adapter_mass*.01**2))
    total, combined_center, combined_inertia = combine(parts)
    return dict(mass_kg=total, COM_wrist_m=combined_center.tolist(), inertia_wrist_kg_m2=combined_inertia.tolist(),
        vendor_hand_mass_kg=mass, vendor_hand_COM_root_m=center.tolist(),
        assumed_adapter_mass_kg=adapter_mass, assumed_mount_position_m=p.tolist(),
        assumed_mount_rpy_deg=list(mount_rpy_deg), closure_fraction_of_URDF_ranges=closure,
        source=str(urdf), source_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
        limitations=['Mount transform and adapter inertia are assumptions; not measured.',
                     'URDF hand is rigidly aggregated at a static closure; no finger contact or O6 actuation dynamics.'])

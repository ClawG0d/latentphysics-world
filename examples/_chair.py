"""Articulated office-chair asset for the chair-push demo.

PROVENANCE: this asset's geometry was authored by an LLM from reference
product photos (owner-approved generative content, 2026-07-08 — see the
scope charter's sign-off rule). It is classical MJCF all the way down:
box/cylinder/capsule/ellipsoid primitives, explicit masses, no learned
anything. Fidelity was verified the charter way — rendered in neutral and
articulated states and reviewed, plus the mechanical semantics test in
tests/test_chair_example.py.

Reference dimensions (annotated product photo, inches -> meters):
    headrest        13.39 x 7.48 in  -> 0.340 x 0.190 m
    headrest gap    3.94 in          -> 0.100 m   (stalk visible)
    backrest        21.26 x 17.72 in -> 0.540 h x 0.450 w m
    base diameter   21.26 in         -> 0.540 m   (star reach 0.27)

Articulation (19 DoF): freejoint root, 5x caster swivel, 5x wheel roll,
seat swivel (yaw), spring-loaded backrest recline, headrest pitch.
Total mass ~19 kg. Visual geoms live in group 2, collision geoms in
group 3 (repo convention); the caster wheels collide as sphere pairs.

Textures come from latentphysics.assets.materials (procedural fabric
grain), which needs imageio on first run — install the ``demos`` extra.
"""

from __future__ import annotations

import math

import numpy as np

from latentphysics.assets.materials import material_assets

# ---------------------------------------------------------------- dimensions
RW = 0.030          # caster wheel radius
BASE_R = 0.270      # star arm reach (base diameter 0.54)
SEAT_TOP = 0.482    # seat cushion crown height
SEAT_W2 = 0.240     # seat half width  (0.48)
SEAT_D2 = 0.235     # seat half depth  (0.47)
BACK_X = -0.215     # backrest plane x (behind seat center)
BACK_Z0 = 0.487     # backrest bottom (sweeps close to the seat)
BACK_Z1 = 1.045     # backrest top    (0.54 tall)
HEAD_GAP = 0.100    # backrest top -> headrest bottom
HEAD_W2 = 0.170     # headrest half width (0.34)
HEAD_H2 = 0.093     # headrest half height (~0.19)

FABRIC = "0.20 0.20 0.215 1"        # charcoal fabric (tints grain_fabric)
PLASTIC = "0.155 0.155 0.165 1"     # near-black frame plastic
PLASTIC_D = "0.125 0.125 0.135 1"   # darker plastic (wheels, underside)
METAL = "0.52 0.53 0.55 1"          # gas-lift piston


def _q(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_zy(yaw, pitch):
    """World-frame: pitch about y, then yaw about z."""
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    return _quat_mul(qz, qy)


def _geom(gtype, size, pos, *, quat=None, rgba=PLASTIC, material=None,
          mass=None, collide=False, friction=None):
    """Visual (group 2, no contact) or collision (group 3) geom."""
    a = [f'type="{gtype}"', f'size="{_q(*size)}"', f'pos="{_q(*pos)}"']
    if quat is not None:
        a.append(f'quat="{_q(*quat)}"')
    if material:
        a.append(f'material="{material}"')
    a.append(f'rgba="{rgba}"')
    if collide:
        a.append(f'group="3" mass="{mass:.4f}"')
        if friction:
            a.append(f'friction="{friction}"')
    else:
        a.append('group="2" contype="0" conaffinity="0" mass="0"')
    return f'<geom {" ".join(a)}/>'


# ------------------------------------------------------------------- pieces
def _star_base():
    """5-arm star base + gas-lift column (root body geoms)."""
    g = []
    g.append(_geom("cylinder", (0.055, 0.030), (0, 0, 0.105), rgba=PLASTIC))
    g.append(_geom("cylinder", (0.058, 0.020), (0, 0, 0.132), mass=3.2,
                   rgba=PLASTIC, collide=True))
    for i in range(5):
        a = math.radians(18 + 72 * i)
        ca, sa = math.cos(a), math.sin(a)
        r0, r1, z0, z1 = 0.045, BASE_R - 0.012, 0.105, 0.058
        rm, zm = (r0 + r1) / 2, (z0 + z1) / 2
        L2 = math.hypot(r1 - r0, z1 - z0) / 2
        pitch = math.atan2(z0 - z1, r1 - r0)
        quat = _quat_zy(a, pitch)
        pos = (rm * ca, rm * sa, zm)
        g.append(_geom("box", (L2, 0.030, 0.016), pos, quat=quat, rgba=PLASTIC))
        g.append(_geom("box", (L2 - 0.003, 0.018, 0.009),
                       (rm * ca, rm * sa, zm + 0.020), quat=quat, rgba=PLASTIC))
    g.append(_geom("cylinder", (0.028, 0.065), (0, 0, 0.185), rgba=PLASTIC))
    g.append(_geom("cylinder", (0.019, 0.062), (0, 0, 0.295), rgba=METAL,
                   material="mat_metal"))
    g.append(_geom("capsule", (0.030, 0.055), (0, 0, 0.20), mass=1.8,
                   rgba=PLASTIC, collide=True))
    return "\n      ".join(g)


def _caster(i):
    """Caster housing (swivel) + twin wheel (roll). Trail 20 mm.
    Collision is the sphere pair; the cylinders are the visible wheels."""
    a = math.radians(18 + 72 * i)
    tip = (BASE_R * math.cos(a), BASE_R * math.sin(a), 0.058)
    return f"""
      <body name="caster{i}" pos="{_q(*tip)}">
        <joint name="caster{i}_swivel" type="hinge" axis="0 0 1"
               damping="0.02" armature="0.0005"/>
        <geom type="cylinder" size="0.020 0.013" pos="0 0 -0.004"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="box" size="0.011 0.019 0.022" pos="0.020 0 -0.022"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="box" size="0.018 0.028 0.007" pos="0.020 0 -0.004"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="sphere" size="0.012" pos="0 0 -0.01" group="3" mass="0.08"/>
        <body name="wheel{i}" pos="0.020 0 -0.028">
          <joint name="wheel{i}_roll" type="hinge" axis="0 1 0"
                 damping="0.010" armature="0.0005"/>
          <geom type="sphere" size="{RW:.3f}" pos="0 0.0135 0" group="3"
                mass="0.05" friction="0.9 0.006 0.0002" rgba="{PLASTIC_D}"/>
          <geom type="sphere" size="{RW:.3f}" pos="0 -0.0135 0" group="3"
                mass="0.05" friction="0.9 0.006 0.0002" rgba="{PLASTIC_D}"/>
          <geom type="cylinder" size="0.030 0.0075" pos="0 0.0135 0"
                quat="0.7071 0.7071 0 0"
                rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
          <geom type="cylinder" size="0.030 0.0075" pos="0 -0.0135 0"
                quat="0.7071 0.7071 0 0"
                rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
          <geom type="cylinder" size="0.011 0.023" pos="0 0 0"
                quat="0.7071 0.7071 0 0"
                rgba="0.22 0.22 0.24 1" group="2" contype="0" conaffinity="0" mass="0"/>
        </body>
      </body>"""


def _seat():
    """Seat cushion + tilt mechanism + paddles + mounted armrests."""
    g = []
    g.append(_geom("box", (0.105, 0.090, 0.024), (0.01, 0, 0.388), rgba=PLASTIC_D))
    g.append(_geom("box", (0.10, 0.09, 0.02), (0.01, 0, 0.388), mass=2.4,
                   collide=True, rgba=PLASTIC_D))
    g.append(_geom("capsule", (0.009, 0.035), (0.06, -0.13, 0.385),
                   quat=_quat_zy(0, math.radians(90)), rgba=PLASTIC_D))
    g.append(_geom("capsule", (0.007, 0.028), (-0.04, -0.125, 0.380),
                   quat=_quat_zy(math.radians(20), math.radians(90)),
                   rgba=PLASTIC_D))
    # main cushion: plump, top surface proudest (waterfall front edge)
    g.append(_geom("box", (SEAT_D2 - 0.040, SEAT_W2, 0.040),
                   (0.012, 0, 0.442), material="mat_fabric", rgba=FABRIC))
    g.append(_geom("cylinder", (0.040, SEAT_W2 - 0.02), (SEAT_D2 - 0.025, 0, 0.442),
                   quat=(0.7071, 0.7071, 0, 0),
                   material="mat_fabric", rgba=FABRIC))
    g.append(_geom("box", (SEAT_D2, SEAT_W2, 0.040), (0.012, 0, 0.442),
                   mass=4.2, collide=True, rgba=FABRIC))
    for s in (1, -1):
        g.append(_geom("box", (SEAT_D2 - 0.06, 0.030, 0.014),
                       (0.0, s * (SEAT_W2 - 0.03), 0.458),
                       quat=(math.cos(s * 0.07), math.sin(s * 0.07), 0, 0),
                       material="mat_fabric", rgba=FABRIC))
    # armrests: bolted mount under the seat -> angled post -> adjuster -> pad
    post_tilt = 0.31
    for s in (1, -1):
        g.append(_geom("box", (0.026, 0.022, 0.016), (0.02, s * 0.185, 0.386),
                       rgba=PLASTIC_D))
        g.append(_geom("capsule", (0.016, 0.084),
                       (0.02, s * 0.214, 0.482),
                       quat=(math.cos(s * post_tilt / 2), -math.sin(s * post_tilt / 2), 0, 0),
                       rgba=PLASTIC))
        g.append(_geom("box", (0.028, 0.026, 0.026), (0.02, s * 0.245, 0.594),
                       rgba=PLASTIC_D))
        g.append(_geom("box", (0.125, 0.042, 0.015), (0.035, s * 0.245, 0.634),
                       rgba=PLASTIC_D))
        g.append(_geom("cylinder", (0.042, 0.015), (0.16, s * 0.245, 0.634),
                       rgba=PLASTIC_D))
        g.append(_geom("cylinder", (0.042, 0.015), (-0.09, s * 0.245, 0.634),
                       rgba=PLASTIC_D))
        g.append(_geom("box", (0.125, 0.042, 0.015), (0.035, s * 0.245, 0.634),
                       mass=0.7, collide=True, rgba=PLASTIC_D))
        g.append(_geom("capsule", (0.016, 0.084), (0.02, s * 0.214, 0.482),
                       quat=(math.cos(s * post_tilt / 2), -math.sin(s * post_tilt / 2), 0, 0),
                       mass=0.3, collide=True, rgba=PLASTIC))
    return "\n        ".join(g)


def _backrest():
    """Sculpted backrest: tangent-continuous slab stack with lumbar bulge,
    3-panel rear shell, spine frame, L-bracket, headrest stalk."""
    g = []
    n = 16
    zs = np.linspace(BACK_Z0 + 0.025, BACK_Z1 - 0.025, n)
    t = np.linspace(0.0, 1.0, n)
    half_w = 0.186 + 0.040 * np.sin(np.pi * (t * 0.92 + 0.04))
    x_off = (0.050 * np.sin(np.pi * np.clip(t * 1.8, 0, 1))
             - 0.090 * np.clip(t - 0.45, 0, 1) ** 1.5)
    # tangent-continuous: pitch each slab to the local slope dx/dz so the
    # front faces line up into one smooth curve instead of a staircase
    pitch = -np.arctan(np.gradient(x_off, zs))
    dz = (zs[1] - zs[0]) / 2 + 0.014
    for z, w2, dx, p in zip(zs, half_w, x_off, pitch):
        quat = _quat_zy(0.0, float(p))
        g.append(_geom("box", (0.034, float(w2), dz), (BACK_X + float(dx), 0, float(z)),
                       quat=quat, material="mat_fabric", rgba=FABRIC))
        for s in (1, -1):
            wq = _quat_zy(s * 0.26, float(p))
            g.append(_geom("box", (0.020, 0.026, dz - 0.004),
                           (BACK_X + float(dx) + 0.016, s * (float(w2) - 0.014), float(z)),
                           quat=wq, material="mat_fabric", rgba=FABRIC))
    # collision: 3 coarse slabs following the curve
    g.append(_geom("box", (0.036, 0.20, 0.095), (BACK_X + 0.045, 0, 0.60),
                   quat=_quat_zy(0, -0.04), mass=1.0, collide=True, rgba=FABRIC))
    g.append(_geom("box", (0.036, 0.225, 0.10), (BACK_X + 0.032, 0, 0.79),
                   quat=_quat_zy(0, 0.08), mass=1.0, collide=True, rgba=FABRIC))
    g.append(_geom("box", (0.036, 0.205, 0.10), (BACK_X - 0.03, 0, 0.975),
                   quat=_quat_zy(0, 0.21), mass=1.0, collide=True, rgba=FABRIC))
    # rear shell: 3 large panels -> clean sculpted back
    g.append(_geom("box", (0.013, 0.165, 0.095), (BACK_X - 0.052, 0, 0.635),
                   quat=_quat_zy(0, 0.02), rgba=PLASTIC))
    g.append(_geom("box", (0.013, 0.175, 0.10), (BACK_X - 0.060, 0, 0.795),
                   quat=_quat_zy(0, 0.11), rgba=PLASTIC))
    g.append(_geom("box", (0.013, 0.160, 0.10), (BACK_X - 0.098, 0, 0.955),
                   quat=_quat_zy(0, 0.26), rgba=PLASTIC))
    # L-bracket to the tilt mechanism: visible side load path seat<->back
    g.append(_geom("box", (0.070, 0.042, 0.017), (BACK_X + 0.062, 0, 0.442),
                   quat=_quat_zy(0, -0.52), rgba=PLASTIC))
    g.append(_geom("box", (0.048, 0.042, 0.015), (BACK_X + 0.095, 0, 0.402),
                   rgba=PLASTIC))
    # spine frame
    g.append(_geom("box", (0.016, 0.052, 0.26), (BACK_X - 0.055, 0, 0.78),
                   quat=_quat_zy(0, 0.10), rgba=PLASTIC))
    g.append(_geom("box", (0.016, 0.052, 0.26), (BACK_X - 0.055, 0, 0.78),
                   quat=_quat_zy(0, 0.10), mass=1.2, collide=True, rgba=PLASTIC))
    # headrest stalk: rises from the top slab up to the headrest pitch hinge
    top_x = BACK_X - 0.062
    g.append(_geom("box", (0.010, 0.032, 0.078),
                   (top_x - 0.020, 0, BACK_Z1 + 0.052),
                   quat=_quat_zy(0, 0.18), rgba=PLASTIC))
    return "\n          ".join(g)


def _headrest():
    """Wide flat pillow: box core + soft ellipsoid ends, on a pitch hinge.
    The bracket bridges the hinge and overlaps the stalk top so the pivot
    reads as one continuous column in every pitch pose."""
    g = []
    g.append(_geom("box", (0.010, 0.027, 0.046), (-0.018, 0, -HEAD_H2 + 0.016),
                   quat=_quat_zy(0, 0.18), rgba=PLASTIC))
    core_q = _quat_zy(0, 0.14)
    g.append(_geom("box", (0.036, HEAD_W2 - 0.038, HEAD_H2 - 0.018), (0, 0, 0),
                   quat=core_q, material="mat_fabric", rgba=FABRIC))
    for s in (1, -1):
        g.append(_geom("ellipsoid", (0.036, 0.055, HEAD_H2 - 0.018),
                       (0, s * (HEAD_W2 - 0.05), 0),
                       quat=core_q, material="mat_fabric", rgba=FABRIC))
    g.append(_geom("box", (0.040, HEAD_W2 - 0.01, HEAD_H2 - 0.01), (0, 0, 0),
                   quat=core_q, mass=0.55, collide=True, rgba=FABRIC))
    return "\n            ".join(g)


def chair_body(name="chair", pos=(0, 0, 0.004), yaw_deg=0.0):
    """The chair as a worldbody fragment (freejoint root)."""
    yaw = math.radians(yaw_deg)
    quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    casters = "".join(_caster(i) for i in range(5))
    hz = BACK_Z1 + HEAD_GAP + HEAD_H2 - 0.01
    return f"""
    <body name="{name}" pos="{_q(*pos)}" quat="{_q(*quat)}">
      <freejoint name="{name}_free"/>
      {_star_base()}
      {casters}
      <body name="{name}_seat" pos="0 0 0">
        <joint name="{name}_swivel" type="hinge" axis="0 0 1" pos="0 0 0.35"
               damping="0.6" armature="0.01" frictionloss="0.4"/>
        {_seat()}
        <body name="{name}_back" pos="0 0 0">
          <joint name="{name}_recline" type="hinge" axis="0 1 0"
                 pos="-0.12 0 0.40" range="-0.03 0.22"
                 stiffness="360" damping="28" armature="0.01"/>
          {_backrest()}
          <body name="{name}_headrest" pos="{_q(BACK_X - 0.092, 0, hz)}">
            <joint name="{name}_head_pitch" type="hinge" axis="0 1 0"
                   pos="0.0 0 {-HEAD_H2 - 0.005:.3f}" range="-0.35 0.35"
                   stiffness="60" damping="4" armature="0.005"/>
            {_headrest()}
          </body>
        </body>
      </body>
    </body>"""


def chair_assets():
    extra = ('<material name="mat_metal" specular="0.9" shininess="0.8" '
             'reflectance="0.35"/>')
    return material_assets() + extra


def chair_scene_xml():
    """Studio scene: the chair alone on a light floor (for inspection)."""
    return f"""<mujoco model="lpw_chair_studio">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <statistic center="0 0 0.65" extent="2.2"/>
  <visual>
    <global offwidth="2560" offheight="1920"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.36 0.36 0.38" diffuse="0.62 0.62 0.62" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    {chair_assets()}
  </asset>
  <worldbody>
    <light pos="1.8 -1.2 2.6" dir="-0.55 0.35 -0.75" diffuse="0.75 0.74 0.72" castshadow="true"/>
    <light pos="-1.6 1.8 2.2" dir="0.5 -0.55 -0.68" diffuse="0.34 0.35 0.38" castshadow="false"/>
    <geom name="floor" type="plane" size="3.2 3.2 0.1" material="mat_plaster"
          quat="0.981 0 0 0.195" rgba="0.87 0.87 0.88 1"
          friction="0.9 0.005 0.0001"/>
    {chair_body()}
  </worldbody>
</mujoco>"""

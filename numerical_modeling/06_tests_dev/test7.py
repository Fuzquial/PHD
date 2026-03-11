# -*- coding: mbcs -*-
# Abaqus/CAE noGUI script
# CVT (Lloyd) Voronoi tessellation "comme l'article" = microstructure implicite par ELSET d'elements (eq.5-9)
# + 5 types de grains (5 materiaux/sections) assignes aux grains (proportions)
#
# IMPORTANT PERF:
# - Lloyd naif est O(N_sites * N_grains) par iteration.
# - Pour eviter le "freeze" avec 250k elems et ~4k grains, on applique un "probabilistic Lloyd":
#   on echantillonne un sous-ensemble des centroïdes pour Lloyd (MAX_SITES_FOR_LLOYD),
#   puis on fait l'assignation finale sur TOUS les elements.
#
# Run:
#   abaqus cae noGUI=cvt_voronoi_5types_cylinder_fast.py

from abaqus import mdb
from abaqusConstants import *
import regionToolset
import mesh
import random
import math
import time

# ============================================================
# USER PARAMETERS
# ============================================================

MODEL_NAME = 'Model-1'

# Geometry (units consistent with your model, e.g. mm)
PART_NAME = 'Cyl3D'
INST_NAME = PART_NAME + '-1'
R = 2.5      # radius
H = 10.0     # height

# Meshing
ELEMENT_SIZE = 0.5
PREFER_HEX = True
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# CVT / Voronoi implicit
RANDOM_SEED = 1234

# Target grain size (X,Y,Z). Smaller => more grains.
GRAIN_SIZE_XYZ = (0.25, 0.25, 0.75)

# --- Performance knobs ---
# Lloyd iterations: 5-12 is usually enough. Set 0 for "pure Voronoi" without CVT smoothing.
LLOYD_ITERS = 8

# Subsampled sites used during Lloyd updates (probabilistic Lloyd spirit).
# With 250k elems, 10k-30k is a good range.
MAX_SITES_FOR_LLOYD = 20000

# Optional: if NP is huge, you can cap it (still "Voronoi", just fewer grains)
NP_CAP = None  # e.g. 3000 or None

# Optional: For very large NP, do fewer Lloyd iters automatically
AUTO_TUNE = True

# 5 grain types (materials/sections)
NB_TYPES = 5
TYPE_PROPORTIONS = [0.40, 0.20, 0.15, 0.15, 0.10]

# Example material data (edit as needed)
E_TYPES  = [50000.0, 52000.0, 48000.0, 51000.0, 49000.0]
NU_TYPES = [0.25,    0.25,    0.25,    0.25,    0.25]

USE_DENSITY = False
RHO_TYPES = [2.62e-9]*NB_TYPES  # tonne/mm^3 if mm-MPa-tonne

# Output naming
GRAIN_SET_PREFIX = 'ELSET_GRAIN_'

# ============================================================
# Helpers
# ============================================================

def aspect_ratio_from_grain_size(grain_xyz):
    gmin = min(grain_xyz)
    r = (grain_xyz[0]/gmin, grain_xyz[1]/gmin, grain_xyz[2]/gmin)
    return r, gmin

def scale_point(p, r):
    return (p[0]/r[0], p[1]/r[1], p[2]/r[2])

def dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]; dz = a[2]-b[2]
    return dx*dx + dy*dy + dz*dz

def nearest_seed_index(p, seeds):
    best_i = 0
    best_d2 = dist2(p, seeds[0])
    for i in range(1, len(seeds)):
        d2 = dist2(p, seeds[i])
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i

def lloyd_cvt(points, n_seeds, n_iters, seed=0):
    """Discrete Lloyd CVT using given sites. If n_iters==0 returns random seeds (Voronoi)."""
    random.seed(seed)

    if n_seeds > len(points):
        n_seeds = len(points)
    if n_seeds < 1:
        n_seeds = 1

    # init from random sites
    seeds = [points[i] for i in random.sample(range(len(points)), n_seeds)]

    if n_iters <= 0:
        return seeds

    for it in range(n_iters):
        accum = [(0.0, 0.0, 0.0) for _ in range(n_seeds)]
        counts = [0 for _ in range(n_seeds)]

        # assignment (within Lloyd)
        for p in points:
            gi = nearest_seed_index(p, seeds)
            ax, ay, az = accum[gi]
            accum[gi] = (ax + p[0], ay + p[1], az + p[2])
            counts[gi] += 1

        # update
        new_seeds = []
        for i in range(n_seeds):
            if counts[i] > 0:
                ax, ay, az = accum[i]
                inv = 1.0 / float(counts[i])
                new_seeds.append((ax*inv, ay*inv, az*inv))
            else:
                new_seeds.append(points[random.randrange(0, len(points))])
        seeds = new_seeds

        print('--- Lloyd iter %d / %d done' % (it+1, n_iters))

    return seeds

def estimate_np_cylinder(R, H, r_aspect, equi_size):
    """NP ~ Vol(domain_scaled)/Vol(grain_scaled). Cylinder volume scaled by 1/(rx*ry*rz)."""
    Vcyl = math.pi * (R**2) * H
    Vscaled = Vcyl / (r_aspect[0] * r_aspect[1] * r_aspect[2])
    Vgrain = equi_size**3
    return int(max(1, round(Vscaled / Vgrain)))

def build_node_map(part):
    return {n.label: n.coordinates for n in part.nodes}

def element_centroid(elem, node_map):
    """Centroid by node average; ignores 0 padding."""
    sx = sy = sz = 0.0
    n_used = 0
    for nl in elem.connectivity:
        if nl == 0:
            continue
        c = node_map.get(nl, None)
        if c is None:
            continue
        sx += c[0]; sy += c[1]; sz += c[2]
        n_used += 1

    if n_used == 0:
        # fallback
        try:
            pt = elem.pointOn[0]
            return (pt[0], pt[1], pt[2])
        except:
            return (0.0, 0.0, 0.0)

    inv = 1.0 / float(n_used)
    return (sx*inv, sy*inv, sz*inv)

def weighted_choice_type(proportions):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(proportions):
        cum += p
        if r <= cum:
            return i+1
    return len(proportions)

def progress(msg, k, n, every=20000):
    if n <= 0:
        return
    if k == 0 or k == n-1 or (k % every == 0):
        pct = 100.0 * float(k) / float(n)
        print('%s %d/%d (%.1f%%)' % (msg, k, n, pct))

# ============================================================
# Model
# ============================================================

t0 = time.time()

if MODEL_NAME not in mdb.models:
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# Cleanup old part if present
if PART_NAME in mdl.parts:
    del mdl.parts[PART_NAME]

# ============================================================
# Geometry (cylinder)
# ============================================================

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0*max(R, H))
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))
p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=H)
del mdl.sketches['__profile__']

# ============================================================
# 5 materials + 5 sections
# ============================================================

if len(TYPE_PROPORTIONS) != NB_TYPES:
    raise ValueError("TYPE_PROPORTIONS length must match NB_TYPES")
if len(E_TYPES) != NB_TYPES or len(NU_TYPES) != NB_TYPES:
    raise ValueError("E_TYPES / NU_TYPES length must match NB_TYPES")

# normalize proportions
sprop = sum(TYPE_PROPORTIONS)
if sprop <= 0:
    raise ValueError("TYPE_PROPORTIONS must sum to > 0")
TYPE_PROPORTIONS = [pi/sprop for pi in TYPE_PROPORTIONS]

# cleanup
for t in range(NB_TYPES):
    mname = 'MatType-%d' % (t+1)
    sname = 'SecType-%d' % (t+1)
    if mname in mdl.materials:
        del mdl.materials[mname]
    if sname in mdl.sections:
        del mdl.sections[sname]

for t in range(NB_TYPES):
    mname = 'MatType-%d' % (t+1)
    sname = 'SecType-%d' % (t+1)
    mat = mdl.Material(name=mname)
    if USE_DENSITY:
        mat.Density(table=((RHO_TYPES[t],),))
    mat.Elastic(table=((E_TYPES[t], NU_TYPES[t]),))
    mdl.HomogeneousSolidSection(name=sname, material=mname)

# ============================================================
# Mesh
# ============================================================

cells = p.cells[:]
p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)

meshed_as = None
if PREFER_HEX:
    try:
        p.setMeshControls(regions=cells, technique=SWEEP, elemShape=HEX)
        et = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
        p.setElementType(regions=(cells,), elemTypes=(et,))
        p.generateMesh()
        meshed_as = 'HEX (SWEEP)'
    except Exception as e:
        print('--- HEX SWEEP failed, fallback to TET. Reason:', str(e))
        meshed_as = None

if meshed_as is None:
    try:
        p.deleteMesh()
    except:
        pass
    p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    p.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
    et = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
    p.setElementType(regions=(cells,), elemTypes=(et,))
    p.generateMesh()
    meshed_as = 'TET (FREE)'

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("No elements generated. Check mesh settings.")

print('--- Meshing done:', meshed_as)
print('--- Elements:', len(elems))

# ============================================================
# Centroids
# ============================================================

random.seed(RANDOM_SEED)
node_map = build_node_map(p)

centroids = []
nE = len(elems)
for k, e in enumerate(elems):
    if k % 20000 == 0:
        progress('--- Centroids', k, nE, every=20000)
    centroids.append(element_centroid(e, node_map))
print('--- Centroids computed.')

# Scale (aspect ratio)
r_aspect, equi_size = aspect_ratio_from_grain_size(GRAIN_SIZE_XYZ)
centroids_scaled = [scale_point(c, r_aspect) for c in centroids]

# Estimate NP
NP = estimate_np_cylinder(R, H, r_aspect, equi_size)
if NP_CAP is not None:
    NP = min(NP, int(NP_CAP))

# Auto-tune if needed
if AUTO_TUNE:
    # if too many grains, reduce Lloyd iters and/or sites
    if NP > 3000:
        LLOYD_ITERS = min(LLOYD_ITERS, 6)
    if NP > 5000:
        LLOYD_ITERS = min(LLOYD_ITERS, 4)
    if MAX_SITES_FOR_LLOYD is None:
        MAX_SITES_FOR_LLOYD = 20000

print('--- Target grain size XYZ:', GRAIN_SIZE_XYZ)
print('--- Aspect ratio r:', r_aspect, 'equiaxed size:', equi_size)
print('--- Estimated NP (nb grains):', NP)
print('--- Lloyd iters:', LLOYD_ITERS)
print('--- MAX_SITES_FOR_LLOYD:', MAX_SITES_FOR_LLOYD)

# ============================================================
# Probabilistic Lloyd sites
# ============================================================

sites_for_lloyd = centroids_scaled
if MAX_SITES_FOR_LLOYD is not None and MAX_SITES_FOR_LLOYD < len(centroids_scaled):
    idx = random.sample(range(len(centroids_scaled)), MAX_SITES_FOR_LLOYD)
    sites_for_lloyd = [centroids_scaled[i] for i in idx]
    print('--- Lloyd sites subsampled:', len(sites_for_lloyd), '/', len(centroids_scaled))

# Lloyd CVT (or 0 iters => pure Voronoi)
t_lloyd0 = time.time()
seeds = lloyd_cvt(sites_for_lloyd, NP, LLOYD_ITERS, seed=RANDOM_SEED)
print('--- Lloyd/CVT done in %.1fs' % (time.time() - t_lloyd0))

# ============================================================
# Final assignment: each element -> nearest seed
# ============================================================

grain_to_labels = [[] for _ in range(len(seeds))]

t_assign0 = time.time()
for k, e in enumerate(elems):
    if k % 20000 == 0:
        progress('--- Final assign', k, nE, every=20000)
    gi = nearest_seed_index(centroids_scaled[k], seeds)
    grain_to_labels[gi].append(e.label)

print('--- Final assignment done in %.1fs' % (time.time() - t_assign0))

# ============================================================
# Create ELSET per grain + assign 5 sections
# ============================================================

# delete old grain sets
for sname in list(p.sets.keys()):
    if sname.startswith(GRAIN_SET_PREFIX):
        del p.sets[sname]

random.seed(RANDOM_SEED)
nonempty = 0
type_counts = [0]*NB_TYPES

t_sets0 = time.time()
for i, lab_list in enumerate(grain_to_labels):
    if not lab_list:
        continue
    nonempty += 1

    set_name = '%s%04d' % (GRAIN_SET_PREFIX, i+1)
    elems_i = p.elements.sequenceFromLabels(labels=tuple(lab_list))
    p.Set(name=set_name, elements=elems_i)

    # choose one of 5 types
    t = weighted_choice_type(TYPE_PROPORTIONS)  # 1..NB_TYPES
    type_counts[t-1] += 1
    sec_name = 'SecType-%d' % t

    region = regionToolset.Region(elements=elems_i)
    p.SectionAssignment(region=region, sectionName=sec_name)

    if i % 500 == 0:
        print('--- Sets/Sections %d/%d' % (i, len(grain_to_labels)))

print('--- Sets/Sections done in %.1fs' % (time.time() - t_sets0))
print('--- Non-empty grains:', nonempty, '/', len(seeds))
print('--- Type counts (grains):', type_counts)

# ============================================================
# Assembly / instance
# ============================================================

a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

if INST_NAME in a.instances:
    del a.instances[INST_NAME]

a.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Instance created:', INST_NAME)
print('--- TOTAL time %.1fs' % (time.time() - t0))
print('--- Done.')
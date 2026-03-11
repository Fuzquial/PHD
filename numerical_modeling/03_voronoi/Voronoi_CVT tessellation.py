# -*- coding: mbcs -*-
# Abaqus/CAE noGUI
# Voronoi/CVT tessellation "comme l'article" :
#  - Maillage d'abord
#  - Centroïdes d'éléments (eq.5)
#  - Scaling par ratio d'aspect (eq.6-7)
#  - NP par rapport de volumes (eq.8)
#  - Assignation élément -> grain par seed le plus proche (eq.9)
#  - Probabilistic Lloyd (sous-échantillonnage) pour rester calculable sur gros maillages
#
# + 5 types de grains (5 sections) assignées grain-par-grain (proportions)

from abaqus import mdb
from abaqusConstants import *
import regionToolset
import mesh
import random
import math
import time

# ============================================================
# PARAMETRES
# ============================================================

MODEL_NAME = 'Model-1'

# Cylindre
PART_NAME = 'Cyl3D'
INST_NAME = PART_NAME + '-1'
R = 25
H = 100.0

# Maillage
ELEMENT_SIZE = 0.5
PREFER_HEX = False
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# Microstructure (grain target size X,Y,Z)
# -> diminue ces valeurs => plus de grains => grains plus petits
GRAIN_SIZE_XYZ = (2, 2, 2)

# Lloyd (CVT)
RANDOM_SEED = 1234
LLOYD_ITERS = 20                # 4-8 suffisent souvent
MAX_SITES_FOR_LLOYD = 350000      # probabilistic Lloyd: 10k-30k conseillé pour 250k elems

# 5 types (5 sections)
NB_TYPES = 5
TYPE_PROPORTIONS = [0.30, 0.40, 0.20, 0.07, 0.03]
E_TYPES  = [50000., 52000., 48000., 51000., 49000.]
NU_TYPES = [0.25,   0.25,   0.25,   0.25,   0.25]

GRAIN_SET_PREFIX = 'ELSET_GRAIN_'

# ============================================================
# OUTILS
# ============================================================

def aspect_ratio_from_grain_size(grain_xyz):
    # eq.(6-7): r = d / min(d)  (min = 1)
    gmin = min(grain_xyz)
    r = (grain_xyz[0]/gmin, grain_xyz[1]/gmin, grain_xyz[2]/gmin)
    return r, gmin

def scale_point(p, r):
    # eq.(6): c~ = c / r
    return (p[0]/r[0], p[1]/r[1], p[2]/r[2])

def estimate_np_cylinder(R, H, r_aspect, equi_size):
    # eq.(8): NP = int( Vol(Ω~) / Vol(grain~) )
    # Ici Ω est un cylindre. Scaling divise le volume par rx*ry*rz.
    Vcyl = math.pi * (R**2) * H
    Vscaled = Vcyl / (r_aspect[0] * r_aspect[1] * r_aspect[2])
    Vgrain = equi_size**3
    return int(max(1, round(Vscaled / Vgrain)))

def dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]; dz = a[2]-b[2]
    return dx*dx + dy*dy + dz*dz

def nearest_seed_index(p, seeds):
    # eq.(9) : affectation au plus proche seed
    best_i = 0
    best_d2 = dist2(p, seeds[0])
    for i in range(1, len(seeds)):
        d2 = dist2(p, seeds[i])
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i

def build_node_map(part):
    return {n.label: n.coordinates for n in part.nodes}

def element_centroid(elem, node_map):
    # eq.(5): centroid = moyenne des coordonnées des noeuds (densité uniforme)
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
        pt = elem.pointOn[0]
        return (pt[0], pt[1], pt[2])

    inv = 1.0 / float(n_used)
    return (sx*inv, sy*inv, sz*inv)

def lloyd_cvt(points, n_seeds, n_iters, seed=0):
    # CVT Lloyd discretisé. Probabilistic Lloyd = points est un sous-échantillonnage des centroïdes.
    random.seed(seed)
    if n_seeds > len(points):
        n_seeds = len(points)
    n_seeds = max(1, n_seeds)

    seeds = [points[i] for i in random.sample(range(len(points)), n_seeds)]
    if n_iters <= 0:
        return seeds

    for it in range(n_iters):
        accum = [(0.0, 0.0, 0.0) for _ in range(n_seeds)]
        counts = [0 for _ in range(n_seeds)]

        for p in points:
            gi = nearest_seed_index(p, seeds)
            ax, ay, az = accum[gi]
            accum[gi] = (ax + p[0], ay + p[1], az + p[2])
            counts[gi] += 1

        new_seeds = []
        for i in range(n_seeds):
            if counts[i] > 0:
                ax, ay, az = accum[i]
                inv = 1.0 / float(counts[i])
                new_seeds.append((ax*inv, ay*inv, az*inv))
            else:
                new_seeds.append(points[random.randrange(0, len(points))])
        seeds = new_seeds

        print('--- Lloyd iter %d/%d' % (it+1, n_iters))

    return seeds

def weighted_choice_type(proportions):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(proportions):
        cum += p
        if r <= cum:
            return i+1
    return len(proportions)

def progress(tag, k, n, step=20000):
    if k == 0 or k == n-1 or (k % step == 0):
        print('%s %d/%d (%.1f%%)' % (tag, k, n, 100.0*float(k)/float(n)))

# ============================================================
# MAIN
# ============================================================

t0 = time.time()

# Model
if MODEL_NAME not in mdb.models:
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# Clean old part
if PART_NAME in mdl.parts:
    del mdl.parts[PART_NAME]

# Geometry
sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0*max(R, H))
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))
p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=H)
del mdl.sketches['__profile__']

# 5 sections
s = sum(TYPE_PROPORTIONS)
TYPE_PROPORTIONS = [pi/s for pi in TYPE_PROPORTIONS]

for t in range(NB_TYPES):
    mname = 'MatType-%d' % (t+1)
    sname = 'SecType-%d' % (t+1)
    if mname in mdl.materials:
        del mdl.materials[mname]
    if sname in mdl.sections:
        del mdl.sections[sname]
    mat = mdl.Material(name=mname)
    mat.Elastic(table=((E_TYPES[t], NU_TYPES[t]),))
    mdl.HomogeneousSolidSection(name=sname, material=mname)

# Mesh
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
        print('--- HEX SWEEP failed -> TET. Reason:', str(e))
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
print('--- Meshing done:', meshed_as)
print('--- Elements:', len(elems))

# Centroids (eq.5)
random.seed(RANDOM_SEED)
node_map = build_node_map(p)

centroids = []
nE = len(elems)
for k, e in enumerate(elems):
    if k % 20000 == 0:
        progress('--- Centroids', k, nE)
    centroids.append(element_centroid(e, node_map))
print('--- Centroids done.')

# Scaling (eq.6-7)
r_aspect, equi_size = aspect_ratio_from_grain_size(GRAIN_SIZE_XYZ)
centroids_scaled = [scale_point(c, r_aspect) for c in centroids]

# NP (eq.8)
NP = estimate_np_cylinder(R, H, r_aspect, equi_size)
print('--- Grain size XYZ:', GRAIN_SIZE_XYZ)
print('--- Aspect ratio r:', r_aspect, 'equiaxed size:', equi_size)
print('--- Estimated NP:', NP)

# Probabilistic Lloyd sites
sites_for_lloyd = centroids_scaled
if MAX_SITES_FOR_LLOYD is not None and MAX_SITES_FOR_LLOYD < len(centroids_scaled):
    idx = random.sample(range(len(centroids_scaled)), MAX_SITES_FOR_LLOYD)
    sites_for_lloyd = [centroids_scaled[i] for i in idx]
    print('--- Lloyd sites:', len(sites_for_lloyd), '/', len(centroids_scaled))

# Lloyd CVT
print('--- Lloyd iters:', LLOYD_ITERS)
seeds = lloyd_cvt(sites_for_lloyd, NP, LLOYD_ITERS, seed=RANDOM_SEED)

# Final assignment on ALL elements (eq.9)
grain_to_labels = [[] for _ in range(len(seeds))]
tA = time.time()
for k, e in enumerate(elems):
    if k % 20000 == 0:
        progress('--- Assign', k, nE)
    gi = nearest_seed_index(centroids_scaled[k], seeds)
    grain_to_labels[gi].append(e.label)
print('--- Assign done in %.1fs' % (time.time() - tA))

# Create ELSET per grain + assign 5 types
for sname in list(p.sets.keys()):
    if sname.startswith(GRAIN_SET_PREFIX):
        del p.sets[sname]

random.seed(RANDOM_SEED)
type_counts = [0]*NB_TYPES
nonempty = 0

for i, labels in enumerate(grain_to_labels):
    if not labels:
        continue
    nonempty += 1

    set_name = '%s%04d' % (GRAIN_SET_PREFIX, i+1)
    elems_i = p.elements.sequenceFromLabels(labels=tuple(labels))
    p.Set(name=set_name, elements=elems_i)

    t = weighted_choice_type(TYPE_PROPORTIONS)
    type_counts[t-1] += 1
    sec_name = 'SecType-%d' % t
    region = regionToolset.Region(elements=elems_i)
    p.SectionAssignment(region=region, sectionName=sec_name)

    if i % 500 == 0:
        print('--- Sets', i, '/', len(grain_to_labels))

print('--- Non-empty grains:', nonempty, '/', len(seeds))
print('--- Type counts:', type_counts)

# Assembly instance
a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
if INST_NAME in a.instances:
    del a.instances[INST_NAME]
a.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- TOTAL time %.1fs' % (time.time() - t0))
print('--- Done.')
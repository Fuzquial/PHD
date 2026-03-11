# -*- coding: utf-8 -*-
# Abaqus/CAE - Volume 3D + hétérogénéité type "grains" (Voronoi rapide)
# Version optimisée temps: pas de CVT, pas de SERVE, assignation accélérée par bins.

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# 1) PARAMETRES
# ============================================================

MODEL_NAME = 'Model-1'
PART_NAME  = 'Solid3D'
INST_NAME  = 'Solid3D-1'

# --- Géométrie (mm) : ici un bloc, remplaçable par ta vraie géométrie
Lx = 50.0
Ly = 50.0
Lz = 20.0   # épaisseur 3D (si tu veux un wedge/cylindre, dis-moi et je te le mets)

# --- Maillage
ELEMENT_SIZE = 1     # mets 0.2 ou 0.1 seulement si nécessaire (ça explose le nb d'éléments)
PREFER_HEX = True
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

# --- 3 minéraux : fractions cibles (volume ~ masse si densités identiques)
p_mass = (0.30, 0.50, 0.20)

materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# --- Microstructure
GRAIN_MEAN_DIAM = 3.0      # règle: ~ 3 à 6 * ELEMENT_SIZE (sinon pas granulaire)
USE_LAGUERRE = False       # True si tu veux dispersion de tailles (un peu plus “patchy”)
LAGUERRE_WEIGHT_STD = 0.2

RANDOM_SEED = 123

# --- Clamp N_SEEDS (clé pour le temps)
N_SEEDS_MIN = 200
N_SEEDS_MAX = 1200

# --- Binning (accélération)
BIN_SIZE = 1.5 * GRAIN_MEAN_DIAM

# ============================================================
# 2) OUTILS
# ============================================================

def element_volume(part, elem):
    try:
        return float(elem.getSize())
    except Exception:
        # fallback grossier
        xs, ys, zs = [], [], []
        for node_label in elem.connectivity:
            node_obj = part.nodes[node_label - 1]
            xs.append(node_obj.coordinates[0])
            ys.append(node_obj.coordinates[1])
            zs.append(node_obj.coordinates[2])
        return (max(xs)-min(xs))*(max(ys)-min(ys))*(max(zs)-min(zs))

def node_xyz(part, node_label):
    n = part.nodes[node_label - 1]
    return n.coordinates[0], n.coordinates[1], n.coordinates[2]

def element_centroid(part, elem):
    xs = ys = zs = 0.0
    nn = float(len(elem.connectivity))
    for nl in elem.connectivity:
        x, y, z = node_xyz(part, nl)
        xs += x; ys += y; zs += z
    return xs/nn, ys/nn, zs/nn

def clamp_int(a, lo, hi):
    if a < lo: return lo
    if a > hi: return hi
    return a

def random_point_in_box(Lx, Ly, Lz):
    return (Lx*random.random(), Ly*random.random(), Lz*random.random())

def safe_counts_for_seeds(n_seeds, p_mass):
    n_seeds = max(n_seeds, 3)
    n0 = int(round(p_mass[0]*n_seeds))
    n1 = int(round(p_mass[1]*n_seeds))
    n2 = n_seeds - n0 - n1
    counts = [n0, n1, n2]
    mins = [1 if p_mass[i] > 0.0 else 0 for i in range(3)]
    for i in range(3):
        if counts[i] < mins[i]:
            counts[i] = mins[i]
    while sum(counts) > n_seeds:
        i = max(range(3), key=lambda k: counts[k])
        if counts[i] > mins[i]:
            counts[i] -= 1
        else:
            break
    while sum(counts) < n_seeds:
        i = max(range(3), key=lambda k: p_mass[k])
        counts[i] += 1
    return counts[0], counts[1], counts[2]

def estimate_n_seeds(Lx, Ly, Lz, d_mean):
    V = Lx*Ly*Lz
    Vg = math.pi*(d_mean**3)/6.0
    n = int(1.5 * V / max(1e-12, Vg))  # facteur Voronoi
    return n

def generate_seeds(Lx, Ly, Lz, n_seeds, p_mass, d_mean, use_laguerre=False, w_std=0.2):
    n0, n1, n2 = safe_counts_for_seeds(n_seeds, p_mass)
    labels = [0]*n0 + [1]*n1 + [2]*n2
    random.shuffle(labels)

    seeds = []
    for lab in labels:
        x, y, z = random_point_in_box(Lx, Ly, Lz)
        if use_laguerre:
            rel = max(0.0, random.gauss(1.0, w_std))
            w = (rel*d_mean)**2
        else:
            w = 0.0
        seeds.append((x, y, z, w, lab))
    return seeds

def build_seed_bins(seeds, Lx, Ly, Lz, bin_size):
    bin_size = max(1e-12, bin_size)
    nx = max(1, int(math.ceil(Lx/bin_size)))
    ny = max(1, int(math.ceil(Ly/bin_size)))
    nz = max(1, int(math.ceil(Lz/bin_size)))
    bins = {}
    for idx, (sx, sy, sz, w, lab) in enumerate(seeds):
        ix = clamp_int(int(sx/bin_size), 0, nx-1)
        iy = clamp_int(int(sy/bin_size), 0, ny-1)
        iz = clamp_int(int(sz/bin_size), 0, nz-1)
        key = (ix, iy, iz)
        bins.setdefault(key, []).append(idx)
    return bins, nx, ny, nz

def assign_elements_binned(part, elems, seeds, bins, nx, ny, nz, Lx, Ly, Lz, bin_size, use_laguerre=False):
    elsets = [[], [], []]
    bin_size = max(1e-12, bin_size)

    for e in elems:
        cx, cy, cz = element_centroid(part, e)
        ix = clamp_int(int(cx/bin_size), 0, nx-1)
        iy = clamp_int(int(cy/bin_size), 0, ny-1)
        iz = clamp_int(int(cz/bin_size), 0, nz-1)

        best_val = 1e99
        best_lab = 0
        found = False

        # 27 voisins
        for dx in (-1, 0, 1):
            jx = ix + dx
            if jx < 0 or jx >= nx: continue
            for dy in (-1, 0, 1):
                jy = iy + dy
                if jy < 0 or jy >= ny: continue
                for dz in (-1, 0, 1):
                    jz = iz + dz
                    if jz < 0 or jz >= nz: continue
                    key = (jx, jy, jz)
                    if key not in bins: continue
                    for sidx in bins[key]:
                        sx, sy, sz, w, lab = seeds[sidx]
                        dxp = cx - sx; dyp = cy - sy; dzp = cz - sz
                        d2 = dxp*dxp + dyp*dyp + dzp*dzp
                        val = d2 - w if use_laguerre else d2
                        if val < best_val:
                            best_val = val
                            best_lab = lab
                            found = True

        # fallback global (rare)
        if not found:
            for (sx, sy, sz, w, lab) in seeds:
                dxp = cx - sx; dyp = cy - sy; dzp = cz - sz
                d2 = dxp*dxp + dyp*dyp + dzp*dzp
                val = d2 - w if use_laguerre else d2
                if val < best_val:
                    best_val = val
                    best_lab = lab

        elsets[best_lab].append(e)

    return elsets

# ============================================================
# 3) MODELE + PART
# ============================================================

if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

if PART_NAME in mdl.parts.keys():
    del mdl.parts[PART_NAME]

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0*max(Lx, Ly, Lz))
sk.rectangle(point1=(0.0, 0.0), point2=(Lx, Ly))
p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=Lz)
del mdl.sketches['__profile__']

# ============================================================
# 4) MATERIAUX + SECTIONS
# ============================================================

for md in materials:
    if md['name'] in mdl.materials.keys():
        del mdl.materials[md['name']]

for i in range(1, 4):
    sname = 'Section-%d' % i
    if sname in mdl.sections.keys():
        del mdl.sections[sname]

for i, md in enumerate(materials, start=1):
    mat = mdl.Material(name=md['name'])
    mat.Density(table=((md['rho'],),))
    mat.Elastic(table=((md['E'], md['nu']),))
    mdl.HomogeneousSolidSection(name='Section-%d' % i, material=md['name'])

# ============================================================
# 5) MAILLAGE
# ============================================================

p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
cells = p.cells[:]
meshed_as = None

if PREFER_HEX:
    try:
        p.setMeshControls(regions=cells, technique=SWEEP, elemShape=HEX)
        elemType = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
        p.setElementType(regions=(cells,), elemTypes=(elemType,))
        p.generateMesh()
        meshed_as = 'HEX'
    except Exception as e:
        print('--- HEX failed -> TET. Reason:', str(e))
        meshed_as = None

if meshed_as is None:
    p.deleteMesh()
    p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    p.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
    elemType = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
    p.setElementType(regions=(cells,), elemTypes=(elemType,))
    p.generateMesh()
    meshed_as = 'TET'

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

V_tot = 0.0
for e in elems:
    V_tot += element_volume(p, e)

print('==============================')
print('Meshed as       =', meshed_as)
print('Nb elements     =', len(elems))
print('Element size    =', ELEMENT_SIZE)
print('Grain d_mean    =', GRAIN_MEAN_DIAM)
print('==============================')

# ============================================================
# 6) MICROSTRUCTURE: Voronoi rapide
# ============================================================

random.seed(RANDOM_SEED)

N_SEEDS = estimate_n_seeds(Lx, Ly, Lz, GRAIN_MEAN_DIAM)
N_SEEDS = max(N_SEEDS_MIN, min(N_SEEDS_MAX, N_SEEDS))

print('N_SEEDS used    =', N_SEEDS)
print('BIN_SIZE        =', BIN_SIZE)

seeds = generate_seeds(Lx, Ly, Lz, N_SEEDS, p_mass, GRAIN_MEAN_DIAM,
                       use_laguerre=USE_LAGUERRE,
                       w_std=LAGUERRE_WEIGHT_STD)

bins, nx, ny, nz = build_seed_bins(seeds, Lx, Ly, Lz, BIN_SIZE)

elsets = assign_elements_binned(p, elems, seeds, bins, nx, ny, nz,
                                Lx, Ly, Lz, BIN_SIZE, use_laguerre=USE_LAGUERRE)

for k in range(3):
    if len(elsets[k]) == 0:
        raise RuntimeError("Mineral-%d vide -> augmente N_SEEDS_MAX ou augmente GRAIN_MEAN_DIAM." % (k+1))

# ============================================================
# 7) ELSETS + ASSIGNATION SECTIONS
# ============================================================

set_names = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']
for sname in set_names:
    if sname in p.sets.keys():
        del p.sets[sname]

for k in range(3):
    labels = [e.label for e in elsets[k]]
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=set_names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 3 minéraux assignés.')

# ============================================================
# 8) ASSEMBLY
# ============================================================

a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

if INST_NAME in a.instances.keys():
    del a.instances[INST_NAME]

a.Instance(name=INST_NAME, part=p, dependent=ON)
print('--- Instance créée :', INST_NAME)
print('--- Script terminé.')
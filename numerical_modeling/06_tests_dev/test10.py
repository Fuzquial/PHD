# -*- coding: utf-8 -*-
# Abaqus/CAE - Bloc 3D hétérogène (3 minéraux)
# Microstructure grain-scale par Voronoi/Laguerre-Voronoi
# Seeds CVT-like (Lloyd light) + assignation accélérée (binning)
# Inspiré de l'approche "mesh + nearest seed classification" de l'article p691 :contentReference[oaicite:1]{index=1}
#
# Unités : mm - MPa - tonne

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# 1) PARAMETRES UTILISATEUR
# ============================================================

MODEL_NAME = 'Model-1'
PART_NAME  = 'Block3D'
INST_NAME  = 'Block3D-1'

# Géométrie bloc (mm)
Lx = 10.0
Ly = 6.0
Lz = 8.0

# Maillage
ELEMENT_SIZE = 0.5
PREFER_HEX = True     # False si tu veux casser un rendu "damier/stries"
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# Fractions cibles (somme=1)
p_mass = (0.30, 0.50, 0.20)

# Matériaux (E en MPa, nu, rho en t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# ============================================================
# 1bis) MICROSTRUCTURE
# ============================================================

MICROSTRUCTURE_MODE = 'VORONOI'  # 'VORONOI' ou 'SERVE'

# Taille moyenne de grain (mm)
# Avec ELEMENT_SIZE=0.1, 0.5 mm est OK, mais coûteux. 0.8–1.2 est plus rapide.
GRAIN_MEAN_DIAM = 0.80

# Laguerre (power diagram) -> tailles un peu variées
USE_LAGUERRE = True
LAGUERRE_WEIGHT_STD = 0.20

# SERVE-like : nb réalisations (réduis si maillage fin)
N_REALISATIONS = 1
SEED_BASE = 123
RANDOM_SEED = 0

# --- CVT-like (Lloyd light) : itérations + nb samples (impacte le temps de génération des seeds)
CVT_N_ITER = 1
CVT_N_SAMPLES = 25000   # 20k–60k typique. Monte si tu veux seeds très régulières.

# --- Accélération (binning) : taille de bin (mm)
# Reco: ~ 1 à 2 x d_mean
BIN_SIZE = 1.5 * GRAIN_MEAN_DIAM

# --- Clamp N_SEEDS pour ne pas exploser le temps
N_SEEDS_MIN = 300
N_SEEDS_MAX = 800

# ============================================================
# 2) OUTILS DE BASE
# ============================================================

def element_volume(part, elem):
    try:
        return float(elem.getSize())
    except Exception:
        xs, ys, zs = [], [], []
        for node_label in elem.connectivity:
            node_obj = part.nodes[node_label - 1]
            xs.append(node_obj.coordinates[0])
            ys.append(node_obj.coordinates[1])
            zs.append(node_obj.coordinates[2])
        return (max(xs)-min(xs)) * (max(ys)-min(ys)) * (max(zs)-min(zs))

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

def vol_list_total(part, elem_list):
    s = 0.0
    for e in elem_list:
        s += element_volume(part, e)
    return s

def random_point_in_block(Lx, Ly, Lz):
    return (Lx * random.random(),
            Ly * random.random(),
            Lz * random.random())

def safe_counts_for_seeds(n_seeds, p_mass):
    """Garantit au moins 1 seed si fraction>0."""
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

def frac_error(V_frac, p_target):
    return abs(V_frac[0]-p_target[0]) + abs(V_frac[1]-p_target[1]) + abs(V_frac[2]-p_target[2])

# ============================================================
# 3) ACCELERATION: BINNING (grille)
# ============================================================

def clamp_int(a, lo, hi):
    if a < lo: return lo
    if a > hi: return hi
    return a

def build_seed_bins(seeds, Lx, Ly, Lz, bin_size):
    bin_size = max(1e-12, bin_size)
    nx = max(1, int(math.ceil(Lx / bin_size)))
    ny = max(1, int(math.ceil(Ly / bin_size)))
    nz = max(1, int(math.ceil(Lz / bin_size)))
    bins = {}
    for idx, (sx, sy, sz, w, lab) in enumerate(seeds):
        ix = clamp_int(int(sx / bin_size), 0, nx-1)
        iy = clamp_int(int(sy / bin_size), 0, ny-1)
        iz = clamp_int(int(sz / bin_size), 0, nz-1)
        key = (ix, iy, iz)
        if key not in bins:
            bins[key] = []
        bins[key].append(idx)
    return bins, nx, ny, nz

def assign_elements_to_seed_mineral_binned(part, elems, seeds, bins, nx, ny, nz,
                                          Lx, Ly, Lz, bin_size, use_laguerre=False):
    elsets = [[], [], []]
    bin_size = max(1e-12, bin_size)

    for e in elems:
        cx, cy, cz = element_centroid(part, e)
        ix = clamp_int(int(cx / bin_size), 0, nx-1)
        iy = clamp_int(int(cy / bin_size), 0, ny-1)
        iz = clamp_int(int(cz / bin_size), 0, nz-1)

        best_val = 1.0e99
        best_lab = 0
        found = False

        # 27 bins voisins
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
                    if key not in bins:
                        continue
                    for sidx in bins[key]:
                        sx, sy, sz, w, lab = seeds[sidx]
                        ddx = cx - sx; ddy = cy - sy; ddz = cz - sz
                        d2 = ddx*ddx + ddy*ddy + ddz*ddz
                        val = d2 - w if use_laguerre else d2
                        if val < best_val:
                            best_val = val
                            best_lab = lab
                            found = True

        # fallback global (rare)
        if not found:
            for (sx, sy, sz, w, lab) in seeds:
                ddx = cx - sx; ddy = cy - sy; ddz = cz - sz
                d2 = ddx*ddx + ddy*ddy + ddz*ddz
                val = d2 - w if use_laguerre else d2
                if val < best_val:
                    best_val = val
                    best_lab = lab

        elsets[best_lab].append(e)

    return elsets

# ============================================================
# 4) SEEDS CVT-LIKE (Lloyd light) + bins
# ============================================================

def nearest_seed_index_binned(px, py, pz, seeds, bins, nx, ny, nz, bin_size, use_laguerre=False):
    bin_size = max(1e-12, bin_size)
    ix = clamp_int(int(px / bin_size), 0, nx-1)
    iy = clamp_int(int(py / bin_size), 0, ny-1)
    iz = clamp_int(int(pz / bin_size), 0, nz-1)

    best = 0
    best_val = 1e99
    found = False

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
                if key not in bins:
                    continue
                for sidx in bins[key]:
                    sx, sy, sz, w, lab = seeds[sidx]
                    dxp = px - sx; dyp = py - sy; dzp = pz - sz
                    d2 = dxp*dxp + dyp*dyp + dzp*dzp
                    val = d2 - w if use_laguerre else d2
                    if val < best_val:
                        best_val = val
                        best = sidx
                        found = True

    if not found:
        # fallback global (rare)
        for i, (sx, sy, sz, w, lab) in enumerate(seeds):
            dxp = px - sx; dyp = py - sy; dzp = pz - sz
            d2 = dxp*dxp + dyp*dyp + dzp*dzp
            val = d2 - w if use_laguerre else d2
            if val < best_val:
                best_val = val
                best = i

    return best

def generate_cvt_seeds_block(Lx, Ly, Lz, n_seeds, p_mass, d_mean,
                             use_laguerre=False, w_std=0.2,
                             n_iter=6, n_samples=25000,
                             bin_size=None):
    """
    CVT-like (Lloyd light):
    - init seeds random
    - itérations: assignation de points échantillonnés -> recentrage seeds
    Accéléré via bins lors du nearest-seed sur les samples.
    """
    n0, n1, n2 = safe_counts_for_seeds(n_seeds, p_mass)
    labels = [0]*n0 + [1]*n1 + [2]*n2
    random.shuffle(labels)

    seeds = []
    for lab in labels:
        x, y, z = random_point_in_block(Lx, Ly, Lz)
        if use_laguerre:
            rel = max(0.0, random.gauss(1.0, w_std))
            w = (rel * d_mean)**2
        else:
            w = 0.0
        seeds.append([x, y, z, w, lab])  # mutable

    if bin_size is None:
        bin_size = max(1e-9, 1.5*d_mean)

    for it in range(n_iter):
        # rebuild bins (seeds bougent)
        seeds_tuple = [(s[0], s[1], s[2], s[3], s[4]) for s in seeds]
        bins, nx, ny, nz = build_seed_bins(seeds_tuple, Lx, Ly, Lz, bin_size)

        sx_sum = [0.0]*len(seeds)
        sy_sum = [0.0]*len(seeds)
        sz_sum = [0.0]*len(seeds)
        cnt    = [0]*len(seeds)

        for _ in range(n_samples):
            px, py, pz = random_point_in_block(Lx, Ly, Lz)
            k = nearest_seed_index_binned(px, py, pz, seeds_tuple, bins, nx, ny, nz, bin_size,
                                          use_laguerre=use_laguerre)
            sx_sum[k] += px; sy_sum[k] += py; sz_sum[k] += pz
            cnt[k] += 1

        # update seeds position
        for k in range(len(seeds)):
            if cnt[k] > 0:
                seeds[k][0] = sx_sum[k] / cnt[k]
                seeds[k][1] = sy_sum[k] / cnt[k]
                seeds[k][2] = sz_sum[k] / cnt[k]

    return [(s[0], s[1], s[2], s[3], s[4]) for s in seeds]

# ============================================================
# 5) CREATION MODELE + PART 3D
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
# 6) MATERIAUX + SECTIONS
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
# 7) MAILLAGE
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
        meshed_as = 'HEX (SWEEP)'
    except Exception as e:
        print('--- HEX SWEEP failed, fallback to TET. Reason:', str(e))
        meshed_as = None

if meshed_as is None:
    p.deleteMesh()
    p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    p.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
    elemType = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
    p.setElementType(regions=(cells,), elemTypes=(elemType,))
    p.generateMesh()
    meshed_as = 'TET (FREE)'

print('--- Meshing done:', meshed_as)

# ============================================================
# 8) MICROSTRUCTURE (CVT seeds + assignation binned)
# ============================================================

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

# Volume total
V_tot = 0.0
for e in elems:
    V_tot += element_volume(p, e)

# N_SEEDS estimé + clamp (CRUCIAL pour le temps)
V = Lx * Ly * Lz
Vg = math.pi * (GRAIN_MEAN_DIAM**3) / 6.0
N_SEEDS = int(1.5 * V / max(1e-12, Vg))
N_SEEDS = max(N_SEEDS_MIN, min(N_SEEDS_MAX, N_SEEDS))

print('==============================')
print('Meshed as        =', meshed_as)
print('Nb elements      =', len(elems))
print('Element size     =', ELEMENT_SIZE)
print('Grain d_mean     =', GRAIN_MEAN_DIAM)
print('N_SEEDS (clamped)=', N_SEEDS)
print('Laguerre         =', USE_LAGUERRE)
print('BIN_SIZE         =', BIN_SIZE)
print('CVT iters/samples=', CVT_N_ITER, '/', CVT_N_SAMPLES)
print('Mode             =', MICROSTRUCTURE_MODE)
print('==============================')

def one_realisation(seed_value):
    random.seed(seed_value)

    # 1) seeds CVT-like (plus régulières)
    seeds = generate_cvt_seeds_block(
        Lx, Ly, Lz, N_SEEDS, p_mass, GRAIN_MEAN_DIAM,
        use_laguerre=USE_LAGUERRE,
        w_std=LAGUERRE_WEIGHT_STD,
        n_iter=CVT_N_ITER,
        n_samples=CVT_N_SAMPLES,
        bin_size=BIN_SIZE
    )

    # 2) bins seeds (pour assignation rapide des éléments)
    bins, nx, ny, nz = build_seed_bins(seeds, Lx, Ly, Lz, BIN_SIZE)

    # 3) assignation éléments -> minéral (nearest seed local)
    elsets = assign_elements_to_seed_mineral_binned(
        p, elems, seeds, bins, nx, ny, nz,
        Lx, Ly, Lz, BIN_SIZE,
        use_laguerre=USE_LAGUERRE
    )

    V_used = [vol_list_total(p, elsets[k]) for k in range(3)]
    V_frac = [V_used[k]/V_tot for k in range(3)]
    err = frac_error(V_frac, p_mass)
    return elsets, V_used, V_frac, err

if MICROSTRUCTURE_MODE.upper() == 'VORONOI':
    elsets, V_used, V_frac, err = one_realisation(RANDOM_SEED)
    print('V_frac  =', V_frac, ' target=', p_mass, ' err=', err)

elif MICROSTRUCTURE_MODE.upper() == 'SERVE':
    best = None
    best_err = 1e99
    for it in range(N_REALISATIONS):
        elsets_i, V_used_i, V_frac_i, err_i = one_realisation(SEED_BASE + it)
        if err_i < best_err:
            best_err = err_i
            best = (elsets_i, V_used_i, V_frac_i, err_i)
    elsets, V_used, V_frac, err = best
    print('SERVE best err =', err)
    print('V_frac         =', V_frac, ' target=', p_mass)

else:
    raise RuntimeError("MICROSTRUCTURE_MODE doit être 'VORONOI' ou 'SERVE'.")

# sécurité : minéraux non vides
for k in range(3):
    if len(elsets[k]) == 0:
        raise RuntimeError("Mineral-%d vide. Augmente N_SEEDS ou diminue GRAIN_MEAN_DIAM." % (k+1))

# ============================================================
# 9) ELSETS + ASSIGNATION SECTIONS
# ============================================================

for sname in ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']:
    if sname in p.sets.keys():
        del p.sets[sname]

names = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']

for k in range(3):
    labels = [e.label for e in elsets[k]]
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 3 minéraux assignés (CVT + Voronoi binned).')

# ============================================================
# 10) ASSEMBLY + INSTANCE
# ============================================================

a_asm = mdl.rootAssembly
a_asm.DatumCsysByDefault(CARTESIAN)

if INST_NAME in a_asm.instances.keys():
    del a_asm.instances[INST_NAME]

a_asm.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Instance créée :', INST_NAME)
print('--- Script terminé.')
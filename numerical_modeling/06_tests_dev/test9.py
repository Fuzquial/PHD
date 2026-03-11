# -*- coding: utf-8 -*-
# Abaqus/CAE - Bloc 3D hétérogène (3 minéraux)
# Grain-scale Voronoi / Laguerre-Voronoi (power diagram)
# Option SERVE-like : plusieurs réalisations, on garde la meilleure (fractions volumiques)
#
# Unités : mm - MPa - tonne (comme ton script initial)

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
ELEMENT_SIZE = 0.1
PREFER_HEX = True              # mets False si tu veux casser un éventuel effet "stries" visuel
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# Fractions massiques cibles (somme=1)
# (si densités identiques -> fractions volumiques ~ identiques)
p_mass = (0.30, 0.50, 0.20)

# Matériaux (E en MPa, nu, rho en t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# ============================================================
# 1bis) MICROSTRUCTURE (grain-scale)
# ============================================================

MICROSTRUCTURE_MODE = 'SERVE'   # 'VORONOI' ou 'SERVE'

# Taille moyenne de grain visée (mm)
# Règle simple : vise ~ 3 à 6 x ELEMENT_SIZE
GRAIN_MEAN_DIAM = 0.50

# Laguerre-Voronoi (power diagram) pour contrôler dispersion tailles
USE_LAGUERRE = True
LAGUERRE_WEIGHT_STD = 0.20

# SERVE-like (nb réalisations testées)
N_REALISATIONS = 25
SEED_BASE = 123
RANDOM_SEED = 0

# ============================================================
# 2) OUTILS
# ============================================================

def element_volume(part, elem):
    """Volume d'élément via Abaqus si possible."""
    try:
        return float(elem.getSize())
    except Exception:
        # fallback très grossier (ne devrait pas servir)
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

def estimate_n_seeds_block(Lx, Ly, Lz, d_mean):
    V = Lx * Ly * Lz
    Vg = math.pi * (d_mean**3) / 6.0  # sphère approx
    n = int(max(10, V / max(1e-12, Vg)))
    return n

def safe_counts_for_seeds(n_seeds, p_mass):
    """
    Renvoie (n0,n1,n2) tel que:
    - somme = n_seeds
    - si p_mass[i] > 0 => ni >= 1 (évite "minéral vide" par absence de seeds)
    """
    n_seeds = max(n_seeds, 3)

    # initial
    n0 = int(round(p_mass[0]*n_seeds))
    n1 = int(round(p_mass[1]*n_seeds))
    n2 = n_seeds - n0 - n1
    counts = [n0, n1, n2]

    mins = [1 if p_mass[i] > 0.0 else 0 for i in range(3)]
    for i in range(3):
        if counts[i] < mins[i]:
            counts[i] = mins[i]

    # corrige somme si trop
    while sum(counts) > n_seeds:
        i = max(range(3), key=lambda k: counts[k])
        if counts[i] > mins[i]:
            counts[i] -= 1
        else:
            # essaie autre phase
            jlist = sorted(range(3), key=lambda k: counts[k], reverse=True)
            changed = False
            for j in jlist:
                if counts[j] > mins[j]:
                    counts[j] -= 1
                    changed = True
                    break
            if not changed:
                break

    # corrige somme si pas assez
    while sum(counts) < n_seeds:
        i = max(range(3), key=lambda k: p_mass[k])
        counts[i] += 1

    return counts[0], counts[1], counts[2]

def generate_seeds_with_minerals_block(Lx, Ly, Lz, n_seeds, p_mass, d_mean,
                                      use_laguerre=False, w_std=0.2):
    """
    Seeds étiquetées par minéral (0/1/2). Voronoi / Laguerre-Voronoi.
    Version robuste: garantit au moins une seed par phase non nulle.
    """
    n0, n1, n2 = safe_counts_for_seeds(n_seeds, p_mass)
    print('--- seeds per mineral:', (n0, n1, n2), '/ N_SEEDS=', n_seeds)

    labels = [0]*n0 + [1]*n1 + [2]*n2
    random.shuffle(labels)

    seeds = []
    for lab in labels:
        x, y, z = random_point_in_block(Lx, Ly, Lz)

        if use_laguerre:
            rel = max(0.0, random.gauss(1.0, w_std))
            # IMPORTANT: échelle sur d_mean (sinon poids trop énormes)
            w = (rel * d_mean)**2
        else:
            w = 0.0

        seeds.append((x, y, z, w, lab))

    return seeds

def assign_elements_to_seed_mineral(part, elems, seeds, use_laguerre=False):
    """
    Chaque élément -> graine la plus proche -> minéral de cette graine.
    """
    elsets = [[], [], []]
    for e in elems:
        cx, cy, cz = element_centroid(part, e)

        best_val = 1.0e99
        best_lab = 0

        for (sx, sy, sz, w, lab) in seeds:
            dx = cx - sx; dy = cy - sy; dz = cz - sz
            d2 = dx*dx + dy*dy + dz*dz
            val = d2 - w if use_laguerre else d2
            if val < best_val:
                best_val = val
                best_lab = lab

        elsets[best_lab].append(e)
    return elsets

def frac_error(V_frac, p_target):
    return abs(V_frac[0]-p_target[0]) + abs(V_frac[1]-p_target[1]) + abs(V_frac[2]-p_target[2])

# ============================================================
# 3) CREATION MODELE + PART 3D BLOC
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
# 5) MAILLAGE (HEX si possible sinon TET)
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
# 6-7) MICROSTRUCTURE: Voronoi / SERVE-like
# ============================================================

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

# Volume total
V_tot = 0.0
for e in elems:
    V_tot += element_volume(p, e)

# Estimation nb grains + sécurité minimum (évite cas dégénérés)
N_SEEDS = estimate_n_seeds_block(Lx, Ly, Lz, GRAIN_MEAN_DIAM)
N_SEEDS = max(50, N_SEEDS)  # <-- très important pour éviter minéraux vides
V = Lx * Ly * Lz
Vg = math.pi * (GRAIN_MEAN_DIAM**3) / 6.0
N_SEEDS = int(V / Vg)

# facteur correctif car Voronoi ≠ sphères parfaites
N_SEEDS = int(1.5 * N_SEEDS)
print('==============================')
print('Meshed as     =', meshed_as)
print('Nb elements   =', len(elems))
print('V_tot         =', V_tot)
print('Element size  =', ELEMENT_SIZE)
print('Mode          =', MICROSTRUCTURE_MODE)
print('Grain d_mean  =', GRAIN_MEAN_DIAM)
print('N_SEEDS       =', N_SEEDS)
print('Laguerre      =', USE_LAGUERRE)
print('==============================')

def one_realisation():
    seeds = generate_seeds_with_minerals_block(
        Lx, Ly, Lz, N_SEEDS, p_mass, GRAIN_MEAN_DIAM,
        use_laguerre=USE_LAGUERRE,
        w_std=LAGUERRE_WEIGHT_STD
    )
    elsets = assign_elements_to_seed_mineral(p, elems, seeds, use_laguerre=USE_LAGUERRE)
    V_used = [vol_list_total(p, elsets[k]) for k in range(3)]
    V_frac = [V_used[k]/V_tot for k in range(3)]
    err = frac_error(V_frac, p_mass)
    return elsets, V_used, V_frac, err

if MICROSTRUCTURE_MODE.upper() == 'VORONOI':
    random.seed(RANDOM_SEED)
    elsets, V_used, V_frac, err = one_realisation()
    print('V_frac        =', V_frac, ' target=', p_mass, ' err=', err)

elif MICROSTRUCTURE_MODE.upper() == 'SERVE':
    best = None
    best_err = 1.0e99

    for it in range(N_REALISATIONS):
        random.seed(SEED_BASE + it)
        elsets_i, V_used_i, V_frac_i, err_i = one_realisation()
        if err_i < best_err:
            best_err = err_i
            best = (elsets_i, V_used_i, V_frac_i, err_i)

    elsets, V_used, V_frac, err = best
    print('SERVE best err=', err)
    print('V_frac        =', V_frac, ' target=', p_mass)

else:
    raise RuntimeError("MICROSTRUCTURE_MODE doit être 'VORONOI' ou 'SERVE'.")

# Sécurité supplémentaire : si un minéral a 0 élément, c'est que le maillage est trop grossier
# ou que N_SEEDS / GRAIN_MEAN_DIAM sont incohérents.
for k in range(3):
    if len(elsets[k]) == 0:
        raise RuntimeError("Mineral-%d vide. Augmente N_SEEDS ou diminue GRAIN_MEAN_DIAM ou raffine le maillage." % (k+1))

# ============================================================
# 8) ELSETS + ASSIGNATION SECTIONS
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

print('--- OK : 3 minéraux assignés (grain-scale Voronoi/Laguerre).')

# ============================================================
# 9) ASSEMBLY + INSTANCE
# ============================================================

a_asm = mdl.rootAssembly
a_asm.DatumCsysByDefault(CARTESIAN)

if INST_NAME in a_asm.instances.keys():
    del a_asm.instances[INST_NAME]

a_asm.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Instance créée :', INST_NAME)
print('--- Script terminé.')
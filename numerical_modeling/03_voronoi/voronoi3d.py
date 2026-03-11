# -*- coding: utf-8 -*-
# Abaqus/CAE - Prisme rectangle 3D (long) hétérogène (3 minéraux)
# Voronoï 3D (tessellation) via "nearest seed" sur centroïdes d'éléments
# + assignation 3 minéraux par regroupement de grains (fractions p_mass)
#
# Sans numpy (compat Abaqus Python)

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
PART_NAME  = 'Rect3D'
INST_NAME  = 'Rect3D-1'

# Géométrie prisme rectangle (mm) : [0,Lx] x [0,Ly] x [0,Lz]
Lx = 40.0   # long
Ly = 10.0
Lz = 5.0

# Maillage Abaqus
ELEMENT_SIZE = 0.1
PREFER_HEX = True
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# Voronoï 3D
N_GRAINS = 50
RANDOM_SEED = 0

# 3 minéraux : fractions massiques (somme=1)
p_mass = (0.30, 0.50, 0.20)

# Matériaux (E MPa, nu, rho t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# Option: créer un set par grain (peut faire beaucoup de sets)
CREATE_GRAIN_SETS = False

# ============================================================
# 2) OUTILS
# ============================================================

def node_xyz(part, node_label):
    n = part.nodes[node_label - 1]
    return n.coordinates[0], n.coordinates[1], n.coordinates[2]

def element_centroid(part, elem):
    sx = sy = sz = 0.0
    nn = float(len(elem.connectivity))
    for nl in elem.connectivity:
        x, y, z = node_xyz(part, nl)
        sx += x; sy += y; sz += z
    return sx/nn, sy/nn, sz/nn

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

def random_point_in_box(rng, Lx, Ly, Lz):
    x = Lx * rng.random()
    y = Ly * rng.random()
    z = Lz * rng.random()
    return x, y, z

def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx*dx + dy*dy + dz*dz

def assign_grain_id(pt, seeds):
    best = 0
    bestd = dist2(pt, seeds[0])
    for i in range(1, len(seeds)):
        d = dist2(pt, seeds[i])
        if d < bestd:
            bestd = d
            best = i
    return best

def all_rhos_equal(mats, tol=1e-30):
    rhos = [m['rho'] for m in mats]
    return (max(rhos) - min(rhos)) < tol

# ============================================================
# 3) CREATION MODELE + PART (PRISME RECTANGLE)
# ============================================================

if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

if PART_NAME in mdl.parts.keys():
    del mdl.parts[PART_NAME]

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0 * max(Lx, Ly, Lz))
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
# 5) MAILLAGE Abaqus
# ============================================================

p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
cells = p.cells[:]
meshed_as = None

if PREFER_HEX:
    try:
        # Pour un prisme extrudé, SWEEP HEX marche généralement très bien
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

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

print('==============================')
print('Meshed as     =', meshed_as)
print('Nb elements   =', len(elems))
print('Element size  =', ELEMENT_SIZE)
print('Box dims      =', (Lx, Ly, Lz))
print('N_GRAINS      =', N_GRAINS)
print('==============================')

# ============================================================
# 6) VORONOI 3D : seeds + assignation grain par centroïde
# ============================================================

rng = random.Random(RANDOM_SEED)

seeds = []
for i in range(N_GRAINS):
    seeds.append(random_point_in_box(rng, Lx, Ly, Lz))

grain_elems = [[] for _ in range(N_GRAINS)]
grain_vol   = [0.0 for _ in range(N_GRAINS)]
V_tot = 0.0

for e in elems:
    c = element_centroid(p, e)
    gid = assign_grain_id(c, seeds)
    grain_elems[gid].append(e)
    v = element_volume(p, e)
    grain_vol[gid] += v
    V_tot += v

empty = 0
for g in range(N_GRAINS):
    if len(grain_elems[g]) == 0:
        empty += 1
if empty > 0:
    print('--- WARNING: %d grains vides. Diminue N_GRAINS ou affine le maillage.' % empty)

print('--- Voronoi mapping done. V_tot =', V_tot)

# ============================================================
# 7) ASSIGNATION 3 MINERAUX (fractions sur volume)
# ============================================================

# si rhos égales -> volume = mass
if not all_rhos_equal(materials):
    print('--- WARNING: rho differents -> ici on vise quand même fractions V (approx).')

targets = [p_mass[i] * V_tot for i in range(3)]  # volume cible
order = list(range(N_GRAINS))
order.sort(key=lambda g: grain_vol[g], reverse=True)

mineral_grains = [[], [], []]
V_acc = [0.0, 0.0, 0.0]

for g in order:
    vg = grain_vol[g]
    if vg <= 0.0:
        continue
    if V_acc[0] < targets[0]:
        mineral_grains[0].append(g); V_acc[0] += vg
    elif V_acc[1] < targets[1]:
        mineral_grains[1].append(g); V_acc[1] += vg
    else:
        mineral_grains[2].append(g); V_acc[2] += vg

print('--- Mineral target V =', targets)
print('--- Mineral used   V =', V_acc)
print('--- Mineral frac ~   =', [V_acc[i]/V_tot for i in range(3)])

# éléments par minéral
elsets = [[], [], []]
for m in range(3):
    for g in mineral_grains[m]:
        elsets[m].extend(grain_elems[g])

# ============================================================
# 8) ELSETS + ASSIGNATION SECTIONS
# ============================================================

for sname in ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']:
    if sname in p.sets.keys():
        del p.sets[sname]

names = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']

for k in range(3):
    labels = [e.label for e in elsets[k]]
    if len(labels) == 0:
        raise RuntimeError("Mineral-%d vide. Ajuste p_mass ou N_GRAINS." % (k+1))
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 3 minéraux assignés via Voronoï 3D sur prisme rectangle.')

# Option : sets par grain
if CREATE_GRAIN_SETS:
    for g in range(N_GRAINS):
        if len(grain_elems[g]) == 0:
            continue
        sname = 'ELSET_GRAIN_%04d' % (g+1)
        if sname in p.sets.keys():
            del p.sets[sname]
        labels = [e.label for e in grain_elems[g]]
        elems_g = p.elements.sequenceFromLabels(labels=labels)
        p.Set(name=sname, elements=elems_g)
    print('--- Grain sets created.')

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
# -*- coding: utf-8 -*-
# Abaqus/CAE - Cylindre 3D hétérogène (3 minéraux)
# Champ gaussien par superposition d'ondes cosinus (Cahn/Lantuéjoul)
# + seuillage par ELEMENTS (valeur élément = moyenne nodale)
# + respect de fractions (ici, rho identiques => fractions massiques == fractions volumiques)
#
# Unités conseillées (comme l'article) : mm - MPa - tonne
# => E en MPa, rho en t/mm^3, géométrie en mm

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
PART_NAME  = 'Cyl3D'
INST_NAME  = 'Cyl3D-1'

# Géométrie cylindre (mm)
R = 2.5    # mm
H = 10     # mm

# Maillage
ELEMENT_SIZE = 0.5    # mm
PREFER_HEX = True
ELEM_CODE_HEX = C3D8   # intégration complète
ELEM_CODE_TET = C3D4

# Champ aléatoire (ondes cos)
RANDOM_SEED = 0
a = 8               # mm : taille caractéristique hétérogénéité (augmente pour "taches" plus grosses)
Nwaves = 1000          # nombre d'ondes

# 3 minéraux : fractions MASSIQUES cibles (somme=1)
p_mass = (0.30, 0.50, 0.20)

# Paramètres matériaux (E en MPa, nu, rho en t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# Tri : Mineral-1 correspond aux petites valeurs du champ ?
MIN1_IS_LOW = True

# ============================================================
# 2) OUTILS
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

def all_rhos_equal(mats, tol=1e-30):
    rhos = [m['rho'] for m in mats]
    return (max(rhos) - min(rhos)) < tol

def compute_targets(V_tot, mats, p_mass):
    """
    - Si rhos identiques (cas de ton tableau): fractions massiques == fractions volumiques
      => on vise V_targets = p_mass * V_tot
    - Sinon: on vise des masses cibles M_targets via rho_moyen * V_tot (approx)
    """
    if all_rhos_equal(mats):
        return [p * V_tot for p in p_mass], 'volume'
    else:
        rho_avg = sum([m['rho'] for m in mats]) / float(len(mats))
        M_tot_est = rho_avg * V_tot
        return [p * M_tot_est for p in p_mass], 'mass'

def vol_list_total(part, elem_list):
    """Somme des volumes d'une liste d'éléments (boucle explicite, pas de generator)."""
    s = 0.0
    for e in elem_list:
        s += element_volume(part, e)
    return s

# ============================================================
# 3) CREATION MODELE + PART 3D CYLINDRE
# ============================================================

if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

if PART_NAME in mdl.parts.keys():
    del mdl.parts[PART_NAME]

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0 * max(R, H))
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))

p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=H)
del mdl.sketches['__profile__']

# ============================================================
# 4) MATERIAUX + SECTIONS (3 minéraux)
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
# 5) MAILLAGE (HEX SWEEP si possible, sinon TET)
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
# 6) CHAMP f(X) = sqrt(2/N) sum cos(Omega·X + phi)
#    Valeur élément = moyenne des valeurs nodales
# ============================================================

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

random.seed(RANDOM_SEED)

sigma = math.sqrt(2.0) / a
waves = []
for i in range(Nwaves):
    ox = random.gauss(0.0, sigma)
    oy = random.gauss(0.0, sigma)
    oz = random.gauss(0.0, sigma)
    phi = 2.0 * math.pi * random.random()
    waves.append((ox, oy, oz, phi))

coef = math.sqrt(2.0 / float(Nwaves))

def f_point(x, y, z):
    s = 0.0
    for (ox, oy, oz, phi) in waves:
        s += math.cos(ox*x + oy*y + oz*z + phi)
    return coef * s

vols = []
fvals = []
V_tot = 0.0

for e in elems:
    v = element_volume(p, e)
    vols.append(v)
    V_tot += v

    fv = 0.0
    nn = float(len(e.connectivity))
    for nl in e.connectivity:
        x, y, z = node_xyz(p, nl)
        fv += f_point(x, y, z)
    fvals.append(fv / nn)

print('==============================')
print('Meshed as     =', meshed_as)
print('Nb elements   =', len(elems))
print('V_tot         =', V_tot)
print('a (mm)        =', a)
print('Nwaves        =', Nwaves)
print('Element size  =', ELEMENT_SIZE)
print('==============================')

# ============================================================
# 7) SEUILLAGE 3 MINERAUX (fractions)
# ============================================================

targets, mode = compute_targets(V_tot, materials, p_mass)
print('Target mode   =', mode)
print('Targets       =', targets)

idx = list(range(len(elems)))
idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

elsets = [[], [], []]

if mode == 'volume':
    V_targets = targets
    V_acc = [0.0, 0.0, 0.0]

    for i in idx:
        if V_acc[0] < V_targets[0]:
            elsets[0].append(elems[i]); V_acc[0] += vols[i]
        elif V_acc[1] < V_targets[1]:
            elsets[1].append(elems[i]); V_acc[1] += vols[i]
        else:
            elsets[2].append(elems[i]); V_acc[2] += vols[i]

    V_used = [vol_list_total(p, elsets[k]) for k in range(3)]
    print('V_used        =', V_used)
    print('V_frac        =', [V_used[k] / V_tot for k in range(3)])

else:
    M_targets = targets
    M_acc = [0.0, 0.0, 0.0]
    rhos = [materials[k]['rho'] for k in range(3)]

    for i in idx:
        if M_acc[0] < M_targets[0]:
            elsets[0].append(elems[i]); M_acc[0] += rhos[0] * vols[i]
        elif M_acc[1] < M_targets[1]:
            elsets[1].append(elems[i]); M_acc[1] += rhos[1] * vols[i]
        else:
            elsets[2].append(elems[i]); M_acc[2] += rhos[2] * vols[i]

    M_tot = M_acc[0] + M_acc[1] + M_acc[2]
    print('M_frac        =', [M_acc[k] / M_tot for k in range(3)])

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
        raise RuntimeError("Mineral-%d vide. Ajuste p_mass ou champ." % (k+1))
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 3 minéraux assignés.')

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
# -*- coding: utf-8 -*-
# Abaqus/CAE - Cylindre 3D hétérogène (2 minéraux)
# Champ gaussien par superposition d'ondes cosinus (Cahn/Lantuéjoul)
# + seuillage par ELEMENTS (valeur élément = moyenne nodale)
#
# Unités conseillées : mm - MPa - tonne
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
ELEMENT_SIZE = 0.5     # mm
PREFER_HEX = True
ELEM_CODE_HEX = C3D8   # intégration complète
ELEM_CODE_TET = C3D4

# Champ aléatoire (ondes cos)
RANDOM_SEED = 0
a = 8                 # mm : taille caractéristique hétérogénéité
Nwaves = 1000         # nombre d'ondes

# 2 minéraux : fractions MASSIQUES cibles (somme=1)
p_mass = (0.30, 0.70)

# Paramètres matériaux (E en MPa, nu, rho en t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
]

# Tri : Mineral-1 correspond aux petites valeurs du champ ?
MIN1_IS_LOW = True

# --- CDP (Concrete Damaged Plasticity) ---
# Paramètres CDP globaux
CDP_DILAT = 35.0
CDP_ECC   = 0.1
CDP_FBFC  = 1.16
CDP_K     = 0.554
CDP_VISC  = 5e-05

# Courbes (formes identiques, niveau via fc0/ft0)
EIN_PIC  = 0.00065
EIN_END  = 0.0094
R_F1     = 153.7 / 196.4
R_CRES   = 4.27  / 196.4

ETIN_END = 0.00035
R_TRES   = 0.034 / 8.78
DT_DAMAGE = 0.98

# Résistances par minéral (MPa) -> ajuste ici
# (tu peux mettre des valeurs différentes pour Mineral-1 / Mineral-2)
strengths = {
    'Mineral-1': dict(fc0=120.0, ft0=5.0),
    'Mineral-2': dict(fc0=285.0, ft0=11.5),
}

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
    - Si rhos identiques: fractions massiques == fractions volumiques
      => V_targets = p_mass * V_tot
    - Sinon: masses cibles (approx)
    """
    if all_rhos_equal(mats):
        return [p * V_tot for p in p_mass], 'volume'
    else:
        rho_avg = sum([m['rho'] for m in mats]) / float(len(mats))
        M_tot_est = rho_avg * V_tot
        return [p * M_tot_est for p in p_mass], 'mass'

def vol_list_total(part, elem_list):
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
# 4) MATERIAUX + SECTIONS (2 minéraux CDP)
# ============================================================

# purge matériaux / sections
for md in materials:
    if md['name'] in mdl.materials.keys():
        del mdl.materials[md['name']]

for i in range(1, 3):
    sname = 'Section-%d' % i
    if sname in mdl.sections.keys():
        del mdl.sections[sname]

# crée matériaux CDP + sections
for i, md in enumerate(materials, start=1):
    name = md['name']
    mat = mdl.Material(name=name)

    # Density + Elastic
    mat.Density(table=((md['rho'],),))
    mat.Elastic(table=((md['E'], md['nu']),))

    # CDP params
    mat.ConcreteDamagedPlasticity(table=((CDP_DILAT, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC),))

    # Strength level per mineral
    fc0 = strengths[name]['fc0']
    ft0 = strengths[name]['ft0']
    f1   = R_F1 * fc0
    cres = R_CRES * fc0
    tres = R_TRES * ft0

    # Compression hardening
    mat.concreteDamagedPlasticity.ConcreteCompressionHardening(
        table=((f1, 0.0),
               (fc0, EIN_PIC),
               (cres, EIN_END))
    )

    # Tension stiffening
    mat.concreteDamagedPlasticity.ConcreteTensionStiffening(
        table=((ft0, 0.0),
               (tres, ETIN_END))
    )

    # Damage curves
    mat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=((0.0, 0.0),))
    mat.concreteDamagedPlasticity.ConcreteTensionDamage(
        table=((0.0, 0.0),
               (DT_DAMAGE, ETIN_END))
    )

    mdl.HomogeneousSolidSection(name='Section-%d' % i, material=name)

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
# 7) SEUILLAGE 2 MINERAUX (fractions)
# ============================================================

targets, mode = compute_targets(V_tot, materials, p_mass)
print('Target mode   =', mode)
print('Targets       =', targets)

idx = list(range(len(elems)))
idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

elsets = [[], []]

if mode == 'volume':
    V_targets = targets
    V_acc = [0.0, 0.0]

    for i in idx:
        if V_acc[0] < V_targets[0]:
            elsets[0].append(elems[i]); V_acc[0] += vols[i]
        else:
            elsets[1].append(elems[i]); V_acc[1] += vols[i]

    V_used = [vol_list_total(p, elsets[k]) for k in range(2)]
    print('V_used        =', V_used)
    print('V_frac        =', [V_used[k] / V_tot for k in range(2)])

else:
    M_targets = targets
    M_acc = [0.0, 0.0]
    rhos = [materials[k]['rho'] for k in range(2)]

    for i in idx:
        if M_acc[0] < M_targets[0]:
            elsets[0].append(elems[i]); M_acc[0] += rhos[0] * vols[i]
        else:
            elsets[1].append(elems[i]); M_acc[1] += rhos[1] * vols[i]

    M_tot = M_acc[0] + M_acc[1]
    print('M_frac        =', [M_acc[k] / M_tot for k in range(2)])

# ============================================================
# 8) ELSETS + ASSIGNATION SECTIONS
# ============================================================

for sname in ['ELSET_MIN1', 'ELSET_MIN2']:
    if sname in p.sets.keys():
        del p.sets[sname]

names = ['ELSET_MIN1', 'ELSET_MIN2']

for k in range(2):
    labels = [e.label for e in elsets[k]]
    if len(labels) == 0:
        raise RuntimeError("Mineral-%d vide. Ajuste p_mass ou champ." % (k+1))
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 2 minéraux assignés.')

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
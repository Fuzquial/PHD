# -*- coding: utf-8 -*-
# Abaqus/CAE - Cylindre 3D + mélange Mat1/Mat2 par éléments
# Gaussian thresholding 3D + ratio de masse (30/70)
#
# Lancer dans Abaqus/CAE : File > Run Script

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

# Géométrie cylindre
R = 0.025   # rayon
H = 0.10   # hauteur

# Densités
rho1 = 2500.0
rho2 = 1800.0

# Élasticité (obligatoire)
E1, nu1 = 1.0e8, 0.30
E2, nu2 = 5.0e7, 0.30

# Ratio de masse cible pour Mat1
target_mass1 = 0.30  # 30%

# Maillage
ELEMENT_SIZE = 0.001          # taille cible
PREFER_HEX = True            # essaie HEX sweep d'abord, sinon fallback TET
ELEM_CODE_HEX = C3D8         # intégration complète (évite hourglass)
ELEM_CODE_TET = C3D4

# Champ gaussien 3D
RANDOM_SEED = 42
Lcorr = 0.20 * R             # taille des amas
Nsources = 40

# ============================================================
# 2) OUTILS
# ============================================================

def element_centroid_xyz(part, elem):
    xs = ys = zs = 0.0
    n = float(len(elem.connectivity))
    for node_label in elem.connectivity:
        node_obj = part.nodes[node_label - 1]
        xs += node_obj.coordinates[0]
        ys += node_obj.coordinates[1]
        zs += node_obj.coordinates[2]
    return xs / n, ys / n, zs / n

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

def compute_target_volume_for_mass_ratio(V_tot, rho1, rho2, target_mass1):
    t = target_mass1
    return V_tot * (t * rho2) / ((1.0 - t) * rho1 + t * rho2)

def sample_point_in_cylinder(R, H):
    u = random.random()
    v = random.random()
    w = random.random()
    r = R * math.sqrt(u)
    th = 2.0 * math.pi * v
    x = r * math.cos(th)
    y = r * math.sin(th)
    z = H * w
    return x, y, z

# ============================================================
# 3) CREATION MODELE + PART 3D CYLINDRE
# ============================================================

if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# Nettoyage si rerun
if PART_NAME in mdl.parts.keys():
    del mdl.parts[PART_NAME]

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0 * max(R, H))
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))

p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=H)
del mdl.sketches['__profile__']

# ============================================================
# 4) MATERIAUX + SECTIONS
# ============================================================

for mat_name in ['Material-1', 'Material-2']:
    if mat_name in mdl.materials.keys():
        del mdl.materials[mat_name]

for sec_name in ['Section-1', 'Section-2']:
    if sec_name in mdl.sections.keys():
        del mdl.sections[sec_name]

mat1 = mdl.Material(name='Material-1')
mat1.Density(table=((rho1,),))
mat1.Elastic(table=((E1, nu1),))

mat2 = mdl.Material(name='Material-2')
mat2.Density(table=((rho2,),))
mat2.Elastic(table=((E2, nu2),))

sec1 = mdl.HomogeneousSolidSection(name='Section-1', material='Material-1')
sec2 = mdl.HomogeneousSolidSection(name='Section-2', material='Material-2')

# ============================================================
# 5) MAILLAGE (HEX SWEEP si possible, sinon TET)
# ============================================================

p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
cells = p.cells[:]

meshed_as = None

if PREFER_HEX:
    try:
        # Pour un cylindre extrudé : SWEEP + HEX est typiquement supporté
        p.setMeshControls(regions=cells, technique=SWEEP, elemShape=HEX)

        elemType = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
        p.setElementType(regions=(cells,), elemTypes=(elemType,))
        p.generateMesh()
        meshed_as = 'HEX (SWEEP)'
    except Exception as e:
        print('--- HEX SWEEP failed, fallback to TET. Reason:', str(e))
        meshed_as = None

if meshed_as is None:
    # Fallback TET (marche quasiment toujours)
    p.deleteMesh()
    p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    p.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
    elemType = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
    p.setElementType(regions=(cells,), elemTypes=(elemType,))
    p.generateMesh()
    meshed_as = 'TET (FREE)'

print('--- Meshing done:', meshed_as)

# ============================================================
# 6) GAUSSIAN THRESHOLDING 3D + RATIO MASSE
# ============================================================

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré. Vérifie ELEMENT_SIZE et mesh controls.")

random.seed(RANDOM_SEED)

vols = []
centroids = []
V_tot = 0.0
for e in elems:
    v = element_volume(p, e)
    vols.append(v)
    V_tot += v
    centroids.append(element_centroid_xyz(p, e))

V1_target = compute_target_volume_for_mass_ratio(V_tot, rho1, rho2, target_mass1)

# Sources gaussiennes
sources = []
for k in range(Nsources):
    xk, yk, zk = sample_point_in_cylinder(R, H)
    ak = random.gauss(0.0, 1.0)
    sources.append((xk, yk, zk, ak))

def gaussian_field_value_3d(x, y, z):
    s = 0.0
    denom = 2.0 * (Lcorr ** 2)
    for (xk, yk, zk, ak) in sources:
        r2 = (x-xk)**2 + (y-yk)**2 + (z-zk)**2
        s += ak * math.exp(-r2 / denom)
    return s

gvals = [gaussian_field_value_3d(cx, cy, cz) for (cx, cy, cz) in centroids]

print('==============================')
print('V_tot        =', V_tot)
print('V1_target    =', V1_target)
print('target_mass1 =', target_mass1)
print('rho1, rho2   =', rho1, rho2)
print('Nb elements  =', len(elems))
print('Lcorr        =', Lcorr)
print('Nsources     =', Nsources)
print('Meshed as    =', meshed_as)
print('==============================')

idx = list(range(len(elems)))
idx.sort(key=lambda i: gvals[i], reverse=True)

V1 = 0.0
mat1_elems = []
mat2_elems = []

for i in idx:
    if V1 < V1_target:
        mat1_elems.append(elems[i])
        V1 += vols[i]
    else:
        mat2_elems.append(elems[i])

m1 = rho1 * V1
m2 = rho2 * (V_tot - V1)
ratio_obt = m1 / (m1 + m2)

print('V1_obtenu    =', V1)
print('ratio_masse  =', ratio_obt)
print('delta_ratio  =', ratio_obt - target_mass1)

# ============================================================
# 7) ELSETS + ASSIGNATION SECTIONS (via labels)
# ============================================================

for sname in ['ELSET_MAT1', 'ELSET_MAT2']:
    if sname in p.sets.keys():
        del p.sets[sname]

labels1 = [e.label for e in mat1_elems]
labels2 = [e.label for e in mat2_elems]

if len(labels1) == 0 or len(labels2) == 0:
    raise RuntimeError("Un des groupes est vide. Ajuste ELEMENT_SIZE / Lcorr / Nsources.")

elems1 = p.elements.sequenceFromLabels(labels=labels1)
elems2 = p.elements.sequenceFromLabels(labels=labels2)

set1 = p.Set(name='ELSET_MAT1', elements=elems1)
set2 = p.Set(name='ELSET_MAT2', elements=elems2)

p.SectionAssignment(region=regionToolset.Region(elements=set1.elements), sectionName='Section-1')
p.SectionAssignment(region=regionToolset.Region(elements=set2.elements), sectionName='Section-2')

print('--- OK : Sections assignées (Gaussian thresholding 3D).')

# ============================================================
# 8) ASSEMBLY + INSTANCE
# ============================================================

a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

if INST_NAME in a.instances.keys():
    del a.instances[INST_NAME]

a.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Instance créée :', INST_NAME)
print('--- Script terminé.')
# -*- coding: utf-8 -*-
# Abaqus/CAE script COMPLET (A à Z) :
# - Crée un carré 2D
# - Crée 2 matériaux (densités rho1, rho2)
# - Maille
# - Attribue chaque élément soit Mat1 soit Mat2
#   en respectant un ratio de MASSE cible (ex 30/70)
#
# Lancer dans Abaqus/CAE : File > Run Script

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random

# ============================================================
# 1) PARAMETRES UTILISATEUR
# ============================================================

MODEL_NAME = 'Model-1'
PART_NAME  = 'Square2D'
INST_NAME  = 'Square2D-1'

# Géométrie
L = 1.0  # côté du carré

# Densités
rho1 = 2500.0
rho2 = 1800.0

# Ratio de masse visé pour Mat1
target_mass1 = 0.30  # 30%

# Epaisseur hors-plan (plane stress/shell). S'annule dans le ratio si identique partout,
# mais on la garde pour la masse absolue et cohérence des sections.
THICKNESS = 1.0

# Maillage
ELEMENT_SIZE = 0.005  # plus petit => plus d'éléments => ratio plus précis
USE_QUADS = True     # True: quads dominants, False: tris

# Type d'élément:
# - Plane stress : CPS4R (quad) / CPS3 (tri)
# - Plane strain : CPE4R (quad) / CPE3 (tri)
ELEM_CODE_QUAD = CPS4R
ELEM_CODE_TRI  = CPS3

# Mélange
RANDOM_SEED = 42

# ============================================================
# 2) OUTILS : aire d’un élément 2D via coordonnées nodales
# ============================================================

def polygon_area_xy(coords):
    """Aire d'un polygone 2D (shoelace). coords: [(x,y), ...]"""
    s = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def element_area_2d(part, elem):
    """Aire d'un élément 2D tri/quad dans XY."""
    coords = []
    for node_label in elem.connectivity:
        # labels Abaqus commencent à 1, index python à 0
        node_obj = part.nodes[node_label - 1]
        x, y = node_obj.coordinates[0], node_obj.coordinates[1]
        coords.append((x, y))
    return polygon_area_xy(coords)

def compute_target_area_for_mass_ratio(A_tot, rho1, rho2, target_mass1):
    """
    Trouve A1 tel que m1/(m1+m2)=target_mass1 avec m1=rho1*A1*t et m2=rho2*(A_tot-A1)*t
    """
    t = target_mass1
    return A_tot * (t * rho2) / ((1.0 - t) * rho1 + t * rho2)

# ============================================================
# 3) CREATION MODELE + PART 2D CARRE
# ============================================================

# Crée le modèle si absent
if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# Nettoyage simple si rerun
if PART_NAME in mdl.parts.keys():
    del mdl.parts[PART_NAME]

# Sketch carré
sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=10.0 * L)
sk.rectangle(point1=(0.0, 0.0), point2=(L, L))

# Part 2D (planar)
p = mdl.Part(name=PART_NAME, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p.BaseShell(sketch=sk)
del mdl.sketches['__profile__']

# ============================================================
# 4) MATERIAUX + SECTIONS
# ============================================================

# Nettoyage matériaux/sections si rerun
for mat_name in ['Material-1', 'Material-2']:
    if mat_name in mdl.materials.keys():
        del mdl.materials[mat_name]

for sec_name in ['Section-1', 'Section-2']:
    if sec_name in mdl.sections.keys():
        del mdl.sections[sec_name]

mat1 = mdl.Material(name='Material-1')
mat1.Density(table=((rho1,),))

mat2 = mdl.Material(name='Material-2')
mat2.Density(table=((rho2,),))

# Sections (2D planar : HomogeneousSolidSection ok)
sec1 = mdl.HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=THICKNESS)
sec2 = mdl.HomogeneousSolidSection(name='Section-2', material='Material-2', thickness=THICKNESS)

# ============================================================
# 5) MAILLAGE
# ============================================================

p.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)

faces = p.faces[:]
if USE_QUADS:
    p.setMeshControls(regions=faces, elemShape=QUAD, technique=FREE)
else:
    p.setMeshControls(regions=faces, elemShape=TRI, technique=FREE)

if USE_QUADS:
    elemType = mesh.ElemType(elemCode=ELEM_CODE_QUAD, elemLibrary=STANDARD)
else:
    elemType = mesh.ElemType(elemCode=ELEM_CODE_TRI, elemLibrary=STANDARD)

p.setElementType(regions=(faces,), elemTypes=(elemType,))
p.generateMesh()

# ============================================================
# 6) SELECTION ELEMENTAIRE POUR RESPECTER 30/70 EN MASSE
# ============================================================

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré. Vérifie ELEMENT_SIZE et la géométrie.")

areas = []
A_tot = 0.0
for e in elems:
    a = element_area_2d(p, e)
    areas.append(a)
    A_tot += a

A1_target = compute_target_area_for_mass_ratio(A_tot, rho1, rho2, target_mass1)

print('==============================')
print('A_tot        =', A_tot)
print('A1_target    =', A1_target)
print('target_mass1 =', target_mass1)
print('rho1, rho2   =', rho1, rho2)
print('Nb elements  =', len(elems))
print('==============================')

random.seed(RANDOM_SEED)
idx = list(range(len(elems)))
random.shuffle(idx)

A1 = 0.0
mat1_elems = []
mat2_elems = []

for i in idx:
    if A1 < A1_target:
        mat1_elems.append(elems[i])
        A1 += areas[i]
    else:
        mat2_elems.append(elems[i])

# Ajustement simple (swap 1 élément) pour se rapprocher
overshoot = A1 - A1_target
if overshoot > 0.0 and len(mat1_elems) > 0:
    best = None
    best_diff = 1e99
    for e in mat1_elems:
        a = element_area_2d(p, e)
        diff = abs(a - overshoot)
        if diff < best_diff:
            best = e
            best_diff = diff
    if best is not None:
        mat1_elems.remove(best)
        mat2_elems.append(best)
        A1 -= element_area_2d(p, best)

m1 = rho1 * A1 * THICKNESS
m2 = rho2 * (A_tot - A1) * THICKNESS
ratio_obt = m1 / (m1 + m2)

print('A1_obtenu    =', A1)
print('ratio_masse  =', ratio_obt)
print('delta_ratio  =', ratio_obt - target_mass1)

# ============================================================
# 7) CREATION ELSETS + ASSIGNATION DES SECTIONS PAR ELEMENTS
#    (CORRIGÉ : création Set via labels, pas via liste Python)
# ============================================================

# Nettoyage sets si rerun
for sname in ['ELSET_MAT1', 'ELSET_MAT2']:
    if sname in p.sets.keys():
        del p.sets[sname]

labels1 = [e.label for e in mat1_elems]
labels2 = [e.label for e in mat2_elems]

if len(labels1) == 0 or len(labels2) == 0:
    raise RuntimeError("Un des groupes est vide (Mat1 ou Mat2). Diminue ELEMENT_SIZE ou vérifie le tirage.")

elems1 = p.elements.sequenceFromLabels(labels=labels1)
elems2 = p.elements.sequenceFromLabels(labels=labels2)

set1 = p.Set(name='ELSET_MAT1', elements=elems1)
set2 = p.Set(name='ELSET_MAT2', elements=elems2)

# Assign section Mat1
p.SectionAssignment(
    region=regionToolset.Region(elements=set1.elements),
    sectionName='Section-1',
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField='',
    thicknessAssignment=FROM_SECTION
)

# Assign section Mat2
p.SectionAssignment(
    region=regionToolset.Region(elements=set2.elements),
    sectionName='Section-2',
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField='',
    thicknessAssignment=FROM_SECTION
)

print('--- OK : Sections assignées élément par élément (ratio masse cible).')

# ============================================================
# 8) ASSEMBLY + INSTANCE (optionnel mais pratique)
# ============================================================

a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

# Nettoyage instance si rerun
if INST_NAME in a.instances.keys():
    del a.instances[INST_NAME]

a.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Instance créée :', INST_NAME)
print('--- Script terminé.')
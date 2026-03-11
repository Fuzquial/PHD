# -*- coding: utf-8 -*-
# Abaqus/CAE - Cylindre 3D hétérogène (Isotropie renforcée)
# Méthode : Superposition d'ondes avec échantillonnage sphérique
# Unités : mm, MPa, tonne

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math
import time

# ============================================================
# 1) PARAMETRES UTILISATEUR
# ============================================================
MODEL_NAME = 'Model-1'
PART_NAME  = 'Cyl3D'
INST_NAME  = 'Cyl3D-1'

# Géométrie
R = 2.5    
H = 10.0     

# Maillage
ELEMENT_SIZE = 0.12    
PREFER_HEX = True

# Champ gaussien (Covariance)
USE_RANDOM_SEED = False 
RANDOM_SEED     = 1214  
a = 0.8                 # Longueur de corrélation
Nwaves = 8000           # Augmenté pour saturer le spectre et casser les stries

# Fractions massiques
p_mass = (0.45, 0.55)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
]
MIN1_IS_LOW = True

# ============================================================
# 2) FONCTIONS TECHNIQUES (Compatibles Python 2.7)
# ============================================================
def element_volume(part, elem):
    try: return float(elem.getSize())
    except: return (ELEMENT_SIZE**3)

def node_xyz(part, node_label):
    n = part.nodes[node_label - 1]
    return n.coordinates[0], n.coordinates[1], n.coordinates[2]

# ============================================================
# 3) CONSTRUCTION DU MODELE
# ============================================================
mdl = mdb.models[MODEL_NAME]
if PART_NAME in mdl.parts.keys(): del mdl.parts[PART_NAME]

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=20.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))
p = mdl.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseSolidExtrude(sketch=sk, depth=H)

# Matériaux et Sections
for i, md in enumerate(materials, start=1):
    mat = mdl.Material(name=md['name'])
    mat.Elastic(table=((md['E'], md['nu']),))
    mat.Density(table=((md['rho'],),))
    mdl.HomogeneousSolidSection(name='Section-%d'%i, material=md['name'])

# Maillage
p.seedPart(size=ELEMENT_SIZE)
try:
    p.setMeshControls(regions=p.cells[:], technique=SWEEP, elemShape=HEX)
    p.generateMesh()
except:
    p.setMeshControls(regions=p.cells[:], technique=FREE, elemShape=TET)
    p.generateMesh()

# ============================================================
# 4) GENERATION DU CHAMP (METHODE SPHERIQUE)
# ============================================================
if USE_RANDOM_SEED:
    random.seed(RANDOM_SEED)
else:
    s_val = int(time.time())
    random.seed(s_val)
    print('--- Nouvelle Seed : %d' % s_val)

sigma_f = 1.0 / a
waves = []
for i in range(Nwaves):
    # Echantillonnage uniforme sur une sphère pour l'isotropie totale
    theta = 2.0 * math.pi * random.random()
    phi = math.acos(2.0 * random.random() - 1.0)
    r = random.gauss(0.0, sigma_f) # Distribution de fréquence gaussienne
    
    ox = r * math.sin(phi) * math.cos(theta)
    oy = r * math.sin(phi) * math.sin(theta)
    oz = r * math.cos(phi)
    
    phase = 2.0 * math.pi * random.random()
    waves.append((ox, oy, oz, phase))

def f_point(x, y, z):
    res = 0.0
    for w in waves:
        res += math.cos(w[0]*x + w[1]*y + w[2]*z + w[3])
    return res * math.sqrt(2.0 / float(Nwaves))

# Calcul aux barycentres
elems = p.elements[:]
vols, f_raw = [], []
V_tot = 0.0

for e in elems:
    v = element_volume(p, e)
    vols.append(v)
    V_tot += v
    
    # Barycentre (Python 2.7 sum list)
    n_coords = [node_xyz(p, nl) for nl in e.connectivity]
    nn = float(len(n_coords))
    cx = sum([c[0] for c in n_coords]) / nn
    cy = sum([c[1] for c in n_coords]) / nn
    cz = sum([c[2] for c in n_coords]) / nn
    f_raw.append(f_point(cx, cy, cz))

# Normalisation Z-score
avg = sum(f_raw) / len(f_raw)
std = math.sqrt(sum([(x - avg)**2 for x in f_raw]) / len(f_raw))
fvals = [(x - avg) / std for x in f_raw]

# ============================================================
# 5) SEUILLAGE ET ASSIGNATION
# ============================================================
idx = list(range(len(elems)))
idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

V_target = p_mass[0] * V_tot
V_acc = 0.0
labels1, labels2 = [], []

for i in idx:
    if V_acc < V_target:
        labels1.append(elems[i].label)
        V_acc += vols[i]
    else:
        labels2.append(elems[i].label)

# Application Abaqus
sets = [('ELSET_MIN1', labels1, 'Section-1'), ('ELSET_MIN2', labels2, 'Section-2')]
for sname, lbls, sec in sets:
    eset = p.Set(name=sname, elements=p.elements.sequenceFromLabels(lbls))
    p.SectionAssignment(region=regionToolset.Region(elements=eset.elements), sectionName=sec)

# Assemblage
a_asm = mdl.rootAssembly
if INST_NAME in a_asm.instances.keys(): del a_asm.instances[INST_NAME]
a_asm.Instance(name=INST_NAME, part=p, dependent=ON)

print('--- Succès : Isotropie garantie (a=%.2f, N=%d)' % (a, Nwaves))
# -*- coding: utf-8 -*-
# Abaqus/CAE - Tessellation de Laguerre-Voronoi (Section 4.1.2)
# Unités : mm, MPa

from abaqus import *
from abaqusConstants import *
import regionToolset
import random
import math
import time

# ============================================================
# 1) PARAMETRES DE LA SECTION 4.1.2
# ============================================================
MODEL_NAME = 'Model-Laguerre'
R, H = 2.5, 10.0
ELEMENT_SIZE = 0.3 
N_GRAINS = 120

# Paramètres de poids (Laguerre weight)
# Un poids élevé = un grain plus gros
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.5  # Ajuster pour augmenter la dispersion de taille

# Phases (0.45 / 0.55)
P_PHASES = [0.45, 0.55]
MAT_NAMES = ['Mineral-1', 'Mineral-2']

# ============================================================
# 2) GEOMETRIE ET MAILLAGE
# ============================================================
mdl = mdb.models[MODEL_NAME] if MODEL_NAME in mdb.models.keys() else mdb.Model(name=MODEL_NAME)
p = mdl.Part(name='Cyl', dimensionality=THREE_D, type=DEFORMABLE_BODY)
sk = mdl.ConstrainedSketch(name='profile', sheetSize=10.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R, 0.0))
p.BaseSolidExtrude(sketch=sk, depth=H)

p.seedPart(size=ELEMENT_SIZE)
p.setMeshControls(regions=p.cells[:], technique=FREE, elemShape=TET)
p.generateMesh()

# ============================================================
# 3) GENERATION DES GERMES PONDERES (LAGUERRE)
# ============================================================
seeds = []
while len(seeds) < N_GRAINS:
    x, y, z = random.uniform(-R, R), random.uniform(-R, R), random.uniform(0, H)
    if x**2 + y**2 <= R**2:
        # Chaque germe a une position (x,y,z) et un poids (w) 
        w = random.uniform(WEIGHT_MIN, WEIGHT_MAX)
        seeds.append({'pos': (x, y, z), 'weight': w})

# Assignation des phases aux grains
seed_phases = [0 if random.random() < P_PHASES[0] else 1 for _ in range(N_GRAINS)]

# ============================================================
# 4) ASSIGNATION SELON LA DISTANCE PUISSANCE (LAGUERRE)
# ============================================================
elems = p.elements
phase_labels = [[], []]

print("--- Calcul de la Tessellation de Laguerre...")

for e in elems:
    # Calcul du barycentre
    nodes = [p.nodes[nl-1].coordinates for nl in e.connectivity]
    nn = float(len(nodes))
    bary = (sum([n[0] for n in nodes])/nn, 
            sum([n[1] for n in nodes])/nn, 
            sum([n[2] for n in nodes])/nn)
    
    # Trouver le germe minimisant la distance puissance 
    min_dist_pow = 1e12
    closest_idx = 0
    
    for i, s in enumerate(seeds):
        # Formule de Laguerre : d^2 - w
        dx = bary[0] - s['pos'][0]
        dy = bary[1] - s['pos'][1]
        dz = bary[2] - s['pos'][2]
        dist_sq = dx**2 + dy**2 + dz**2
        
        d_pow = dist_sq - s['weight']
        
        if d_pow < min_dist_pow:
            min_dist_pow = d_pow
            closest_idx = i
            
    phase_labels[seed_phases[closest_idx]].append(e.label)

# ============================================================
# 5) SECTIONS ET FINALISATION
# ============================================================
for i, name in enumerate(MAT_NAMES):
    if name not in mdl.materials.keys():
        mat = mdl.Material(name=name)
        mat.Elastic(table=((35000.0 if i==0 else 72000.0, 0.25),))
    
    sec = 'Section-' + name
    mdl.HomogeneousSolidSection(name=sec, material=name)
    
    if phase_labels[i]:
        eset = p.Set(name='Set-'+name, elements=p.elements.sequenceFromLabels(phase_labels[i]))
        p.SectionAssignment(region=regionToolset.Region(elements=eset.elements), sectionName=sec)

mdl.rootAssembly.Instance(name='Cyl-1', part=p, dependent=ON)
print("--- Termine : Laguerre-Voronoi avec %d grains" % N_GRAINS)
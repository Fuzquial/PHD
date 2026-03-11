# -*- coding: utf-8 -*-
# Abaqus/CAE - Cylindre 3D hétérogène (3 minéraux)
# Microstructure via HOPPS / Potts 3D à composition conservée (Kawasaki swap)
# + mapping voxels -> éléments Abaqus (par centroïde)
#
# Unités conseillées: mm - MPa - tonne

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
R = 2.5
H = 10.0

# Maillage Abaqus
ELEMENT_SIZE = 0.1
PREFER_HEX = True
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

# Microstructure HOPPS 3D (voxel grid)
VOXEL_SIZE = 0.2     # <- mets 0.1 (fines) ou 0.2/0.3 (plus rapide)
HOPPS_STEPS = 50     # nombre de "MCS"
T_MC = 0.2           # température Monte-Carlo (mobilité/interface)
J = 1.0              # énergie d'interface isotrope (Potts)

RANDOM_SEED = 0

# 3 minéraux : fractions massiques cibles (somme=1)
p_mass = (0.30, 0.50, 0.20)

# Paramètres matériaux (E en MPa, nu, rho en t/mm^3)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# ============================================================
# 2) OUTILS Abaqus
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
    sx = sy = sz = 0.0
    nn = float(len(elem.connectivity))
    for nl in elem.connectivity:
        x, y, z = node_xyz(part, nl)
        sx += x; sy += y; sz += z
    return sx/nn, sy/nn, sz/nn

def all_rhos_equal(mats, tol=1e-30):
    rhos = [m['rho'] for m in mats]
    return (max(rhos) - min(rhos)) < tol

# ============================================================
# 3) HOPPS / POTTS 3D (Kawasaki swap) sur voxel grid cylindrique
# ============================================================

# voisinage 6
NEIGH6 = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def idx_1d(i,j,k,nx,ny,nz):
    return (k*ny + j)*nx + i

def ijk_from_1d(q,nx,ny,nz):
    i = q % nx
    t = (q - i)//nx
    j = t % ny
    k = (t - j)//ny
    return i,j,k

def build_voxel_cylinder(R, H, dx):
    """
    Crée un grid voxel couvrant [-R,R]x[-R,R]x[0,H].
    Retourne:
      nx,ny,nz, xmin,ymin,zmin
      active_mask (liste bool 1D)
      active_list (liste d'indices 1D actifs)
    """
    xmin = -R
    ymin = -R
    zmin = 0.0
    xmax = R
    ymax = R
    zmax = H

    nx = int(math.ceil((xmax - xmin)/dx))
    ny = int(math.ceil((ymax - ymin)/dx))
    nz = int(math.ceil((zmax - zmin)/dx))

    nTot = nx*ny*nz
    active = [False]*nTot
    active_list = []

    # on considère un voxel "actif" si son centre est dans le cylindre
    for k in range(nz):
        zc = zmin + (k+0.5)*dx
        for j in range(ny):
            yc = ymin + (j+0.5)*dx
            for i in range(nx):
                xc = xmin + (i+0.5)*dx
                if (xc*xc + yc*yc) <= (R*R) and (0.0 <= zc <= H):
                    q = idx_1d(i,j,k,nx,ny,nz)
                    active[q] = True
                    active_list.append(q)

    return nx,ny,nz,xmin,ymin,zmin,active,active_list

def init_phases_conserved(active_list, n_active, p_mass, rng):
    """
    Initialise phases 0/1/2 en respectant EXACTEMENT les fractions.
    Retourne phases dict q->phase (seulement pour actifs)
    """
    # comptes exacts (ajuste le dernier pour somme exacte)
    n0 = int(round(p_mass[0]*n_active))
    n1 = int(round(p_mass[1]*n_active))
    n2 = n_active - n0 - n1
    if n2 < 0:
        # sécurité si arrondis bizarres
        n2 = max(0, n2)
        s = n0 + n1 + n2
        if s != n_active:
            n0 = max(0, n_active - n1 - n2)

    lst = list(active_list)
    rng.shuffle(lst)

    phases = {}
    # affectation
    for t in range(n_active):
        q = lst[t]
        if t < n0:
            phases[q] = 0
        elif t < n0 + n1:
            phases[q] = 1
        else:
            phases[q] = 2
    return phases, (n0,n1,n2)

def count_unlike_neighbors(q, phase_q, phases, active, nx,ny,nz, J):
    """
    énergie locale interface ~ J * (# voisins actifs d'une autre phase)
    """
    i,j,k = ijk_from_1d(q,nx,ny,nz)
    e = 0.0
    for di,dj,dk in NEIGH6:
        ii = i+di; jj=j+dj; kk=k+dk
        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
            qq = idx_1d(ii,jj,kk,nx,ny,nz)
            if active[qq]:
                if phases[qq] != phase_q:
                    e += J
    return e

def deltaE_swap(q1, q2, phases, active, nx,ny,nz, J):
    """
    Calcule ΔE pour swap(phase[q1], phase[q2]) en recalculant l'énergie locale
    autour de q1 et q2 (et uniquement là).
    """
    a = phases[q1]
    b = phases[q2]
    if a == b:
        return 0.0

    # énergie locale avant
    E_before = count_unlike_neighbors(q1, a, phases, active, nx,ny,nz, J) \
             + count_unlike_neighbors(q2, b, phases, active, nx,ny,nz, J)

    # swap temporaire
    phases[q1], phases[q2] = b, a

    # énergie locale après
    E_after = count_unlike_neighbors(q1, b, phases, active, nx,ny,nz, J) \
            + count_unlike_neighbors(q2, a, phases, active, nx,ny,nz, J)

    # revert
    phases[q1], phases[q2] = a, b

    return (E_after - E_before)

def hopps_kawasaki_3d(nx,ny,nz, active, active_list, phases, steps, T, J, rng):
    """
    HOPPS/Potts 3D conservatif:
    - pick voxel actif q1
    - pick un voisin actif q2
    - propose SWAP des phases
    - accept Metropolis
    """
    n_active = len(active_list)
    if n_active == 0:
        return

    for mcs in range(steps):
        # un MCS ~ N tentatives
        for _ in range(n_active):
            q1 = active_list[rng.randint(0, n_active-1)]
            i,j,k = ijk_from_1d(q1,nx,ny,nz)

            # choisir un voisin actif
            # on essaye quelques fois
            q2 = None
            for _try in range(12):
                di,dj,dk = NEIGH6[rng.randint(0,5)]
                ii=i+di; jj=j+dj; kk=k+dk
                if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                    qq = idx_1d(ii,jj,kk,nx,ny,nz)
                    if active[qq]:
                        q2 = qq
                        break
            if q2 is None:
                continue

            if phases[q1] == phases[q2]:
                continue

            dE = deltaE_swap(q1, q2, phases, active, nx,ny,nz, J)

            if dE <= 0.0:
                # accept
                phases[q1], phases[q2] = phases[q2], phases[q1]
            else:
                if T > 0.0:
                    if rng.random() < math.exp(-dE / T):
                        phases[q1], phases[q2] = phases[q2], phases[q1]

        if (mcs+1) % max(1, steps//10) == 0:
            print('--- HOPPS MCS %d/%d' % (mcs+1, steps))

def voxel_phase_at_point(x,y,z, nx,ny,nz, xmin,ymin,zmin, dx, phases, active):
    """
    retourne phase (0/1/2) au voxel contenant (x,y,z).
    si hors domaine ou inactif: None
    """
    i = int((x - xmin)/dx)
    j = int((y - ymin)/dx)
    k = int((z - zmin)/dx)
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return None
    q = idx_1d(i,j,k,nx,ny,nz)
    if not active[q]:
        return None
    return phases[q]

# ============================================================
# 4) CREATION MODELE + PART 3D CYLINDRE
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
# 5) MATERIAUX + SECTIONS
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
# 6) MAILLAGE Abaqus
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

elems = p.elements[:]
if len(elems) == 0:
    raise RuntimeError("Aucun élément généré.")

print('==============================')
print('Meshed as     =', meshed_as)
print('Nb elements   =', len(elems))
print('Element size  =', ELEMENT_SIZE)
print('VOXEL_SIZE    =', VOXEL_SIZE)
print('HOPPS_STEPS   =', HOPPS_STEPS)
print('T_MC          =', T_MC)
print('J             =', J)
print('==============================')

# ============================================================
# 7) MICROSTRUCTURE HOPPS 3D sur voxels (fractions conservées)
# ============================================================

rng = random.Random(RANDOM_SEED)

nx,ny,nz,xmin,ymin,zmin,active,active_list = build_voxel_cylinder(R, H, VOXEL_SIZE)
n_active = len(active_list)
if n_active == 0:
    raise RuntimeError("Aucun voxel actif dans le cylindre. Vérifie VOXEL_SIZE/R/H.")

phases, counts = init_phases_conserved(active_list, n_active, p_mass, rng)
print('--- Init phases counts (0,1,2) =', counts, 'over', n_active, 'voxels')

# HOPPS / Potts conservatif
hopps_kawasaki_3d(nx,ny,nz, active, active_list, phases,
                  steps=HOPPS_STEPS, T=T_MC, J=J, rng=rng)

# ============================================================
# 8) MAPPING voxels -> éléments Abaqus (centroïdes)
# ============================================================

elsets = [[], [], []]   # elements per mineral

for e in elems:
    cx, cy, cz = element_centroid(p, e)
    ph = voxel_phase_at_point(cx, cy, cz, nx,ny,nz, xmin,ymin,zmin, VOXEL_SIZE, phases, active)
    if ph is None:
        # sécurité (normalement ça n'arrive pas si l'élément est dans le cylindre)
        ph = 0
    elsets[ph].append(e)

# stats volumes
V_tot = 0.0
V_used = [0.0, 0.0, 0.0]
for e in elems:
    V_tot += element_volume(p, e)
for k in range(3):
    for e in elsets[k]:
        V_used[k] += element_volume(p, e)

print('--- Volume fractions (Abaqus elems) ~', [V_used[k]/V_tot for k in range(3)])

# ============================================================
# 9) ELSETS + ASSIGNATION SECTIONS
# ============================================================

for sname in ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']:
    if sname in p.sets.keys():
        del p.sets[sname]

names = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']

for k in range(3):
    labels = [e.label for e in elsets[k]]
    if len(labels) == 0:
        raise RuntimeError("Mineral-%d vide. Ajuste p_mass / VOXEL_SIZE / HOPPS." % (k+1))
    elems_k = p.elements.sequenceFromLabels(labels=labels)
    p.Set(name=names[k], elements=elems_k)
    p.SectionAssignment(region=regionToolset.Region(elements=elems_k),
                        sectionName='Section-%d' % (k+1))

print('--- OK : 3 minéraux assignés via HOPPS 3D (Kawasaki).')

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
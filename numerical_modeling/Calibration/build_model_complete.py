# -*- coding: utf-8 -*-
# Abaqus/CAE Python script - VERSION COMPLETE
# Pair-based contact: frictionless + hard, small sliding
# Triaxial: confinement step + axial compression step
# Minerals are TRUE CDP materials via Abaqus API
# Boucle sur pressions de confinement : 5, 10, 30, 50 MPa
# BCs supplementaires : blocage des rotations sur un noeud lateral
#                       et un noeud de la face superieure du cylindre
# VERSION : creation des modeles uniquement, sans lancer les jobs

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# User parameters
# ============================================================
BASE_MODEL_NAME = 'Model-Base'

CYL_R    = 2.5
CYL_H    = 10.0
PLATEN_R = 3.0
PLATEN_T = 2.0

ELEMENT_SIZE_CYL    = 0.5
ELEMENT_SIZE_PLATEN = 0.6
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

# Pressions de confinement a simuler (MPa)
PRESSURE_LIST = [5, 10, 30, 50]

TOP_DISP   = -0.5
STEP2_TIME = 1200.0

# Heterogeneity field
RANDOM_SEED = 42
a      = 10
Nwaves = 300
p_frac = (0.30, 0.50, 0.20)

minerals = [
    dict(name='Feldspath', E=48000.0, nu=0.25, rho=2.1e-9, cdp=dict(sigc0=180.0, sigt0=8.0,  dilat=30.0, dt=0.98)),
    dict(name='Quartz',    E=85000.0, nu=0.25, rho=2.2e-9, cdp=dict(sigc0=350.0, sigt0=10.0, dilat=35.0, dt=0.98)),
    dict(name='Biotite',   E=30000.0, nu=0.25, rho=2.1e-9, cdp=dict(sigc0=250.0, sigt0=5.0,  dilat=30.0, dt=0.98)),
]

PLATEN_MAT = dict(name='Material-2', E=1.0e10, nu=0.25, rho=6.9e-9)

# ----------------- CDP global params -----------------
CDP_DILAT = 35.0
CDP_ECC   = 0.1
CDP_FBFC  = 1.16
CDP_K     = 0.554
CDP_VISC  = 5e-4

# Strain abscissas
EIN_PIC   = 0.00065
EIN_END   = 0.0094
ETIN_END  = 0.00035
DT_DAMAGE = 0.98

R_F1   = 153.7 / 196.4
R_CRES = 4.27  / 196.4
R_TRES = 0.034 / 8.78

# Face selection tolerances
Z_TOL   = 1e-3
XY_TOL  = 1e-2
RAD_TOL = 1e-3

# ============================================================
# Helpers generaux
# ============================================================
def safe_delete(container_obj, key):
    try:
        if key in container_obj.keys():
            del container_obj[key]
    except Exception:
        pass

def build_gaussian_waves(nwaves, seed):
    random.seed(seed)
    waves = []
    for _ in range(nwaves):
        kx = random.gauss(0.0, 1.0)
        ky = random.gauss(0.0, 1.0)
        kz = random.gauss(0.0, 1.0)
        norm = math.sqrt(kx*kx + ky*ky + kz*kz) + 1e-12
        kx /= norm
        ky /= norm
        kz /= norm
        phi = random.random() * 2.0 * math.pi
        waves.append((1.0, a*kx, a*ky, a*kz, phi))
    return waves

def field_value_at_point(x, y, z, waves):
    s = 0.0
    for (amp, kx, ky, kz, phi) in waves:
        s += amp * math.cos(kx*x + ky*y + kz*z + phi)
    return s / float(len(waves))

def element_centroid(part, elem):
    xs = ys = zs = 0.0
    conn = elem.connectivity
    n = float(len(conn))
    for nid in conn:
        node = part.nodes[nid-1]
        xs += node.coordinates[0]
        ys += node.coordinates[1]
        zs += node.coordinates[2]
    return (xs/n, ys/n, zs/n)

def element_volume(elem):
    try:
        return float(elem.getSize())
    except Exception:
        return 1.0

def weighted_quantile_thresholds(values, weights, fracs):
    f1 = fracs[0]
    f2 = fracs[0] + fracs[1]
    pairs = []
    i = 0
    n = len(values)
    while i < n:
        pairs.append((values[i], weights[i]))
        i += 1
    pairs.sort(key=lambda vw: vw[0])
    total_w = 0.0
    for vw in pairs:
        total_w += vw[1]
    total_w += 1e-30
    cum = 0.0
    t1 = pairs[-1][0]
    t2 = pairs[-1][0]
    got1 = False
    for (v, w) in pairs:
        cum += w
        p = cum / total_w
        if (not got1) and (p >= f1):
            t1 = v
            got1 = True
        if p >= f2:
            t2 = v
            break
    return t1, t2

def create_s2s_std(mdl, name, stepName, masterSurf, slaveSurf, propName):
    safe_delete(mdl.interactions, name)
    mdl.SurfaceToSurfaceContactStd(
        name=name,
        createStepName=stepName,
        main=masterSurf,
        secondary=slaveSurf,
        sliding=SMALL,
        thickness=ON,
        interactionProperty=propName,
        adjustMethod=NONE,
        initialClearance=OMIT
    )

# ============================================================
# Helpers : recherche de noeuds pour les BCs de rotation
# ============================================================
def find_node_on_lateral_surface(inst, cyl_r, rad_tol):
    """
    Retourne le premier noeud de l'instance dont le rayon r = sqrt(x^2+y^2)
    est proche de cyl_r (surface laterale du cylindre).
    Le noeud est choisi a mi-hauteur pour etre representatif.
    """
    best_node = None
    best_dz   = None
    z_mid     = CYL_H / 2.0
    for node in inst.nodes:
        x, y, z = node.coordinates
        r = math.sqrt(x*x + y*y)
        if abs(r - cyl_r) < rad_tol:
            dz = abs(z - z_mid)
            if best_dz is None or dz < best_dz:
                best_node = node
                best_dz   = dz
    return best_node

def find_node_on_top_face(inst, cyl_h, z_tol):
    """
    Retourne le premier noeud de l'instance dont la coordonnee Z
    est proche de cyl_h (face superieure du cylindre).
    Le noeud le plus proche du centre est prefere.
    """
    best_node = None
    best_r    = None
    for node in inst.nodes:
        x, y, z = node.coordinates
        if abs(z - cyl_h) < z_tol:
            r = math.sqrt(x*x + y*y)
            if best_r is None or r < best_r:
                best_node = node
                best_r    = r
    return best_node

# ============================================================
# ETAPE 1 : Construction du modele de base
# ============================================================
print("=" * 60)
print("Construction du modele de base...")
print("=" * 60)

safe_delete(mdb.models, BASE_MODEL_NAME)
mdl = mdb.Model(name=BASE_MODEL_NAME)

for pname in ['Part-1', 'Part-3']:
    safe_delete(mdl.parts, pname)

# --- Parts ---
sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=200.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(CYL_R, 0.0))
cyl = mdl.Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
cyl.BaseSolidExtrude(sketch=sk, depth=CYL_H)
del mdl.sketches['__profile__']

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=200.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(PLATEN_R, 0.0))
pl = mdl.Part(name='Part-3', dimensionality=THREE_D, type=DEFORMABLE_BODY)
pl.BaseSolidExtrude(sketch=sk, depth=PLATEN_T)
del mdl.sketches['__profile__']

# --- Materials ---
safe_delete(mdl.materials, PLATEN_MAT['name'])
matP = mdl.Material(name=PLATEN_MAT['name'])
matP.Density(table=((PLATEN_MAT['rho'],),))
matP.Elastic(table=((PLATEN_MAT['E'], PLATEN_MAT['nu']),))

for m in minerals:
    safe_delete(mdl.materials, m['name'])
    mat = mdl.Material(name=m['name'])
    mat.Density(table=((m['rho'],),))
    mat.Elastic(table=((m['E'], m['nu']),))
    dilat = m['cdp'].get('dilat', CDP_DILAT)
    mat.ConcreteDamagedPlasticity(table=((dilat, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC),))

    fc0 = m['cdp']['sigc0']
    ft0 = m['cdp']['sigt0']
    f1_val = R_F1   * fc0
    cres   = R_CRES * fc0
    tres   = max(0.01*ft0, R_TRES*ft0)

    mat.concreteDamagedPlasticity.ConcreteCompressionHardening(
        table=((f1_val, 0.0), (fc0, EIN_PIC), (cres, EIN_END))
    )
    mat.concreteDamagedPlasticity.ConcreteTensionStiffening(
        table=((ft0, 0.0), (tres, ETIN_END))
    )
    mat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=((0.0, 0.0),))
    dt = m['cdp'].get('dt', DT_DAMAGE)
    mat.concreteDamagedPlasticity.ConcreteTensionDamage(
        table=((0.0, 0.0), (dt, ETIN_END))
    )

# --- Sections ---
safe_delete(mdl.sections, 'Section-Platen')
mdl.HomogeneousSolidSection(name='Section-Platen', material=PLATEN_MAT['name'], thickness=None)

for sname, matname in [('Section-Feldspath', 'Feldspath'),
                       ('Section-Quartz',    'Quartz'),
                       ('Section-Biotite',   'Biotite')]:
    safe_delete(mdl.sections, sname)
    mdl.HomogeneousSolidSection(name=sname, material=matname, thickness=None)

pl.Set(name='SET_PLATEN_ALL', cells=pl.cells)
pl.SectionAssignment(region=pl.sets['SET_PLATEN_ALL'], sectionName='Section-Platen')

# --- Mesh ---
elemType_hex = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
elemType_tet = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)

cyl.seedPart(size=ELEMENT_SIZE_CYL, deviationFactor=0.1, minSizeFactor=0.1)
cyl.setMeshControls(regions=cyl.cells, technique=SWEEP, algorithm=MEDIAL_AXIS)
cyl.setElementType(regions=(cyl.cells,), elemTypes=(elemType_hex, elemType_tet))
cyl.generateMesh()

pl.seedPart(size=ELEMENT_SIZE_PLATEN, deviationFactor=0.1, minSizeFactor=0.1)
pl.setMeshControls(regions=pl.cells, technique=SWEEP, algorithm=MEDIAL_AXIS)
pl.setElementType(regions=(pl.cells,), elemTypes=(elemType_hex, elemType_tet))
pl.generateMesh()

# --- Heterogeneity ---
waves = build_gaussian_waves(Nwaves, RANDOM_SEED)
fvals, vols = [], []

for e in cyl.elements:
    cx, cy, cz = element_centroid(cyl, e)
    fvals.append(field_value_at_point(cx, cy, cz, waves))
    vols.append(element_volume(e))

t1, t2 = weighted_quantile_thresholds(fvals, vols, p_frac)

labels_m1, labels_m2, labels_m3 = [], [], []
for i, e in enumerate(cyl.elements):
    f = fvals[i]
    if f <= t1:
        labels_m1.append(e.label)
    elif f <= t2:
        labels_m2.append(e.label)
    else:
        labels_m3.append(e.label)

for sname in ['ELSET_M1', 'ELSET_M2', 'ELSET_M3', 'cylindre']:
    safe_delete(cyl.sets, sname)

if labels_m1:
    cyl.SetFromElementLabels(name='ELSET_M1', elementLabels=labels_m1)
if labels_m2:
    cyl.SetFromElementLabels(name='ELSET_M2', elementLabels=labels_m2)
if labels_m3:
    cyl.SetFromElementLabels(name='ELSET_M3', elementLabels=labels_m3)

cyl.Set(name='cylindre', elements=cyl.elements)

if 'ELSET_M1' in cyl.sets:
    cyl.SectionAssignment(region=cyl.sets['ELSET_M1'], sectionName='Section-Feldspath')
if 'ELSET_M2' in cyl.sets:
    cyl.SectionAssignment(region=cyl.sets['ELSET_M2'], sectionName='Section-Quartz')
if 'ELSET_M3' in cyl.sets:
    cyl.SectionAssignment(region=cyl.sets['ELSET_M3'], sectionName='Section-Biotite')

# --- Assembly ---
aAsm = mdl.rootAssembly
aAsm.DatumCsysByDefault(CARTESIAN)

inst_cyl = aAsm.Instance(name='Part-1-1', part=cyl, dependent=ON)
inst_bot = aAsm.Instance(name='Part-3-1', part=pl,  dependent=ON)
inst_top = aAsm.Instance(name='Part-3-2', part=pl,  dependent=ON)

aAsm.translate(instanceList=('Part-3-1',), vector=(0.0, 0.0, -PLATEN_T))
aAsm.translate(instanceList=('Part-3-2',), vector=(0.0, 0.0,  CYL_H))

rp_top_feat = aAsm.ReferencePoint(point=(0.0, 0.0, CYL_H + PLATEN_T))
rp_bot_feat = aAsm.ReferencePoint(point=(0.0, 0.0, -PLATEN_T))
rp_top = aAsm.referencePoints[rp_top_feat.id]
rp_bot = aAsm.referencePoints[rp_bot_feat.id]

aAsm.Set(name='RP_TOP', referencePoints=(rp_top,))
aAsm.Set(name='RP_BOT', referencePoints=(rp_bot,))

aAsm.regenerate()

aAsm.Set(name='SET_PLATEN_TOP_EL', elements=inst_top.elements)
aAsm.Set(name='SET_PLATEN_BOT_EL', elements=inst_bot.elements)

mdl.RigidBody(name='RB_TOP', refPointRegion=aAsm.sets['RP_TOP'], bodyRegion=aAsm.sets['SET_PLATEN_TOP_EL'])
mdl.RigidBody(name='RB_BOT', refPointRegion=aAsm.sets['RP_BOT'], bodyRegion=aAsm.sets['SET_PLATEN_BOT_EL'])

# --- Surfaces ---
f_cyl_top = inst_cyl.faces.getByBoundingBox(
    -CYL_R-XY_TOL, -CYL_R-XY_TOL, CYL_H-Z_TOL,
     CYL_R+XY_TOL,  CYL_R+XY_TOL, CYL_H+Z_TOL
)
f_cyl_bot = inst_cyl.faces.getByBoundingBox(
    -CYL_R-XY_TOL, -CYL_R-XY_TOL, -Z_TOL,
     CYL_R+XY_TOL,  CYL_R+XY_TOL,  Z_TOL
)
f_cyl_lat = inst_cyl.faces.getByBoundingCylinder(
    (0.0, 0.0, 0.0), (0.0, 0.0, CYL_H), CYL_R+RAD_TOL
)

aAsm.Surface(name='SURF_CYL_TOP', side1Faces=f_cyl_top)
aAsm.Surface(name='SURF_CYL_BOT', side1Faces=f_cyl_bot)
aAsm.Surface(name='SURF_CYL_LAT', side1Faces=f_cyl_lat)

f_top_pl_bot = inst_top.faces.getByBoundingBox(
    -PLATEN_R-XY_TOL, -PLATEN_R-XY_TOL, CYL_H-Z_TOL,
     PLATEN_R+XY_TOL,  PLATEN_R+XY_TOL, CYL_H+Z_TOL
)
f_bot_pl_top = inst_bot.faces.getByBoundingBox(
    -PLATEN_R-XY_TOL, -PLATEN_R-XY_TOL, -Z_TOL,
     PLATEN_R+XY_TOL,  PLATEN_R+XY_TOL,  Z_TOL
)

aAsm.Surface(name='SURF_PLATEN_TOP_BOT', side1Faces=f_top_pl_bot)
aAsm.Surface(name='SURF_PLATEN_BOT_TOP', side1Faces=f_bot_pl_top)

# --- Contact ---
ip = mdl.ContactProperty('IntProp-1')
ip.TangentialBehavior(formulation=FRICTIONLESS)
ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                  constraintEnforcementMethod=DEFAULT)

create_s2s_std(
    mdl, 'Int-Top', 'Initial',
    aAsm.surfaces['SURF_PLATEN_TOP_BOT'],
    aAsm.surfaces['SURF_CYL_TOP'],
    'IntProp-1'
)
create_s2s_std(
    mdl, 'Int-Bot', 'Initial',
    aAsm.surfaces['SURF_PLATEN_BOT_TOP'],
    aAsm.surfaces['SURF_CYL_BOT'],
    'IntProp-1'
)

# --- Steps ---
T2 = float(STEP2_TIME)
mdl.StaticStep(
    name='Step-1', previous='Initial', nlgeom=ON,
    timePeriod=1.0, initialInc=1.0, minInc=1e-5, maxInc=1.0
)
mdl.StaticStep(
    name='Step-2', previous='Step-1', nlgeom=ON,
    timePeriod=T2,
    initialInc=T2/200.0,
    minInc=T2/1.0e7,
    maxInc=T2/20.0,
    stabilizationMagnitude=0.002,
    stabilizationMethod=DISSIPATED_ENERGY_FRACTION
)

# --- History output ---
mdl.HistoryOutputRequest(
    name='H-RP_TOP', createStepName='Step-1',
    region=aAsm.sets['RP_TOP'], variables=('RF3', 'U3')
)
mdl.historyOutputRequests['H-RP_TOP'].setValuesInStep(
    stepName='Step-2', variables=('RF3', 'U3')
)

# --- BCs standards (platens) ---
mdl.EncastreBC(
    name='BC_RP_BOT', createStepName='Initial',
    region=aAsm.sets['RP_BOT']
)
mdl.DisplacementBC(
    name='BC_RP_TOP_GUIDE', createStepName='Initial',
    region=aAsm.sets['RP_TOP'],
    u1=0.0, u2=0.0, u3=UNSET, ur1=0.0, ur2=0.0, ur3=0.0
)
mdl.DisplacementBC(
    name='BC_RP_TOP_U3', createStepName='Step-1',
    region=aAsm.sets['RP_TOP'], u3=0.0
)
mdl.boundaryConditions['BC_RP_TOP_U3'].setValuesInStep(
    stepName='Step-2', u3=TOP_DISP
)

# ============================================================
# --- BCs supplementaires : blocage des rotations ---
#
# BC_ROT_LAT  : noeud a mi-hauteur sur la surface laterale
#               -> ur1 = ur2 = ur3 = 0
# BC_ROT_FACE : noeud le plus central sur la face superieure
#               -> ur1 = ur2 = ur3 = 0
#
# Note : les elements C3D8R n'ont pas de DDL de rotation.
# Ces BCs sont sans effet cinematique pour des elements
# volumiques purs, mais peuvent etre utiles si le modele
# contient des elements avec DDL de rotation (ex. coques,
# poutres) ou si l'on souhaite les conserver pour robustesse.
# ============================================================

# -- Noeud lateral --
node_lat = find_node_on_lateral_surface(inst_cyl, CYL_R, RAD_TOL)
if node_lat is None:
    print("ATTENTION : aucun noeud trouve sur la surface laterale.")
else:
    nodes_lat = inst_cyl.nodes.sequenceFromLabels((node_lat.label,))
    safe_delete(aAsm.sets, 'SET_NODE_LAT_ROT')
    aAsm.Set(name='SET_NODE_LAT_ROT', nodes=nodes_lat)
    mdl.DisplacementBC(
        name='BC_ROT_LAT',
        createStepName='Initial',
        region=aAsm.sets['SET_NODE_LAT_ROT'],
        ur1=0.0, ur2=0.0, ur3=0.0
    )
    r_lat = math.sqrt(node_lat.coordinates[0]**2 + node_lat.coordinates[1]**2)
    print("BC_ROT_LAT -> noeud %d  (r = %.4f mm, z = %.4f mm)"
          % (node_lat.label, r_lat, node_lat.coordinates[2]))

# -- Noeud face superieure --
node_top = find_node_on_top_face(inst_cyl, CYL_H, Z_TOL)
if node_top is None:
    print("ATTENTION : aucun noeud trouve sur la face superieure.")
else:
    nodes_top = inst_cyl.nodes.sequenceFromLabels((node_top.label,))
    safe_delete(aAsm.sets, 'SET_NODE_TOP_ROT')
    aAsm.Set(name='SET_NODE_TOP_ROT', nodes=nodes_top)
    mdl.DisplacementBC(
        name='BC_ROT_FACE',
        createStepName='Initial',
        region=aAsm.sets['SET_NODE_TOP_ROT'],
        ur1=0.0, ur2=0.0, ur3=0.0
    )
    print("BC_ROT_FACE -> noeud %d  (x = %.4f mm, y = %.4f mm, z = %.4f mm)"
          % (node_top.label,
             node_top.coordinates[0],
             node_top.coordinates[1],
             node_top.coordinates[2]))

print("Modele de base construit.")

# ============================================================
# ETAPE 2 : Creation des modeles copies avec pression specifique
#           SANS creation ni lancement de job
# ============================================================
for PRESSURE_MAG in PRESSURE_LIST:
    MODEL_NAME = '%dMPa_model' % PRESSURE_MAG

    print("--- Creation du modele %s (confinement = %d MPa) ---"
          % (MODEL_NAME, PRESSURE_MAG))

    safe_delete(mdb.models, MODEL_NAME)
    mdb.Model(name=MODEL_NAME, objectToCopy=mdb.models[BASE_MODEL_NAME])

    mdl_p = mdb.models[MODEL_NAME]

    safe_delete(mdl_p.loads, 'Load-Pressure')
    mdl_p.Pressure(
        name='Load-Pressure',
        createStepName='Step-1',
        region=mdl_p.rootAssembly.surfaces['SURF_CYL_LAT'],
        magnitude=PRESSURE_MAG
    )

print("=" * 60)
print("Tous les modeles ont ete crees. Aucun job n'a ete lance.")
print("BCs de rotation incluses : BC_ROT_LAT + BC_ROT_FACE")
print("=" * 60)

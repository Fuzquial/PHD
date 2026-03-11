# -*- coding: utf-8 -*-
# Abaqus/CAE Python script
# Pair-based contact: frictionless + hard, small sliding
# Triaxial: confinement step + axial compression step
# Minerals are TRUE CDP materials via Abaqus API
# Boucle sur pressions de confinement : 5, 10, 30, 50 MPa
# VERSION : heterogeneite par Champ Aleatoire Correle (CRF) log-normal
#           methode spectrale FFT (Gutjahr 1989) - Le Goc et al. 2015

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import math

# numpy est disponible dans Abaqus Python
import numpy as np

# ============================================================
# User parameters
# ============================================================
BASE_MODEL_NAME = 'Model-Base'

CYL_R    = 2.5
CYL_H    = 10.0
PLATEN_R = 3.0
PLATEN_T = 2.0

ELEMENT_SIZE_CYL    = 0.1
ELEMENT_SIZE_PLATEN = 0.6
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

# Pressions de confinement a simuler (MPa)
PRESSURE_LIST = [5, 10, 30, 50]

TOP_DISP    = -0.5
STEP2_TIME  = 1200.0

# ── Parametres CRF (remplacent a, Nwaves, RANDOM_SEED) ──
RANDOM_SEED = 42
LAMBDA_RF   = 0.8    # Longueur de correlation physique [mm] (ex-parametre a)
SIGMA_RF    = 0.4    # Ecart-type du champ log-normal
MU_RF       = 1.0    # Moyenne du champ log-normal

# Resolution de la grille CRF (nx, ny, nz)
# Proportionnelle au domaine : CYL 5x5x10 mm -> grille 32x32x64
CRF_NX, CRF_NY, CRF_NZ = 32, 32, 64

# Fractions volumiques des phases (Feldspath, Quartz, Biotite)
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


def element_centroid(part, elem):
    xs = ys = zs = 0.0
    conn = elem.connectivity
    n = float(len(conn))
    for nid in conn:
        node = part.nodes[nid - 1]
        xs += node.coordinates[0]
        ys += node.coordinates[1]
        zs += node.coordinates[2]
    return xs / n, ys / n, zs / n


def element_volume(elem):
    try:
        return float(elem.getSize())
    except Exception:
        return 1.0


def weighted_quantile_thresholds(values, weights, fracs):
    """Seuils ponderes par volume pour respecter les fractions volumiques.
    Compatible Abaqus Python (pas de generateur, pas de zip)."""
    f1 = fracs[0]
    f2 = fracs[0] + fracs[1]

    # Construction de la liste de paires sans zip ni generateur
    n = len(values)
    pairs = []
    i = 0
    while i < n:
        pairs.append((values[i], weights[i]))
        i += 1

    pairs.sort(key=lambda vw: vw[0])

    # Somme totale sans generateur
    total_w = 1e-30
    for vw in pairs:
        total_w += vw[1]

    cum  = 0.0
    t1   = pairs[-1][0]
    t2   = pairs[-1][0]
    got1 = False

    for vw in pairs:
        v = vw[0]
        w = vw[1]
        cum += w
        p = cum / total_w
        if (not got1) and p >= f1:
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
# Generation du Champ Aleatoire Correle (CRF) log-normal
# Methode spectrale (Gutjahr 1989) via FFT numpy
# Remplace build_gaussian_waves + field_value_at_point
# ============================================================

def generate_crf_lognormal(nx, ny, nz, lambda_rf, sigma_rf, mu_rf,
                           x_range, y_range, z_range, seed=42):
    """
    Genere un champ log-normal correle spatialement sur une grille reguliere.

    Parametres
    ----------
    nx, ny, nz   : resolution de la grille
    lambda_rf    : longueur de correlation [mm]
    sigma_rf     : ecart-type cible du champ log-normal
    mu_rf        : moyenne cible du champ log-normal
    x_range      : (xmin, xmax) du domaine [mm]
    y_range      : (ymin, ymax) du domaine [mm]
    z_range      : (zmin, zmax) du domaine [mm]
    seed         : graine aleatoire

    Retour
    ------
    field_ln     : ndarray (nx, ny, nz) — champ log-normal
    xs, ys, zs   : coordonnees des noeuds de la grille
    """
    np.random.seed(seed)

    Lx = x_range[1] - x_range[0]
    Ly = y_range[1] - y_range[0]
    Lz = z_range[1] - z_range[0]

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    # Frequences spatiales angulaires
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Spectre de puissance gaussien : S(k) = (lambda^3 * pi^1.5) * exp(-k^2*lambda^2/4)
    S = (lambda_rf**3 * np.pi**1.5) * np.exp(-K2 * lambda_rf**2 / 4.0)

    # Bruit blanc complexe + filtrage spectral
    noise   = (np.random.randn(nx, ny, nz) + 1j * np.random.randn(nx, ny, nz)) / np.sqrt(2.0)
    field_f = np.sqrt(S) * noise

    # Retour espace reel + normalisation (mu=0, sigma=1)
    field_g = np.real(np.fft.ifftn(field_f))
    field_g = (field_g - field_g.mean()) / (field_g.std() + 1e-12)

    # Transformation log-normale
    sigma_ln = np.sqrt(np.log(1.0 + (sigma_rf / mu_rf)**2))
    mu_ln    = np.log(mu_rf) - 0.5 * sigma_ln**2
    field_ln = np.exp(mu_ln + sigma_ln * field_g)

    # Coordonnees de la grille
    xs = np.linspace(x_range[0] + 0.5*dx, x_range[1] - 0.5*dx, nx)
    ys = np.linspace(y_range[0] + 0.5*dy, y_range[1] - 0.5*dy, ny)
    zs = np.linspace(z_range[0] + 0.5*dz, z_range[1] - 0.5*dz, nz)

    return field_ln, xs, ys, zs


def interpolate_crf_at_point(px, py, pz, field_ln, xs, ys, zs):
    """
    Interpolation au plus proche voisin du champ CRF au point (px, py, pz).
    Simple et robuste pour l'usage Abaqus element-par-element.
    """
    ix = int(np.argmin(np.abs(xs - px)))
    iy = int(np.argmin(np.abs(ys - py)))
    iz = int(np.argmin(np.abs(zs - pz)))
    return float(field_ln[ix, iy, iz])


# ============================================================
# ETAPE 1 : Construction du modele de base
# ============================================================
print("Construction du modele de base (CRF)...")

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

    fc0  = m['cdp']['sigc0']
    ft0  = m['cdp']['sigt0']
    f1   = R_F1   * fc0
    cres = R_CRES * fc0
    tres = max(0.01 * ft0, R_TRES * ft0)

    mat.concreteDamagedPlasticity.ConcreteCompressionHardening(
        table=((f1, 0.0), (fc0, EIN_PIC), (cres, EIN_END))
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

# ============================================================
# Heterogeneite par CRF log-normal (remplace les ondes gaussiennes)
# ============================================================
print("Generation du champ CRF log-normal...")

# Domaine englobe le cylindre avec une legere marge
margin  = LAMBDA_RF
x_range = (-CYL_R - margin, CYL_R + margin)
y_range = (-CYL_R - margin, CYL_R + margin)
z_range = (-margin,         CYL_H + margin)

field_ln, xs, ys, zs = generate_crf_lognormal(
    CRF_NX, CRF_NY, CRF_NZ,
    LAMBDA_RF, SIGMA_RF, MU_RF,
    x_range, y_range, z_range,
    seed=RANDOM_SEED
)

print("  CRF genere : mean=%.3f  std=%.3f  min=%.3f  max=%.3f" % (
    float(field_ln.mean()), float(field_ln.std()),
    float(field_ln.min()),  float(field_ln.max())
))

# Valeur CRF et volume pour chaque element du cylindre
print("Interpolation du CRF sur les elements...")
fvals = []
vols  = []

for e in cyl.elements:
    cx, cy, cz = element_centroid(cyl, e)
    fvals.append(interpolate_crf_at_point(cx, cy, cz, field_ln, xs, ys, zs))
    vols.append(element_volume(e))

# Seuils ponderes par volume
t1, t2 = weighted_quantile_thresholds(fvals, vols, p_frac)
print("  Seuils CRF : t1=%.4f  t2=%.4f" % (t1, t2))

# Attribution des phases
labels_m1, labels_m2, labels_m3 = [], [], []
for i, e in enumerate(cyl.elements):
    f = fvals[i]
    if f <= t1:
        labels_m1.append(e.label)
    elif f <= t2:
        labels_m2.append(e.label)
    else:
        labels_m3.append(e.label)

print("  Feldspath : %d elem  Quartz : %d elem  Biotite : %d elem" % (
    len(labels_m1), len(labels_m2), len(labels_m3)
))

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
aAsm.translate(instanceList=('Part-3-2',), vector=(0.0, 0.0, CYL_H))

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
    -CYL_R - XY_TOL, -CYL_R - XY_TOL, CYL_H - Z_TOL,
     CYL_R + XY_TOL,  CYL_R + XY_TOL, CYL_H + Z_TOL
)
f_cyl_bot = inst_cyl.faces.getByBoundingBox(
    -CYL_R - XY_TOL, -CYL_R - XY_TOL, -Z_TOL,
     CYL_R + XY_TOL,  CYL_R + XY_TOL,  Z_TOL
)
f_cyl_lat = inst_cyl.faces.getByBoundingCylinder(
    (0.0, 0.0, 0.0), (0.0, 0.0, CYL_H), CYL_R + RAD_TOL
)

aAsm.Surface(name='SURF_CYL_TOP', side1Faces=f_cyl_top)
aAsm.Surface(name='SURF_CYL_BOT', side1Faces=f_cyl_bot)
aAsm.Surface(name='SURF_CYL_LAT', side1Faces=f_cyl_lat)

f_top_pl_bot = inst_top.faces.getByBoundingBox(
    -PLATEN_R - XY_TOL, -PLATEN_R - XY_TOL, CYL_H - Z_TOL,
     PLATEN_R + XY_TOL,  PLATEN_R + XY_TOL, CYL_H + Z_TOL
)
f_bot_pl_top = inst_bot.faces.getByBoundingBox(
    -PLATEN_R - XY_TOL, -PLATEN_R - XY_TOL, -Z_TOL,
     PLATEN_R + XY_TOL,  PLATEN_R + XY_TOL,  Z_TOL
)

aAsm.Surface(name='SURF_PLATEN_TOP_BOT', side1Faces=f_top_pl_bot)
aAsm.Surface(name='SURF_PLATEN_BOT_TOP', side1Faces=f_bot_pl_top)

# --- Contact ---
ip = mdl.ContactProperty('IntProp-1')
ip.TangentialBehavior(formulation=FRICTIONLESS)
ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                  constraintEnforcementMethod=DEFAULT)

create_s2s_std(mdl, 'Int-Top', 'Initial',
               aAsm.surfaces['SURF_PLATEN_TOP_BOT'],
               aAsm.surfaces['SURF_CYL_TOP'], 'IntProp-1')
create_s2s_std(mdl, 'Int-Bot', 'Initial',
               aAsm.surfaces['SURF_PLATEN_BOT_TOP'],
               aAsm.surfaces['SURF_CYL_BOT'], 'IntProp-1')

# --- Steps ---
T2 = float(STEP2_TIME)
mdl.StaticStep(
    name='Step-1', previous='Initial', nlgeom=ON,
    timePeriod=1.0, initialInc=1.0, minInc=1e-5, maxInc=1.0
)
mdl.StaticStep(
    name='Step-2', previous='Step-1', nlgeom=ON,
    timePeriod=T2,
    initialInc=T2 / 200.0,
    minInc=T2 / 1.0e7,
    maxInc=T2 / 20.0,
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

# --- BCs ---
mdl.EncastreBC(name='BC_RP_BOT', createStepName='Initial',
               region=aAsm.sets['RP_BOT'])

mdl.DisplacementBC(
    name='BC_RP_TOP_GUIDE', createStepName='Initial',
    region=aAsm.sets['RP_TOP'],
    u1=0.0, u2=0.0, u3=UNSET, ur1=0.0, ur2=0.0, ur3=0.0
)
mdl.DisplacementBC(
    name='BC_RP_TOP_U3', createStepName='Step-1',
    region=aAsm.sets['RP_TOP'], u3=0.0
)
mdl.boundaryConditions['BC_RP_TOP_U3'].setValuesInStep(stepName='Step-2', u3=TOP_DISP)

print("Modele de base construit (CRF).")

# ============================================================
# ETAPE 2 : Copies avec pression de confinement specifique
# ============================================================
for PRESSURE_MAG in PRESSURE_LIST:
    MODEL_NAME = '%dMPa_CRF_model' % PRESSURE_MAG

    print("--- Creation du modele %s (confinement = %d MPa) ---" % (MODEL_NAME, PRESSURE_MAG))

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

print("Tous les modeles CRF ont ete crees. Aucun job n'a ete lance.")

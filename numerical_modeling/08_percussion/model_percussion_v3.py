# -*- coding: utf-8 -*-
# Abaqus/CAE Python script
# Percussion dynamique sur parallelepipede heterogene (granit)
# Indenteur rigide analytique (profil Dist2.inp, mis a l'echelle)
# Modele complet pret a lancer : geometrie, materiaux, maillage,
#   heterogeneite, contact, BCs, conditions initiales, step, loads, outputs
#
# Convention : Z = axe vertical (impact en -Z)

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# 1. PARAMETRES UTILISATEUR — a piloter par PSO
# ============================================================
MODEL_NAME = 'Model-Percussion'
JOB_NAME   = 'Job-Percussion'

# --- Geometrie echantillon (mm) ---
BOX_LX = 25.0
BOX_LY = 25.0
BOX_LZ = 15.0

# --- Zone raffinee (centree sur la face superieure, zone d'impact) ---
REFINE_HALF_X = 4.0    # demi-largeur X de la zone raffinee
REFINE_HALF_Y = 4.0    # demi-largeur Y
REFINE_DEPTH  = 6.0    # profondeur depuis la face superieure

# --- Maillage ---
ELEMENT_SIZE_FINE   = 0.2    # zone d'impact
ELEMENT_SIZE_COARSE = 1.5    # zone eloignee
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

# --- Rayon max de l'indenteur (mm) — decouple de la taille du domaine ---
INDENTER_RMAX = 2.5

# --- Champ aleatoire heterogene ---
RANDOM_SEED = 42
a = 3          # longueur de correlation ~2mm (2*pi/a) -> ~10 elements fins par grain
Nwaves = 300
p_frac = (0.30, 0.50, 0.20)   # Feldspath, Quartz, Biotite

# --- Proprietes mecaniques des mineraux (variables PSO) ---
minerals = [
    dict(name='Feldspath', E=48000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=180.0, sigt0=8.0,  dilat=30.0, dt=0.98)),
    dict(name='Quartz',    E=85000.0, nu=0.25, rho=2.2e-9,
         cdp=dict(sigc0=350.0, sigt0=10.0, dilat=35.0, dt=0.98)),
    dict(name='Biotite',   E=30000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=250.0, sigt0=5.0,  dilat=30.0, dt=0.98)),
]

# --- CDP global params ---
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

# --- Chargement dynamique (adapte de Dist2.inp) ---
CONFINEMENT  = 30.0       # MPa - contrainte initiale isotrope + pression
IND_VELOCITY = -2950.0    # mm/s - vitesse initiale indenteur (en -Z)
IND_FORCE    = -2880.0    # N    - force concentree sur indenteur (en Z)
IND_MASS     = 0.0069     # tonnes (6.9 kg)

# --- Step dynamique ---
STEP_TIME    = 0.001      # s
NUM_OUTPUT_INTERVALS = 200 # nombre d'intervalles pour les outputs

# --- Tolerances selection faces ---
Z_TOL   = 1e-3
XY_TOL  = 1e-2

# ============================================================
# 2. FONCTIONS UTILITAIRES
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
        kx /= norm; ky /= norm; kz /= norm
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
        node = part.nodes[nid - 1]
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
    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total_w = sum([w for _, w in pairs]) + 1e-30
    cum = 0.0
    t1 = t2 = pairs[-1][0]
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

# ============================================================
# 3. PART 1 : ECHANTILLON (parallelepipede)
# ============================================================
print("=" * 60)
print("Construction du modele '%s'" % MODEL_NAME)
print("=" * 60)

safe_delete(mdb.models, MODEL_NAME)
mdl = mdb.Model(name=MODEL_NAME)

sk = mdl.ConstrainedSketch(name='__profile__', sheetSize=200.0)
sk.rectangle(point1=(0.0, 0.0), point2=(BOX_LX, BOX_LY))
box = mdl.Part(name='Part-Box', dimensionality=THREE_D, type=DEFORMABLE_BODY)
box.BaseSolidExtrude(sketch=sk, depth=BOX_LZ)
del mdl.sketches['__profile__']

# ============================================================
# 4. MATERIAUX CDP
# ============================================================
for m in minerals:
    safe_delete(mdl.materials, m['name'])
    mat = mdl.Material(name=m['name'])
    mat.Density(table=((m['rho'],),))
    mat.Elastic(table=((m['E'], m['nu']),))

    dilat = m['cdp'].get('dilat', CDP_DILAT)
    mat.ConcreteDamagedPlasticity(
        table=((dilat, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC),)
    )

    fc0 = m['cdp']['sigc0']
    ft0 = m['cdp']['sigt0']
    f1   = R_F1   * fc0
    cres = R_CRES * fc0
    tres = max(0.01 * ft0, R_TRES * ft0)

    mat.concreteDamagedPlasticity.ConcreteCompressionHardening(
        table=((f1, 0.0), (fc0, EIN_PIC), (cres, EIN_END))
    )
    mat.concreteDamagedPlasticity.ConcreteTensionStiffening(
        table=((ft0, 0.0), (tres, ETIN_END))
    )
    mat.concreteDamagedPlasticity.ConcreteCompressionDamage(
        table=((0.0, 0.0),)
    )
    dt = m['cdp'].get('dt', DT_DAMAGE)
    mat.concreteDamagedPlasticity.ConcreteTensionDamage(
        table=((0.0, 0.0), (dt, ETIN_END))
    )

# ============================================================
# 5. PARTITIONNEMENT + MAILLAGE + HETEROGENEITE
# ============================================================
for sname, matname in [('Section-Feldspath', 'Feldspath'),
                       ('Section-Quartz',    'Quartz'),
                       ('Section-Biotite',   'Biotite')]:
    safe_delete(mdl.sections, sname)
    mdl.HomogeneousSolidSection(name=sname, material=matname, thickness=None)

# --- Partitionnement : 5 plans pour creer la zone raffinee ---
cx = BOX_LX / 2.0
cy = BOX_LY / 2.0

dp_x1 = box.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE,
                                        offset=cx - REFINE_HALF_X)
box.PartitionCellByDatumPlane(datumPlane=box.datums[dp_x1.id],
                               cells=box.cells)

dp_x2 = box.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE,
                                        offset=cx + REFINE_HALF_X)
box.PartitionCellByDatumPlane(datumPlane=box.datums[dp_x2.id],
                               cells=box.cells)

dp_y1 = box.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                        offset=cy - REFINE_HALF_Y)
box.PartitionCellByDatumPlane(datumPlane=box.datums[dp_y1.id],
                               cells=box.cells)

dp_y2 = box.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                        offset=cy + REFINE_HALF_Y)
box.PartitionCellByDatumPlane(datumPlane=box.datums[dp_y2.id],
                               cells=box.cells)

dp_z1 = box.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE,
                                        offset=BOX_LZ - REFINE_DEPTH)
box.PartitionCellByDatumPlane(datumPlane=box.datums[dp_z1.id],
                               cells=box.cells)

print("Partitionnement : %d cellules creees." % len(box.cells))

# --- Maillage : grossier partout, puis raffine dans la zone d'impact ---
elemType_hex = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=EXPLICIT)
elemType_tet = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=EXPLICIT)

box.setMeshControls(regions=box.cells, elemShape=HEX,
                    technique=STRUCTURED)
box.setElementType(regions=(box.cells,), elemTypes=(elemType_hex, elemType_tet))

# Seed grossier sur tout le part
box.seedPart(size=ELEMENT_SIZE_COARSE, deviationFactor=0.1, minSizeFactor=0.1)

# Re-seed fin sur les aretes de la zone raffinee
tol = 0.01
refined_edges = box.edges.getByBoundingBox(
    cx - REFINE_HALF_X - tol, cy - REFINE_HALF_Y - tol, BOX_LZ - REFINE_DEPTH - tol,
    cx + REFINE_HALF_X + tol, cy + REFINE_HALF_Y + tol, BOX_LZ + tol
)
box.seedEdgeBySize(edges=refined_edges, size=ELEMENT_SIZE_FINE,
                   deviationFactor=0.1, constraint=FINER)

print("Seeding : fin=%.2f mm (%d aretes), grossier=%.2f mm"
      % (ELEMENT_SIZE_FINE, len(refined_edges), ELEMENT_SIZE_COARSE))

box.generateMesh()
print("Maillage : %d elements, %d noeuds" % (len(box.elements), len(box.nodes)))

# --- Champ d'heterogeneite ---
waves = build_gaussian_waves(Nwaves, RANDOM_SEED)
fvals, vols = [], []
for e in box.elements:
    cx, cy, cz = element_centroid(box, e)
    fvals.append(field_value_at_point(cx, cy, cz, waves))
    vols.append(element_volume(e))

t1, t2 = weighted_quantile_thresholds(fvals, vols, p_frac)

labels_m1, labels_m2, labels_m3 = [], [], []
for i, e in enumerate(box.elements):
    f = fvals[i]
    if f <= t1:
        labels_m1.append(e.label)
    elif f <= t2:
        labels_m2.append(e.label)
    else:
        labels_m3.append(e.label)

for sname in ['ELSET_FELD', 'ELSET_QTZ', 'ELSET_BIO', 'ALL_ELEMENTS']:
    safe_delete(box.sets, sname)

if labels_m1:
    box.SetFromElementLabels(name='ELSET_FELD', elementLabels=labels_m1)
    box.SectionAssignment(region=box.sets['ELSET_FELD'], sectionName='Section-Feldspath')
if labels_m2:
    box.SetFromElementLabels(name='ELSET_QTZ', elementLabels=labels_m2)
    box.SectionAssignment(region=box.sets['ELSET_QTZ'], sectionName='Section-Quartz')
if labels_m3:
    box.SetFromElementLabels(name='ELSET_BIO', elementLabels=labels_m3)
    box.SectionAssignment(region=box.sets['ELSET_BIO'], sectionName='Section-Biotite')

box.Set(name='ALL_ELEMENTS', elements=box.elements)
print("Heterogeneite : Feld=%d, Qtz=%d, Bio=%d" % (len(labels_m1), len(labels_m2), len(labels_m3)))

# ============================================================
# 6. PART 2 : INDENTEUR RIGIDE ANALYTIQUE (mis a l'echelle)
# ============================================================
IND_REF   = 8.0
IND_SCALE = INDENTER_RMAX / IND_REF

IND_START   = (8.0,              11.5)
IND_LINE    = (3.63730669587198, 2.09999999996075)
IND_ARC_C   = (0.0,              4.2)
IND_ARC_END = (0.0,              0.0)

s_start   = (IND_START[0]   * IND_SCALE, IND_START[1]   * IND_SCALE)
s_line    = (IND_LINE[0]    * IND_SCALE, IND_LINE[1]    * IND_SCALE)
s_arc_c   = (IND_ARC_C[0]  * IND_SCALE, IND_ARC_C[1]  * IND_SCALE)
s_arc_end = (IND_ARC_END[0] * IND_SCALE, IND_ARC_END[1] * IND_SCALE)

indenter = mdl.Part(name='Part-Indenter', dimensionality=THREE_D,
                    type=ANALYTIC_RIGID_SURFACE)

sk2 = mdl.ConstrainedSketch(name='__profile_ind__', sheetSize=200.0)
sk2.ConstructionLine(point1=(0.0, -10.0), point2=(0.0, 20.0))
sk2.Line(point1=s_start, point2=s_line)
sk2.ArcByCenterEnds(center=s_arc_c, point1=s_line, point2=s_arc_end,
                    direction=CLOCKWISE)
indenter.AnalyticRigidSurfRevolve(sketch=sk2)
del mdl.sketches['__profile_ind__']

rp_ind_feat = indenter.ReferencePoint(point=(0.0, s_start[1], 0.0))

# Surface nommee sur la part analytique (necessaire pour le contact)
indenter.Surface(name='SURF_ANALYTICAL', side1Faces=indenter.faces)

print("Indenteur cree (scale=%.4f, Rmax=%.3f, H=%.3f mm)" %
      (IND_SCALE, s_start[0], s_start[1]))

# ============================================================
# 7. ASSEMBLY : instances + positionnement indenteur
# ============================================================
aAsm = mdl.rootAssembly
aAsm.DatumCsysByDefault(CARTESIAN)

inst_box = aAsm.Instance(name='Part-Box-1', part=box, dependent=ON)
inst_ind = aAsm.Instance(name='Part-Indenter-1', part=indenter, dependent=ON)

# Rotation : axe de revolution Y -> Z (rotation +90 deg autour de X)
aAsm.rotate(
    instanceList=('Part-Indenter-1',),
    axisPoint=(0.0, 0.0, 0.0),
    axisDirection=(1.0, 0.0, 0.0),
    angle=90.0
)

# Translation : pointe de l'indenteur au centre de la face superieure
aAsm.translate(
    instanceList=('Part-Indenter-1',),
    vector=(BOX_LX / 2.0, BOX_LY / 2.0, BOX_LZ)
)

aAsm.regenerate()

# ============================================================
# 8. SETS ET SURFACES
# ============================================================
# RP de l'indenteur dans l'assembly
rp_ind_key = inst_ind.referencePoints.keys()[0]
rp_ind_asm = inst_ind.referencePoints[rp_ind_key]
aAsm.Set(name='RP_INDENTER', referencePoints=(rp_ind_asm,))

# Tous les elements de l'echantillon
aAsm.Set(name='ALL_SAMPLE', elements=inst_box.elements)

# Face inferieure z = 0 (BOTTOM)
f_bot = inst_box.faces.getByBoundingBox(
    -XY_TOL, -XY_TOL, -Z_TOL,
    BOX_LX + XY_TOL, BOX_LY + XY_TOL, Z_TOL
)
aAsm.Set(name='NSET_BOTTOM', faces=f_bot)
aAsm.Surface(name='SURF_BOTTOM', side1Faces=f_bot)

# Face superieure z = BOX_LZ (CONTACT)
f_top = inst_box.faces.getByBoundingBox(
    -XY_TOL, -XY_TOL, BOX_LZ - Z_TOL,
    BOX_LX + XY_TOL, BOX_LY + XY_TOL, BOX_LZ + Z_TOL
)
aAsm.Surface(name='SURF_TOP', side1Faces=f_top)

# 4 faces laterales
f_xmin = inst_box.faces.getByBoundingBox(
    -XY_TOL, -XY_TOL, -Z_TOL,
     XY_TOL, BOX_LY + XY_TOL, BOX_LZ + Z_TOL
)
f_xmax = inst_box.faces.getByBoundingBox(
    BOX_LX - XY_TOL, -XY_TOL, -Z_TOL,
    BOX_LX + XY_TOL, BOX_LY + XY_TOL, BOX_LZ + Z_TOL
)
f_ymin = inst_box.faces.getByBoundingBox(
    -XY_TOL, -XY_TOL, -Z_TOL,
    BOX_LX + XY_TOL, XY_TOL, BOX_LZ + Z_TOL
)
f_ymax = inst_box.faces.getByBoundingBox(
    -XY_TOL, BOX_LY - XY_TOL, -Z_TOL,
    BOX_LX + XY_TOL, BOX_LY + XY_TOL, BOX_LZ + Z_TOL
)

aAsm.Surface(name='SURF_LAT', side1Faces=f_xmin + f_xmax + f_ymin + f_ymax)

# Surface top + lateral pour pression de confinement
aAsm.SurfaceByBoolean(name='SURF_TOP_AND_LAT', operation=UNION,
                       surfaces=(aAsm.surfaces['SURF_TOP'],
                                 aAsm.surfaces['SURF_LAT']))

print("Sets et surfaces crees.")

# ============================================================
# 9. CONTACT (Explicit General Contact)
# ============================================================
# Propriete de contact : sans frottement + hard contact
ip = mdl.ContactProperty('IntProp-1')
ip.TangentialBehavior(formulation=FRICTIONLESS)
ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                  constraintEnforcementMethod=DEFAULT)

# General Contact Explicit : detecte automatiquement toutes les surfaces
mdl.ContactExp(name='Contact-General', createStepName='Initial')
mdl.interactions['Contact-General'].includedPairs.setValuesInStep(
    stepName='Initial',
    useAllstar=ON
)
mdl.interactions['Contact-General'].contactPropertyAssignments.appendInStep(
    stepName='Initial',
    assignments=((GLOBAL, SELF, 'IntProp-1'),)
)

print("General Contact Explicit cree.")

# ============================================================
# 10. STEP DYNAMIQUE EXPLICITE
# ============================================================
mdl.ExplicitDynamicsStep(
    name='Step-1',
    previous='Initial',
    nlgeom=ON,
    timePeriod=STEP_TIME
)
print("Step Explicit cree (T=%.4f s)." % STEP_TIME)

# ============================================================
# 11. CONDITIONS AUX LIMITES
# ============================================================
mdl.EncastreBC(
    name='BC_BOTTOM',
    createStepName='Initial',
    region=aAsm.sets['NSET_BOTTOM']
)

mdl.DisplacementBC(
    name='BC_INDENTER_GUIDE',
    createStepName='Initial',
    region=aAsm.sets['RP_INDENTER'],
    u1=0.0, u2=0.0, u3=UNSET,
    ur1=0.0, ur2=0.0, ur3=0.0
)
print("BCs creees (encastre bas + guidage indenteur en Z).")

# ============================================================
# 12. CONDITIONS INITIALES
# ============================================================
mdl.Stress(
    name='IC_STRESS',
    region=aAsm.sets['ALL_SAMPLE'],
    distributionType=UNIFORM,
    sigma11=-CONFINEMENT, sigma22=-CONFINEMENT, sigma33=-CONFINEMENT,
    sigma12=0.0, sigma13=0.0, sigma23=0.0
)

mdl.Velocity(
    name='IC_VELOCITY',
    region=aAsm.sets['RP_INDENTER'],
    velocity1=0.0, velocity2=0.0, velocity3=IND_VELOCITY,
    omega=0.0
)

mdl.rootAssembly.engineeringFeatures.PointMassInertia(
    name='Inertia-Indenter',
    region=aAsm.sets['RP_INDENTER'],
    mass=IND_MASS,
    alpha=0.0, composite=0.0
)
print("Conditions initiales creees (sigma0=%.0f MPa, V0=%.0f mm/s, m=%.4f t)."
      % (CONFINEMENT, IND_VELOCITY, IND_MASS))

# ============================================================
# 13. CHARGEMENTS
# ============================================================
mdl.Pressure(
    name='Load-Confinement',
    createStepName='Step-1',
    region=aAsm.surfaces['SURF_TOP_AND_LAT'],
    magnitude=CONFINEMENT
)

mdl.ConcentratedForce(
    name='Load-Impact',
    createStepName='Step-1',
    region=aAsm.sets['RP_INDENTER'],
    cf3=IND_FORCE
)
print("Chargements crees (P=%.0f MPa, F=%.0f N)." % (CONFINEMENT, IND_FORCE))

# ============================================================
# 14. FIELD + HISTORY OUTPUT
# ============================================================
# Supprimer les output par defaut
for key in list(mdl.fieldOutputRequests.keys()):
    if key != 'F-Output-1':
        safe_delete(mdl.fieldOutputRequests, key)
for key in list(mdl.historyOutputRequests.keys()):
    safe_delete(mdl.historyOutputRequests, key)

mdl.FieldOutputRequest(
    name='F-Output-1',
    createStepName='Step-1',
    variables=('S', 'LE', 'PE', 'PEEQ', 'MISES',
               'U', 'V', 'A', 'RF', 'CF', 'STATUS',
               'CSTRESS', 'CFORCE'),
    numIntervals=NUM_OUTPUT_INTERVALS
)

mdl.HistoryOutputRequest(
    name='H-Indenter',
    createStepName='Step-1',
    region=aAsm.sets['RP_INDENTER'],
    variables=('U3', 'V3', 'RF3'),
    numIntervals=NUM_OUTPUT_INTERVALS * 5
)

print("Output requests crees (%d field intervals, %d history intervals)."
      % (NUM_OUTPUT_INTERVALS, NUM_OUTPUT_INTERVALS * 5))

# ============================================================
# 15. CREATION DU JOB (sans lancement)
# ============================================================
safe_delete(mdb.jobs, JOB_NAME)
mdb.Job(
    name=JOB_NAME,
    model=MODEL_NAME,
    description='Percussion dynamique sur echantillon heterogene',
    type=ANALYSIS,
    numCpus=14,
    numDomains=14,
    multiprocessingMode=DEFAULT
)

print("=" * 60)
print("MODELE COMPLET")
print("  Modele : %s" % MODEL_NAME)
print("  Job    : %s" % JOB_NAME)
print("  Box    : %.1f x %.1f x %.1f mm" % (BOX_LX, BOX_LY, BOX_LZ))
print("  Zone raffinee : %.1f x %.1f x %.1f mm (fine=%.2f, coarse=%.2f)"
      % (REFINE_HALF_X*2, REFINE_HALF_Y*2, REFINE_DEPTH, ELEMENT_SIZE_FINE, ELEMENT_SIZE_COARSE))
print("  Indenteur : Rmax=%.3f mm, H=%.3f mm" % (s_start[0], s_start[1]))
print("  V0=%.0f mm/s, F=%.0f N, m=%.4f t" % (IND_VELOCITY, IND_FORCE, IND_MASS))
print("  Confinement : %.0f MPa" % CONFINEMENT)
print("  Solveur : Abaqus/Explicit")
print("  Pour lancer : mdb.jobs['%s'].submit()" % JOB_NAME)
print("=" * 60)
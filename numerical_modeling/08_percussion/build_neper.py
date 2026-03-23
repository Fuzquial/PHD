# -*- coding: utf-8 -*-
# Abaqus/CAE Python script — Percussion dynamique
# =====================================================================
# build_neper.py
# Modele de percussion : heterogeneite par NEPER (maillage polycristallin)
#   - Importe le maillage Neper genere par prep_neper_box.py
#   - Specimen parallelepiped 5x5x10 mm (elements deja dans l'orphan mesh)
#   - Assigne les materiaux CDP aux phases Feldspath/Quartz/Biotite
#   - Meme impacteur et chargement que les autres scripts
#
# Prerequis : lancer prep_neper_box.py (Python standard) pour generer
#             granite_box/granite_box_abaqus.inp
#
# Utilisation : Abaqus/CAE > File > Run Script > build_neper.py
# =====================================================================

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import os

# ─────────────────────────────────────────────────────────────
# PARAMETRES
# ─────────────────────────────────────────────────────────────

MODEL_NAME = 'Percussion-Neper'

# Chemin vers le .inp genere par prep_neper_box.py
INP_FILE = r'C:\Users\fuzquianoalricabi\Documents\python\08_percussion\granite_box\granite_box_abaqus.inp'

# Geometrie (doit correspondre a prep_neper_box.py)
BOX_LX = 5.0
BOX_LY = 5.0
BOX_LZ = 10.0  # mm (axe de chargement)

# Impacteur
IMP_R_SHOULDER = 8.0
IMP_Y_SHOULDER = 11.5
IMP_R_CONE_END = 3.63730670
IMP_Y_CONE_END = 2.1
IMP_SPHERE_R   = 4.2
IMP_MASS       = 0.0069
IMP_MESH_SIZE  = 1.5

# Materiaux CDP
minerals = [
    dict(name='Feldspath', E=48000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=180.0, sigt0=8.0,  dilat=30.0, dt=0.98)),
    dict(name='Quartz',    E=85000.0, nu=0.25, rho=2.2e-9,
         cdp=dict(sigc0=350.0, sigt0=10.0, dilat=35.0, dt=0.98)),
    dict(name='Biotite',   E=30000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=250.0, sigt0=5.0,  dilat=30.0, dt=0.98)),
]

CDP_ECC  = 0.1
CDP_FBFC = 1.16
CDP_K    = 0.554
CDP_VISC = 5e-4

EIN_PIC  = 0.00065
EIN_END  = 0.0094
ETIN_END = 0.00035
R_F1     = 153.7 / 196.4
R_CRES   = 4.27  / 196.4
R_TRES   = 0.034 / 8.78

# Chargement
V_IMPACT    = -2950.0
F_IMPACT    =  2880.0
P_CONF      =  30.0
STRESS_INIT = -30.0

STEP_DURATION = 1.0e-3
DT_MAX        = 1.0e-5

TOL = 1e-3

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def safe_delete(container, key):
    try:
        if key in container.keys():
            del container[key]
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# MODELE
# ─────────────────────────────────────────────────────────────

print("=== Percussion Neper : construction du modele ===")

if not os.path.isfile(INP_FILE):
    raise IOError("Fichier .inp introuvable : {}\n"
                  "Lancer d'abord prep_neper_box.py".format(INP_FILE))

safe_delete(mdb.models, MODEL_NAME)
mdb.ModelFromInputFile(name=MODEL_NAME, inputFileName=INP_FILE)
mdl = mdb.models[MODEL_NAME]

granite_part = mdl.parts['GRANITE']
print("  -> Part GRANITE : {} elements".format(len(granite_part.elements)))

# ── Materiaux CDP ───────────────────────────────────────────

print("Definition des materiaux CDP...")
for m in minerals:
    safe_delete(mdl.materials, m['name'])
    mat = mdl.Material(name=m['name'])
    mat.Density(table=((m['rho'],),))
    mat.Elastic(table=((m['E'], m['nu']),))
    dilat = m['cdp'].get('dilat', 35.0)
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
    dt = m['cdp'].get('dt', 0.98)
    mat.concreteDamagedPlasticity.ConcreteTensionDamage(
        table=((0.0, 0.0), (dt, ETIN_END))
    )

# ── Sections + affectation par phase ───────────────────────

print("Affectation des sections...")
for sname, matname in [('Section-Feldspath', 'Feldspath'),
                       ('Section-Quartz',    'Quartz'),
                       ('Section-Biotite',   'Biotite')]:
    safe_delete(mdl.sections, sname)
    mdl.HomogeneousSolidSection(name=sname, material=matname, thickness=None)

for phase, section in [('FELDSPATH', 'Section-Feldspath'),
                       ('QUARTZ',    'Section-Quartz'),
                       ('BIOTITE',   'Section-Biotite')]:
    elset_name = 'ELSET_' + phase
    if elset_name in granite_part.sets.keys():
        granite_part.SectionAssignment(
            region=granite_part.sets[elset_name],
            sectionName=section
        )
        print("  {} -> {}".format(elset_name, section))
    else:
        print("  AVERTISSEMENT : {} non trouve".format(elset_name))

# ── Impacteur ──────────────────────────────────────────────

print("Creation de l'impacteur...")
safe_delete(mdl.parts, 'IMPACTOR')

sk = mdl.ConstrainedSketch(name='__imp__', sheetSize=50.)
# Profil FERME : epaule -> cone -> sphere -> axe -> haut
sk.Line(point1=(IMP_R_SHOULDER, IMP_Y_SHOULDER),
        point2=(IMP_R_CONE_END,  IMP_Y_CONE_END))
sk.ArcByCenterEnds(
    center=(0.0, IMP_SPHERE_R),
    point1=(IMP_R_CONE_END, IMP_Y_CONE_END),
    point2=(0.0, 0.0),
    direction=CLOCKWISE
)
sk.Line(point1=(0.0, 0.0),            point2=(0.0, IMP_Y_SHOULDER))
sk.Line(point1=(0.0, IMP_Y_SHOULDER), point2=(IMP_R_SHOULDER, IMP_Y_SHOULDER))
imp_part = mdl.Part(name='IMPACTOR', dimensionality=THREE_D,
                    type=DEFORMABLE_BODY)
imp_part.BaseSolidRevolution(sketch=sk, angle=360.)
del mdl.sketches['__imp__']

imp_part.seedPart(size=IMP_MESH_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
imp_part.generateMesh()

rp_feat = imp_part.ReferencePoint(point=(0., IMP_Y_SHOULDER, 0.))
imp_part.Set(name='RP_IMP',
             referencePoints=(imp_part.referencePoints[rp_feat.id],))
imp_part.engineeringFeatures.PointMassInertia(
    name='Mass-Imp', region=imp_part.sets['RP_IMP'],
    mass=IMP_MASS, alpha=0., composite=0.
)

# ── Assemblage ─────────────────────────────────────────────

print("Assemblage...")
aAsm = mdl.rootAssembly
aAsm.DatumCsysByDefault(CARTESIAN)

# L'import du .inp cree deja une instance GRANITE-1 si elle y est definie,
# sinon on la cree ici. On nettoie et on repart proprement.
for iname in list(aAsm.instances.keys()):
    safe_delete(aAsm.instances, iname)

inst_granite = aAsm.Instance(name='GRANITE-1', part=granite_part, dependent=ON)
inst_imp     = aAsm.Instance(name='IMPACTOR-1', part=imp_part,    dependent=ON)

# Centrer le specimen en x,y (Neper genere de 0 a LX et 0 a LY)
aAsm.translate(instanceList=('GRANITE-1',),
               vector=(-BOX_LX / 2., -BOX_LY / 2., 0.))

# Impacteur : rotation +90 deg autour de x -> axe z ; pointe en z=BOX_LZ
aAsm.rotate(instanceList=('IMPACTOR-1',),
            axisPoint=(0., 0., 0.), axisDirection=(1., 0., 0.), angle=90.)
aAsm.translate(instanceList=('IMPACTOR-1',), vector=(0., 0., BOX_LZ))

rp_imp_asm = aAsm.instances['IMPACTOR-1'].referencePoints.values()[0]
aAsm.Set(name='RP_IMP', referencePoints=(rp_imp_asm,))

# Contrainte rigide sur l'impacteur
aAsm.Set(name='SET_IMP_ALL', elements=inst_imp.elements)
mdl.RigidBody(name='RB_IMP',
              refPointRegion=aAsm.sets['RP_IMP'],
              bodyRegion=aAsm.sets['SET_IMP_ALL'])

aAsm.regenerate()

# ── Surfaces ───────────────────────────────────────────────

# Surfaces deja definies dans le .inp (SURF_BOT, SURF_TOP, SURF_X0/X1/Y0/Y1)
# On les recupere depuis l'instance GRANITE-1
surf_names = ('SURF_BOT', 'SURF_TOP', 'SURF_X0', 'SURF_X1', 'SURF_Y0', 'SURF_Y1')
for sname in surf_names:
    if sname in inst_granite.surfaces.keys():
        pass  # deja accessible via inst_granite.surfaces[sname]
    else:
        print("  AVERTISSEMENT : surface {} non trouvee dans l'orphan mesh".format(sname))

# Surface de contact impacteur
aAsm.Surface(name='SURF_IMP', side1Faces=inst_imp.faces)

# Set ALL pour condition initiale stress
if 'ALL' in inst_granite.sets.keys():
    all_region = inst_granite.sets['ALL']
else:
    all_region = regionToolset.Region(elements=inst_granite.elements)

# ── Contact ────────────────────────────────────────────────

safe_delete(mdl.contactProperties, 'IntProp-1')
ip = mdl.ContactProperty('IntProp-1')
ip.TangentialBehavior(formulation=FRICTIONLESS)
ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                  constraintEnforcementMethod=DEFAULT)

# ── Step explicite ─────────────────────────────────────────

print("Creation du step explicite...")
safe_delete(mdl.steps, 'Step-1')
mdl.ExplicitDynamicsStep(
    name='Step-1',
    previous='Initial',
    timePeriod=STEP_DURATION,
    maxIncrement=DT_MAX,
    scaleFactor=1.0,
    improvedDtMethod=ON
)

# ── Contact impact ─────────────────────────────────────────

safe_delete(mdl.interactions, 'Int-Impact')
surf_top_granite = inst_granite.surfaces['SURF_TOP']
mdl.SurfaceToSurfaceContactExp(
    name='Int-Impact',
    createStepName='Step-1',
    main=aAsm.surfaces['SURF_IMP'],
    secondary=surf_top_granite,
    sliding=FINITE,
    interactionProperty='IntProp-1'
)

# ── BCs ────────────────────────────────────────────────────

safe_delete(mdl.boundaryConditions, 'BC_BOT')
mdl.EncastreBC(name='BC_BOT', createStepName='Initial',
               region=inst_granite.surfaces['SURF_BOT'])

safe_delete(mdl.boundaryConditions, 'BC_IMP_GUIDE')
mdl.DisplacementBC(
    name='BC_IMP_GUIDE', createStepName='Initial',
    region=aAsm.sets['RP_IMP'],
    u1=0., u2=0., u3=UNSET, ur1=0., ur2=0., ur3=0.
)

# ── Chargement ─────────────────────────────────────────────

for surf_name in ('SURF_X0', 'SURF_X1', 'SURF_Y0', 'SURF_Y1'):
    if surf_name in inst_granite.surfaces.keys():
        safe_delete(mdl.loads, 'Pressure_' + surf_name)
        mdl.Pressure(
            name='Pressure_' + surf_name,
            createStepName='Step-1',
            region=inst_granite.surfaces[surf_name],
            magnitude=P_CONF
        )

safe_delete(mdl.loads, 'Load_Imp_Force')
mdl.ConcentratedForce(
    name='Load_Imp_Force',
    createStepName='Step-1',
    region=aAsm.sets['RP_IMP'],
    cf3=-F_IMPACT
)

# ── Conditions initiales ───────────────────────────────────

safe_delete(mdl.predefinedFields, 'IC_Velocity')
mdl.Velocity(
    name='IC_Velocity',
    region=aAsm.sets['RP_IMP'],
    field='', distributionType=MAGNITUDE,
    velocity1=0., velocity2=0., velocity3=V_IMPACT,
    omega=0.
)

safe_delete(mdl.predefinedFields, 'IC_Stress')
mdl.StressManagedEQ(
    name='IC_Stress',
    region=all_region,
    stressManagementType=STRESS,
    sigma11=STRESS_INIT, sigma22=STRESS_INIT, sigma33=STRESS_INIT,
    sigma12=0., sigma13=0., sigma23=0.
)

# ── Sorties ────────────────────────────────────────────────

mdl.fieldOutputRequests['F-Output-1'].setValues(
    variables=('S', 'E', 'U', 'V', 'A', 'RF', 'PEEQ', 'PEEQT', 'MISES',
               'EVOL', 'STATUS'),
    numIntervals=100
)

safe_delete(mdl.historyOutputRequests, 'H-IMP')
mdl.HistoryOutputRequest(
    name='H-IMP',
    createStepName='Step-1',
    region=aAsm.sets['RP_IMP'],
    variables=('RF3', 'U3', 'V3', 'A3'),
    numIntervals=1000
)

print("=== Modele Percussion-Neper construit. ===")
print("  V0 = {} mm/s  |  F = {} N".format(V_IMPACT, F_IMPACT))
print("  Confinement = {} MPa  |  Stress init = {} MPa".format(P_CONF, STRESS_INIT))

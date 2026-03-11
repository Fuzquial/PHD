# -*- coding: utf-8 -*-
# Abaqus/CAE Python script  —  Etape 2
# =====================================================================
# build_model_neper.py
# Construit le modele triaxial Abaqus a partir du maillage Neper + CRF
# genere par granite_abaqus_prep.py
#
# Workflow :
#   1. Importe le maillage orphan (.inp) genere par Neper/CRF
#   2. Cree les plateaux rigides
#   3. Assigne les materiaux CDP par phase (Feldspath/Quartz/Biotite)
#   4. Assemble, contacts, BCs, steps de chargement
#   5. Boucle sur les pressions de confinement
# =====================================================================

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import os

# ─────────────────────────────────────────────────────────────
# CHEMIN VERS LE FICHIER .INP GENERE PAR granite_abaqus_prep.py
# ─────────────────────────────────────────────────────────────

# Adapter ce chemin si nécessaire
INP_FILE = r'C:\Users\fuzquianoalricabi\Documents\python\granite_abaqus\granite_cyl_abaqus.inp'

BASE_MODEL_NAME = 'Model-Neper'

# ─────────────────────────────────────────────────────────────
# PARAMETRES GEOMETRIQUES (doivent correspondre au prep script)
# ─────────────────────────────────────────────────────────────

CYL_R    = 2.5
CYL_H    = 10.0
PLATEN_R = 3.0
PLATEN_T = 2.0

ELEMENT_SIZE_PLATEN = 0.6
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

TOP_DISP   = -0.5
STEP2_TIME = 1200.0

PRESSURE_LIST = [5, 10, 30, 50]

# ─────────────────────────────────────────────────────────────
# MATERIAUX CDP
# ─────────────────────────────────────────────────────────────

minerals = [
    dict(name='Feldspath', E=48000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=180.0, sigt0=8.0,  dilat=30.0, dt=0.98)),
    dict(name='Quartz',    E=85000.0, nu=0.25, rho=2.2e-9,
         cdp=dict(sigc0=350.0, sigt0=10.0, dilat=35.0, dt=0.98)),
    dict(name='Biotite',   E=30000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=250.0, sigt0=5.0,  dilat=30.0, dt=0.98)),
]

PLATEN_MAT = dict(name='Material-Platen', E=1.0e10, nu=0.25, rho=6.9e-9)

CDP_ECC  = 0.1
CDP_FBFC = 1.16
CDP_K    = 0.554
CDP_VISC = 5e-4

EIN_PIC  = 0.00065
EIN_END  = 0.0094
ETIN_END = 0.00035

R_F1   = 153.7 / 196.4
R_CRES = 4.27  / 196.4
R_TRES = 0.034 / 8.78

Z_TOL   = 1e-3
XY_TOL  = 1e-2
RAD_TOL = 1e-3


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def safe_delete(container, key):
    try:
        if key in container.keys():
            del container[key]
    except Exception:
        pass


def create_s2s_std(mdl, name, stepName, masterSurf, slaveSurf, propName):
    safe_delete(mdl.interactions, name)
    mdl.SurfaceToSurfaceContactStd(
        name=name, createStepName=stepName,
        main=masterSurf, secondary=slaveSurf,
        sliding=SMALL, thickness=ON,
        interactionProperty=propName,
        adjustMethod=NONE, initialClearance=OMIT
    )


# ─────────────────────────────────────────────────────────────
# ETAPE 1 : Modele de base
# ─────────────────────────────────────────────────────────────

print("=== Construction du modele de base (Neper + CRF) ===")

safe_delete(mdb.models, BASE_MODEL_NAME)
mdl = mdb.Model(name=BASE_MODEL_NAME)

# ── Import du maillage orphan Neper ──
print("Import du maillage orphan : {}".format(INP_FILE))
mdb.ModelFromInputFile(name=BASE_MODEL_NAME, inputFileName=INP_FILE)
mdl = mdb.models[BASE_MODEL_NAME]

# La Part importee s'appelle 'GRANITE' (nom defini dans le .inp)
granite_part = mdl.parts['GRANITE']
print("  -> Part GRANITE importee : {} elements".format(
    len(granite_part.elements)
))

# ── Materiaux ──
print("Definition des materiaux CDP...")

safe_delete(mdl.materials, PLATEN_MAT['name'])
matP = mdl.Material(name=PLATEN_MAT['name'])
matP.Density(table=((PLATEN_MAT['rho'],),))
matP.Elastic(table=((PLATEN_MAT['E'], PLATEN_MAT['nu']),))

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
    tres = max(0.01*ft0, R_TRES*ft0)

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

# ── Sections et assignation par elset ──
print("Assignation des sections par phase mineralogique...")

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

# ── Plateau de chargement ──
print("Creation des plateaux rigides...")

safe_delete(mdl.parts, 'PLATEN')
sk = mdl.ConstrainedSketch(name='__platen__', sheetSize=200.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(PLATEN_R, 0.0))
pl = mdl.Part(name='PLATEN', dimensionality=THREE_D, type=DEFORMABLE_BODY)
pl.BaseSolidExtrude(sketch=sk, depth=PLATEN_T)
del mdl.sketches['__platen__']

safe_delete(mdl.sections, 'Section-Platen')
mdl.HomogeneousSolidSection(name='Section-Platen', material=PLATEN_MAT['name'], thickness=None)
pl.Set(name='SET_PLATEN_ALL', cells=pl.cells)
pl.SectionAssignment(region=pl.sets['SET_PLATEN_ALL'], sectionName='Section-Platen')

elemType_hex = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
elemType_tet = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
pl.seedPart(size=ELEMENT_SIZE_PLATEN, deviationFactor=0.1, minSizeFactor=0.1)
pl.setMeshControls(regions=pl.cells, technique=SWEEP, algorithm=MEDIAL_AXIS)
pl.setElementType(regions=(pl.cells,), elemTypes=(elemType_hex, elemType_tet))
pl.generateMesh()

# ── Assemblage ──
print("Assemblage...")
aAsm = mdl.rootAssembly
aAsm.DatumCsysByDefault(CARTESIAN)

inst_cyl = aAsm.Instance(name='GRANITE-1', part=granite_part, dependent=ON)
inst_bot = aAsm.Instance(name='PLATEN-1',  part=pl,           dependent=ON)
inst_top = aAsm.Instance(name='PLATEN-2',  part=pl,           dependent=ON)

aAsm.translate(instanceList=('PLATEN-1',), vector=(0.0, 0.0, -PLATEN_T))
aAsm.translate(instanceList=('PLATEN-2',), vector=(0.0, 0.0, CYL_H))

rp_top_feat = aAsm.ReferencePoint(point=(0.0, 0.0, CYL_H + PLATEN_T))
rp_bot_feat = aAsm.ReferencePoint(point=(0.0, 0.0, -PLATEN_T))
rp_top = aAsm.referencePoints[rp_top_feat.id]
rp_bot = aAsm.referencePoints[rp_bot_feat.id]

aAsm.Set(name='RP_TOP', referencePoints=(rp_top,))
aAsm.Set(name='RP_BOT', referencePoints=(rp_bot,))

aAsm.regenerate()

aAsm.Set(name='SET_PLATEN_TOP_EL', elements=inst_top.elements)
aAsm.Set(name='SET_PLATEN_BOT_EL', elements=inst_bot.elements)

mdl.RigidBody(name='RB_TOP',
              refPointRegion=aAsm.sets['RP_TOP'],
              bodyRegion=aAsm.sets['SET_PLATEN_TOP_EL'])
mdl.RigidBody(name='RB_BOT',
              refPointRegion=aAsm.sets['RP_BOT'],
              bodyRegion=aAsm.sets['SET_PLATEN_BOT_EL'])

# ── Surfaces ──
f_cyl_top = inst_cyl.faces.getByBoundingBox(
    -CYL_R-XY_TOL, -CYL_R-XY_TOL, CYL_H-Z_TOL,
     CYL_R+XY_TOL,  CYL_R+XY_TOL, CYL_H+Z_TOL
)
f_cyl_bot = inst_cyl.faces.getByBoundingBox(
    -CYL_R-XY_TOL, -CYL_R-XY_TOL, -Z_TOL,
     CYL_R+XY_TOL,  CYL_R+XY_TOL,  Z_TOL
)
f_cyl_lat = inst_cyl.faces.getByBoundingCylinder(
    (0.0, 0.0, 0.0), (0.0, 0.0, CYL_H), CYL_R + RAD_TOL
)

aAsm.Surface(name='SURF_CYL_TOP', side1Faces=f_cyl_top)
aAsm.Surface(name='SURF_CYL_BOT', side1Faces=f_cyl_bot)
aAsm.Surface(name='SURF_CYL_LAT', side1Faces=f_cyl_lat)

f_top_bot = inst_top.faces.getByBoundingBox(
    -PLATEN_R-XY_TOL, -PLATEN_R-XY_TOL, CYL_H-Z_TOL,
     PLATEN_R+XY_TOL,  PLATEN_R+XY_TOL, CYL_H+Z_TOL
)
f_bot_top = inst_bot.faces.getByBoundingBox(
    -PLATEN_R-XY_TOL, -PLATEN_R-XY_TOL, -Z_TOL,
     PLATEN_R+XY_TOL,  PLATEN_R+XY_TOL,  Z_TOL
)

aAsm.Surface(name='SURF_PLATEN_TOP_BOT', side1Faces=f_top_bot)
aAsm.Surface(name='SURF_PLATEN_BOT_TOP', side1Faces=f_bot_top)

# ── Contact ──
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

# ── Steps ──
T2 = float(STEP2_TIME)
mdl.StaticStep(
    name='Step-1', previous='Initial', nlgeom=ON,
    timePeriod=1.0, initialInc=1.0, minInc=1e-5, maxInc=1.0
)
mdl.StaticStep(
    name='Step-2', previous='Step-1', nlgeom=ON,
    timePeriod=T2,
    initialInc=T2/200.0, minInc=T2/1.0e7, maxInc=T2/20.0,
    stabilizationMagnitude=0.002,
    stabilizationMethod=DISSIPATED_ENERGY_FRACTION
)

# ── Historique et sorties ──
mdl.HistoryOutputRequest(
    name='H-RP_TOP', createStepName='Step-1',
    region=aAsm.sets['RP_TOP'], variables=('RF3', 'U3')
)
mdl.historyOutputRequests['H-RP_TOP'].setValuesInStep(
    stepName='Step-2', variables=('RF3', 'U3')
)

# ── Conditions aux limites ──
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
mdl.boundaryConditions['BC_RP_TOP_U3'].setValuesInStep(
    stepName='Step-2', u3=TOP_DISP
)

print("Modele de base Neper construit.")

# ─────────────────────────────────────────────────────────────
# ETAPE 2 : Boucle sur les pressions de confinement
# ─────────────────────────────────────────────────────────────

for PRESSURE_MAG in PRESSURE_LIST:
    MODEL_NAME = 'Neper_%dMPa' % PRESSURE_MAG
    print("--- Modele %s (confinement = %d MPa) ---" % (MODEL_NAME, PRESSURE_MAG))

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

print("Tous les modeles Neper crees. Aucun job lance.")

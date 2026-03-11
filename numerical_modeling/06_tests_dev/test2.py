# -*- coding: utf-8 -*-
# TRIAxial-like: cylinder + 2 rigid platens + frictionless contact + pressure + axial displacement
# + 3-mineral heterogeneity (cosine-wave Gaussian field) at element scale
#
# Fixes:
# - RP_TOP displacement applied like inp: ONE BC updated in Step-2 via setValuesInStep()
# - Anti rigid-body drift: pin one cylinder node U1=U2=0
# - Step-2 stabilization injected with keywordBlock.replace(index, new_line)

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# 1) PARAMETERS
# ============================================================

MODEL_NAME = 'Model-1'

CYL_R = 2.5
CYL_H = 10.0

PLATEN_R = 3.0
PLATEN_T = 2.0

ELEMENT_SIZE_CYL = 0.5
ELEMENT_SIZE_PLATEN = 0.6

PREFER_HEX = True
ELEM_CODE_HEX = C3D8
ELEM_CODE_TET = C3D4

RANDOM_SEED = 42
a = 3.0
Nwaves = 300
MIN1_IS_LOW = True

p_frac = (0.30, 0.50, 0.20)

materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]
PLATEN_MAT = dict(name='PlatenMat', E=2.0e5, nu=0.30, rho=7.8e-9)

PRESSURE_MAG = 5.0
TOP_DISP = -0.5

STEP2_TIMEPERIOD = 0.05
STEP2_INITIALINC = 1.0e-4
STEP2_MININC = 5.0e-8
STEP2_MAXINC = 1.0e-3

STEP2_STABILIZE = 2.0e-4

PIN_TOL = 1.0e-3

# ============================================================
# 2) HELPERS
# ============================================================

def element_volume(part, elem):
    try:
        return float(elem.getSize())
    except Exception:
        xs, ys, zs = [], [], []
        for nl in elem.connectivity:
            n = part.nodes[nl - 1]
            xs.append(n.coordinates[0]); ys.append(n.coordinates[1]); zs.append(n.coordinates[2])
        return (max(xs)-min(xs))*(max(ys)-min(ys))*(max(zs)-min(zs))

def node_xyz(part, node_label):
    n = part.nodes[node_label - 1]
    return n.coordinates[0], n.coordinates[1], n.coordinates[2]

def mesh_solid_part(part, size, prefer_hex=True):
    part.seedPart(size=size, deviationFactor=0.1, minSizeFactor=0.1)
    cells = part.cells[:]
    meshed_as = None

    if prefer_hex:
        try:
            part.setMeshControls(regions=cells, technique=SWEEP, elemShape=HEX)
            et = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
            part.setElementType(regions=(cells,), elemTypes=(et,))
            part.generateMesh()
            meshed_as = 'HEX(SWEEP)'
        except Exception as e:
            print('--- HEX sweep failed, fallback TET. Reason:', str(e))
            meshed_as = None

    if meshed_as is None:
        part.deleteMesh()
        part.seedPart(size=size, deviationFactor=0.1, minSizeFactor=0.1)
        part.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
        et = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
        part.setElementType(regions=(cells,), elemTypes=(et,))
        part.generateMesh()
        meshed_as = 'TET(FREE)'

    return meshed_as

def create_s2s_std(mdl, name, step, mainSurf, secSurf, propName):
    if name in mdl.interactions:
        del mdl.interactions[name]

    try:
        mdl.SurfaceToSurfaceContactStd(
            name=name, createStepName=step,
            main=mainSurf, secondary=secSurf,
            sliding=SMALL, thickness=ON,
            interactionProperty=propName,
            adjustMethod=NONE
        )
        return
    except TypeError:
        pass

    try:
        mdl.SurfaceToSurfaceContactStd(
            name=name, createStepName=step,
            master=mainSurf, slave=secSurf,
            sliding=SMALL, thickness=ON,
            interactionProperty=propName,
            adjustMethod=NONE
        )
        return
    except TypeError:
        pass

    mdl.SurfaceToSurfaceContactStd(
        name=name, createStepName=step,
        masterSurface=mainSurf, slaveSurface=secSurf,
        sliding=SMALL, thickness=ON,
        interactionProperty=propName,
        adjustMethod=NONE
    )

def inject_step2_static_stabilize_replace(mdl, step_name='Step-2', stabilize=2.0e-4):
    """
    Replace the first *Static line inside Step-2 by:
      *Static, stabilize=<stabilize>
    using keywordBlock.replace(index, new_line)
    """
    kb = mdl.keywordBlock
    kb.synchVersions(storeNodesAndElements=False)

    blocks = kb.sieBlocks  # read-only list-like

    step_idx = None
    for i in range(len(blocks)):
        low = blocks[i].strip().lower()
        if low.startswith('*step') and (step_name.lower() in low):
            step_idx = i
            break

    if step_idx is None:
        print('--- WARNING: Step-2 not found in keyword block. No stabilization injected.')
        return

    static_idx = None
    for j in range(step_idx, min(step_idx + 80, len(blocks))):
        if blocks[j].strip().lower().startswith('*static'):
            static_idx = j
            break

    if static_idx is None:
        print('--- WARNING: *Static not found inside Step-2. No stabilization injected.')
        return

    new_line = '*Static, stabilize=%.6g' % stabilize

    # IMPORTANT: this Abaqus expects replace(index:int, new_text:str)
    kb.replace(static_idx, new_line)

    kb.synchVersions(storeNodesAndElements=False)
    print('--- Injected stabilization at keyword line', static_idx, ':', new_line)

# ============================================================
# 3) MODEL SETUP + CLEANUP
# ============================================================

if MODEL_NAME not in mdb.models:
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# Clean parts
for pname in ['CYL', 'PLATEN']:
    if pname in mdl.parts:
        del mdl.parts[pname]

# Clean steps
for s in ['Step-1', 'Step-2']:
    if s in mdl.steps:
        del mdl.steps[s]

# Clean materials / sections
for md in materials + [PLATEN_MAT]:
    if md['name'] in mdl.materials:
        del mdl.materials[md['name']]
for sname in ['SEC_MIN1', 'SEC_MIN2', 'SEC_MIN3', 'SEC_PLATEN']:
    if sname in mdl.sections:
        del mdl.sections[sname]

# Clean interactions/props/constraints/bcs/loads
for ipn in ['IntProp-1']:
    if ipn in mdl.interactionProperties:
        del mdl.interactionProperties[ipn]
for it in ['Int-Top', 'Int-Bot']:
    if it in mdl.interactions:
        del mdl.interactions[it]
for cst in ['RB_TOP', 'RB_BOT']:
    if cst in mdl.constraints:
        del mdl.constraints[cst]
for bc in ['BC_RP_BOT', 'BC_RP_TOP', 'BC_PIN_CYL']:
    if bc in mdl.boundaryConditions:
        del mdl.boundaryConditions[bc]
for ld in ['Load-Pressure']:
    if ld in mdl.loads:
        del mdl.loads[ld]

# ============================================================
# 4) PARTS
# ============================================================

sk = mdl.ConstrainedSketch(name='__cyl__', sheetSize=200.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(CYL_R, 0.0))
pC = mdl.Part(name='CYL', dimensionality=THREE_D, type=DEFORMABLE_BODY)
pC.BaseSolidExtrude(sketch=sk, depth=CYL_H)
del mdl.sketches['__cyl__']

sk = mdl.ConstrainedSketch(name='__pl__', sheetSize=200.0)
sk.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(PLATEN_R, 0.0))
pP = mdl.Part(name='PLATEN', dimensionality=THREE_D, type=DEFORMABLE_BODY)
pP.BaseSolidExtrude(sketch=sk, depth=PLATEN_T)
del mdl.sketches['__pl__']

# ============================================================
# 5) MATERIALS + SECTIONS
# ============================================================

for md in materials:
    mat = mdl.Material(name=md['name'])
    mat.Density(table=((md['rho'],),))
    mat.Elastic(table=((md['E'], md['nu']),))

mdl.HomogeneousSolidSection(name='SEC_MIN1', material='Mineral-1')
mdl.HomogeneousSolidSection(name='SEC_MIN2', material='Mineral-2')
mdl.HomogeneousSolidSection(name='SEC_MIN3', material='Mineral-3')

matP = mdl.Material(name=PLATEN_MAT['name'])
matP.Density(table=((PLATEN_MAT['rho'],),))
matP.Elastic(table=((PLATEN_MAT['E'], PLATEN_MAT['nu']),))
mdl.HomogeneousSolidSection(name='SEC_PLATEN', material=PLATEN_MAT['name'])

# ============================================================
# 6) MESH
# ============================================================

print('--- Meshing CYL...')
mesh_cyl = mesh_solid_part(pC, ELEMENT_SIZE_CYL, prefer_hex=PREFER_HEX)
print('--- CYL meshed as:', mesh_cyl)

print('--- Meshing PLATEN...')
mesh_pl = mesh_solid_part(pP, ELEMENT_SIZE_PLATEN, prefer_hex=PREFER_HEX)
print('--- PLATEN meshed as:', mesh_pl)

# ============================================================
# 7) HETEROGENEITY on CYL
# ============================================================

elems = pC.elements[:]
if len(elems) == 0:
    raise RuntimeError('Cylinder has no elements. Check mesh sizes.')

random.seed(RANDOM_SEED)

sigma = math.sqrt(2.0) / a
waves = []
for i in range(Nwaves):
    waves.append((random.gauss(0.0, sigma),
                  random.gauss(0.0, sigma),
                  random.gauss(0.0, sigma),
                  2.0 * math.pi * random.random()))
coef = math.sqrt(2.0 / float(Nwaves))

def f_point(x, y, z):
    s = 0.0
    for (ox, oy, oz, phi) in waves:
        s += math.cos(ox*x + oy*y + oz*z + phi)
    return coef * s

vols, fvals = [], []
V_tot = 0.0
for e in elems:
    v = element_volume(pC, e)
    vols.append(v); V_tot += v
    fv = 0.0
    nn = float(len(e.connectivity))
    for nl in e.connectivity:
        x, y, z = node_xyz(pC, nl)
        fv += f_point(x, y, z)
    fvals.append(fv / nn)

V_targets = [p_frac[0]*V_tot, p_frac[1]*V_tot, p_frac[2]*V_tot]
idx = list(range(len(elems)))
idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

elsets = [[], [], []]
Vacc = [0.0, 0.0, 0.0]
for i in idx:
    if Vacc[0] < V_targets[0]:
        elsets[0].append(elems[i]); Vacc[0] += vols[i]
    elif Vacc[1] < V_targets[1]:
        elsets[1].append(elems[i]); Vacc[1] += vols[i]
    else:
        elsets[2].append(elems[i]); Vacc[2] += vols[i]

for sname in ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']:
    if sname in pC.sets:
        del pC.sets[sname]

set_names = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']
sec_names = ['SEC_MIN1', 'SEC_MIN2', 'SEC_MIN3']

for k in range(3):
    labels = [e.label for e in elsets[k]]
    if len(labels) == 0:
        raise RuntimeError('Mineral-%d empty. Adjust p_frac/a/Nwaves.' % (k+1))
    ek = pC.elements.sequenceFromLabels(labels=labels)
    pC.Set(name=set_names[k], elements=ek)
    pC.SectionAssignment(region=regionToolset.Region(elements=ek), sectionName=sec_names[k])

pP.SectionAssignment(region=regionToolset.Region(cells=pP.cells[:]), sectionName='SEC_PLATEN')

print('--- Hetero assigned. Volume fractions ~', [Vacc[i]/V_tot for i in range(3)])

# ============================================================
# 8) ASSEMBLY + RPs + rigid bodies + surfaces
# ============================================================

aAsm = mdl.rootAssembly
aAsm.DatumCsysByDefault(CARTESIAN)

for iname in ['CYL-1', 'PLATEN_BOT', 'PLATEN_TOP']:
    if iname in aAsm.instances:
        del aAsm.instances[iname]

instC = aAsm.Instance(name='CYL-1', part=pC, dependent=ON)
instB = aAsm.Instance(name='PLATEN_BOT', part=pP, dependent=ON)
instT = aAsm.Instance(name='PLATEN_TOP', part=pP, dependent=ON)

aAsm.translate(instanceList=('PLATEN_BOT',), vector=(0.0, 0.0, -PLATEN_T))
aAsm.translate(instanceList=('PLATEN_TOP',), vector=(0.0, 0.0, CYL_H))

rp_top = aAsm.ReferencePoint(point=(0.0, 0.0, CYL_H + PLATEN_T))
rp_bot = aAsm.ReferencePoint(point=(0.0, 0.0, -PLATEN_T))

aAsm.Set(name='RP_TOP', referencePoints=(aAsm.referencePoints[rp_top.id],))
aAsm.Set(name='RP_BOT', referencePoints=(aAsm.referencePoints[rp_bot.id],))

aAsm.Set(name='SET_PLATEN_TOP', elements=instT.elements[:])
aAsm.Set(name='SET_PLATEN_BOT', elements=instB.elements[:])

mdl.RigidBody(name='RB_TOP', refPointRegion=aAsm.sets['RP_TOP'], bodyRegion=aAsm.sets['SET_PLATEN_TOP'])
mdl.RigidBody(name='RB_BOT', refPointRegion=aAsm.sets['RP_BOT'], bodyRegion=aAsm.sets['SET_PLATEN_BOT'])

cyl_top = instC.faces.findAt(((0.0, 0.0, CYL_H),))
cyl_bot = instC.faces.findAt(((0.0, 0.0, 0.0),))
cyl_lat = instC.faces.findAt(((CYL_R, 0.0, 0.5*CYL_H),))

aAsm.Surface(name='SURF_CYL_TOP', side1Faces=cyl_top)
aAsm.Surface(name='SURF_CYL_BOT', side1Faces=cyl_bot)
aAsm.Surface(name='SURF_CYL_LAT', side1Faces=cyl_lat)

pl_bot_top = instB.faces.findAt(((0.0, 0.0, 0.0),))
pl_top_bot = instT.faces.findAt(((0.0, 0.0, CYL_H),))

aAsm.Surface(name='SURF_PLATEN_BOT_TOP', side1Faces=pl_bot_top)
aAsm.Surface(name='SURF_PLATEN_TOP_BOT', side1Faces=pl_top_bot)

# ============================================================
# 9) CONTACT
# ============================================================

ip = mdl.ContactProperty('IntProp-1')
ip.TangentialBehavior(formulation=FRICTIONLESS)
ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)

create_s2s_std(mdl, 'Int-Top', 'Initial',
               aAsm.surfaces['SURF_PLATEN_TOP_BOT'],
               aAsm.surfaces['SURF_CYL_TOP'],
               'IntProp-1')

create_s2s_std(mdl, 'Int-Bot', 'Initial',
               aAsm.surfaces['SURF_PLATEN_BOT_TOP'],
               aAsm.surfaces['SURF_CYL_BOT'],
               'IntProp-1')

# ============================================================
# 10) STEPS + STABILIZE injection
# ============================================================

mdl.StaticStep(name='Step-1', previous='Initial', nlgeom=ON,
               timePeriod=1.0, initialInc=1.0, minInc=1e-5, maxInc=1.0)

mdl.StaticStep(name='Step-2', previous='Step-1', nlgeom=ON,
               timePeriod=STEP2_TIMEPERIOD,
               initialInc=STEP2_INITIALINC, minInc=STEP2_MININC, maxInc=STEP2_MAXINC)

inject_step2_static_stabilize_replace(mdl, step_name='Step-2', stabilize=STEP2_STABILIZE)

# ============================================================
# 11) BCs (correct like inp) + anti-drift pin
# ============================================================

mdl.EncastreBC(name='BC_RP_BOT', createStepName='Initial', region=aAsm.sets['RP_BOT'])

bc_top = mdl.DisplacementBC(
    name='BC_RP_TOP', createStepName='Initial', region=aAsm.sets['RP_TOP'],
    u1=0.0, u2=0.0, u3=UNSET,
    ur1=0.0, ur2=0.0, ur3=0.0
)
bc_top.setValuesInStep(stepName='Step-2', u3=TOP_DISP)

pin_nodes = instC.nodes.getByBoundingBox(
    xMin=CYL_R - PIN_TOL, xMax=CYL_R + PIN_TOL,
    yMin=-PIN_TOL, yMax=PIN_TOL,
    zMin=-PIN_TOL, zMax=PIN_TOL
)
if len(pin_nodes) == 0:
    pin_nodes = instC.nodes.getByBoundingBox(zMin=-PIN_TOL, zMax=PIN_TOL)
if len(pin_nodes) == 0:
    raise RuntimeError('Could not find a node to pin. Increase PIN_TOL.')

aAsm.Set(name='PIN_CYL', nodes=pin_nodes[0:1])
mdl.DisplacementBC(name='BC_PIN_CYL', createStepName='Initial',
                   region=aAsm.sets['PIN_CYL'], u1=0.0, u2=0.0, u3=UNSET)

# ============================================================
# 12) LOAD
# ============================================================

mdl.Pressure(name='Load-Pressure', createStepName='Step-1',
             region=aAsm.surfaces['SURF_CYL_LAT'], magnitude=PRESSURE_MAG)

print('--- OK built. Displacement applied like inp, stabilization injected, anti-drift pin set.')
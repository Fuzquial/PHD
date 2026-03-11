# -*- coding: utf-8 -*-
# Abaqus/CAE - Bloc 3D hétérogène + Insert rigid analytic (révolution 360°)
# Insert défini par équation r(z) (cône + option arrondi au bout)
# Pas de lecture de .inp

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math

# ============================================================
# 1) PARAMETRES GEOMETRIE SOL
# ============================================================
MODEL_NAME = 'Model-1'
SOIL_PART  = 'Block3D'
SOIL_INST  = 'Block3D-1'

Lx, Ly, Lz = 10.0, 6.0, 8.0

# Maillage
ELEMENT_SIZE = 0.2
PREFER_HEX = True
ELEM_CODE_HEX = C3D8R
ELEM_CODE_TET = C3D4

# ============================================================
# 2) INSERT (equation)
# ============================================================
INS_PART = 'InsertRigid'
INS_INST = 'InsertRigid-1'
RP_SET_NAME = 'RP_INSERT'
INS_SURF_NAME = 'INSERT_SURF'

# Position initiale : pointe au-dessus de la surface z=Lz
IMPACT_GAP = 2.0  # mm

# Géométrie du cône 
INSERT_HEIGHT = 0.5        # H (mm) hauteur axiale
INSERT_R_TOP  = 0.2         # R (mm) rayon en haut du cône
TIP_FILLET_R  = 0.15        # (mm) arrondi au bout ; mets 0.0 pour une pointe parfaite
N_PROFILE_PTS = 120         # nb de points pour discrétiser le profil

# ============================================================
# 3) HETEROGENEITE (3 phases)
# ============================================================
p_vol = (0.30, 0.50, 0.20)

GRAIN_MEAN_DIAM = 0.8
N_SEEDS_MIN, N_SEEDS_MAX = 300, 1200
RANDOM_SEED = 123
HASH_CELL = max(1e-6, 0.9 * GRAIN_MEAN_DIAM)

# ============================================================
# 4) MATERIAUX (3 minéraux) + CDP
# ============================================================
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),  
]

# CDP global
CDP_DILAT = 35.0
CDP_ECC   = 0.1
CDP_FBFC  = 1.16
CDP_K     = 0.554
CDP_VISC  = 5e-05

# Courbes shape (identiques)
EIN_PIC  = 0.00065
EIN_END  = 0.0094
R_F1     = 153.7 / 196.4
R_CRES   = 4.27  / 196.4

ETIN_END = 0.00035
R_TRES   = 0.034 / 8.78
DT_DAMAGE = 0.98

# Résistances (MPa)
strengths = {
    'Mineral-1': dict(fc0=120.0, ft0=5.0),
    'Mineral-2': dict(fc0=285.0, ft0=11.5),
    # -> 3e minéral (intermédiaire) proposé
    'Mineral-3': dict(fc0=200.0, ft0=8.0),
}

# ============================================================
# 5) OUTILS
# ============================================================

def element_centroid(part, elem):
    xs = ys = zs = 0.0
    nn = float(len(elem.connectivity))
    for nl in elem.connectivity:
        n = part.nodes[nl-1]
        x, y, z = n.coordinates
        xs += x; ys += y; zs += z
    return xs/nn, ys/nn, zs/nn

def safe_counts(n_seeds, p):
    n0 = int(round(p[0]*n_seeds))
    n1 = int(round(p[1]*n_seeds))
    n2 = n_seeds - n0 - n1
    counts = [max(1,n0), max(1,n1), max(1,n2)]
    while sum(counts) > n_seeds:
        i = max(range(3), key=lambda k: counts[k])
        if counts[i] > 1:
            counts[i] -= 1
        else:
            break
    while sum(counts) < n_seeds:
        i = max(range(3), key=lambda k: p[k])
        counts[i] += 1
    return counts

def estimate_n_seeds_block(Lx, Ly, Lz, d):
    V = Lx*Ly*Lz
    Vg = math.pi*d**3/6.0
    return int(1.5*V/max(1e-12, Vg))

def rand_point_in_block(Lx, Ly, Lz):
    return (Lx*random.random(), Ly*random.random(), Lz*random.random())

def hash_key(x,y,z,h):
    return (int(math.floor(x/h)), int(math.floor(y/h)), int(math.floor(z/h)))

def build_seed_hash(seeds,h):
    grid={}
    for (x,y,z,lab) in seeds:
        k = hash_key(x,y,z,h)
        grid.setdefault(k, []).append((x,y,z,lab))
    return grid

def nearest_seed_label(cx,cy,cz,grid,h):
    k0 = hash_key(cx,cy,cz,h)
    best_lab=0; best_d2=1e99
    for rings in (1,2,3):
        found=False
        i0,j0,k0z = k0
        for ii in range(i0-rings, i0+rings+1):
            for jj in range(j0-rings, j0+rings+1):
                for kk in range(k0z-rings, k0z+rings+1):
                    key=(ii,jj,kk)
                    if key not in grid: continue
                    found=True
                    for (sx,sy,sz,lab) in grid[key]:
                        dx=cx-sx; dy=cy-sy; dz=cz-sz
                        d2=dx*dx+dy*dy+dz*dz
                        if d2<best_d2:
                            best_d2=d2; best_lab=lab
        if found: return best_lab
    return best_lab

def build_cdp_tables(fc0, ft0):
    """
    Tables CDP robustes (compat Abaqus) :
    - PAS de ligne (0,0)
    - Strains strictement > 0
    - Stresses strictement > 0
    """

    eps = 1e-8  # petite valeur pour éviter zéro exact

    # Compression hardening: (sigma_c, inelastic_strain)
    # On construit une montée -> pic -> résiduel, sans zéro.
    fc1 = max(eps, fc0 * R_F1)
    fc_res = max(eps, fc0 * R_CRES)

    e1   = max(eps, EIN_PIC * 0.40)
    e_pk = max(eps, EIN_PIC)
    e_mid= max(eps, 0.5 * (EIN_PIC + EIN_END))
    e_end= max(eps, EIN_END)

    # sécurise ordre strict croissant
    if not (e1 < e_pk < e_mid < e_end):
        # fallback simple
        e1, e_pk, e_mid, e_end = eps, 2*eps, 3*eps, 4*eps

    comp = (
        (fc1,   e1),
        (fc0,   e_pk),
        (0.6*fc0, e_mid),
        (fc_res, e_end),
    )

    # Tension stiffening: (sigma_t, cracking_strain)
    ft_res = max(eps, ft0 * R_TRES)

    et1   = max(eps, ETIN_END * 0.35)
    et_end= max(eps, ETIN_END)
    if not (et1 < et_end):
        et1, et_end = eps, 2*eps

    tens = (
        (ft0,    eps),     # stress positif, strain > 0
        (0.6*ft0, et1),
        (ft_res, et_end),
    )

    # Compression damage: (inelastic_strain, dc)  (dc in [0,1))
    dc_pk  = 0.2
    dc_end = min(0.999, DT_DAMAGE)

    dc = (
        (e_pk,  dc_pk),
        (e_end, dc_end),
    )

    # Tension damage: (cracking_strain, dt)
    dt_mid = 0.4
    dt_end = min(0.999, DT_DAMAGE)

    dt = (
        (et1,   dt_mid),
        (et_end, dt_end),
    )

    return comp, tens, dc, dt

def make_cone_profile_points(H, Rtop, r_fillet, npts):
    """
    Retourne une liste (r,z) du profil:
    - Si r_fillet > 0 : petit arrondi au bout (ogive) r = sqrt(2*r_fillet*z - z^2) pour z∈[0,r_fillet]
      puis cône linéaire raccordé à z=r_fillet
    - Sinon : cône parfait r = (Rtop/H) * z
    """
    pts = []

    if r_fillet > 0.0:
        z0 = r_fillet  # longueur axiale de l'arrondi
        # rayon à la fin de l'arrondi
        r0 = math.sqrt(max(0.0, 2.0*r_fillet*z0 - z0*z0))  # = r_fillet

        # pente du cône pour atteindre Rtop à H
        if H <= z0:
            raise RuntimeError("INSERT_HEIGHT doit être > TIP_FILLET_R")
        slope = (Rtop - r0) / (H - z0)

        # partie arrondie (0 -> z0)
        n1 = max(10, int(0.25*npts))
        for i in range(n1):
            z = z0 * (float(i)/float(n1-1))
            r = math.sqrt(max(0.0, 2.0*r_fillet*z - z*z))
            pts.append((r, z))

        # partie conique (z0 -> H)
        n2 = max(10, npts - n1 + 1)
        for i in range(1, n2):
            z = z0 + (H - z0) * (float(i)/float(n2-1))
            r = r0 + slope * (z - z0)
            pts.append((r, z))

    else:
        # cône parfait
        for i in range(npts):
            z = H * (float(i)/float(npts-1))
            r = (Rtop/H) * z
            pts.append((r, z))

    # sécurité: forcer la pointe
    pts[0] = (0.0, 0.0)
    return pts

# ============================================================
# 6) MODELE
# ============================================================
if MODEL_NAME not in mdb.models.keys():
    mdb.Model(name=MODEL_NAME)
mdl = mdb.models[MODEL_NAME]

# ============================================================
# 7) SOL: bloc
# ============================================================
if SOIL_PART in mdl.parts.keys():
    del mdl.parts[SOIL_PART]

sk = mdl.ConstrainedSketch(name='__soil_profile__', sheetSize=10.0*max(Lx, Ly, Lz))
sk.rectangle(point1=(0.0, 0.0), point2=(Lx, Ly))
soil = mdl.Part(name=SOIL_PART, dimensionality=THREE_D, type=DEFORMABLE_BODY)
soil.BaseSolidExtrude(sketch=sk, depth=Lz)
del mdl.sketches['__soil_profile__']

# ============================================================
# 8) INSERT: profil par équation + revolve
# ============================================================
if INS_PART in mdl.parts.keys():
    del mdl.parts[INS_PART]

profile = make_cone_profile_points(INSERT_HEIGHT, INSERT_R_TOP, TIP_FILLET_R, N_PROFILE_PTS)
z_top = max([z for (r,z) in profile])

sk2 = mdl.ConstrainedSketch(name='__insert_profile__', sheetSize=500.0)
sk2.ConstructionLine(point1=(0.0, -1000.0), point2=(0.0, 1000.0))

for i in range(len(profile)-1):
    r1, z1 = profile[i]
    r2, z2 = profile[i+1]
    sk2.Line(point1=(r1, z1), point2=(r2, z2))

insert = mdl.Part(name=INS_PART, dimensionality=THREE_D, type=ANALYTIC_RIGID_SURFACE)
insert.AnalyticRigidSurfRevolve(sketch=sk2)
del mdl.sketches['__insert_profile__']

# RP au sommet
rp = insert.ReferencePoint(point=(0.0, 0.0, z_top))
insert.Set(referencePoints=(insert.referencePoints[rp.id],), name=RP_SET_NAME)
insert.Surface(name=INS_SURF_NAME, side1Faces=insert.faces[:])

print('--- Insert created: H=', INSERT_HEIGHT, ' Rtop=', INSERT_R_TOP, ' fillet=', TIP_FILLET_R)

# ============================================================
# 9) MATERIAUX + SECTIONS (CDP)
# ============================================================
for md in materials:
    if md['name'] in mdl.materials.keys():
        del mdl.materials[md['name']]
for i in range(1, 4):
    sname = 'Section-%d' % i
    if sname in mdl.sections.keys():
        del mdl.sections[sname]

for i, md in enumerate(materials, start=1):
    name = md['name']
    mat = mdl.Material(name=name)
    mat.Density(table=((md['rho'],),))
    mat.Elastic(table=((md['E'], md['nu']),))

    # CDP
    mat.ConcreteDamagedPlasticity(table=((CDP_DILAT, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC),))

    fc0 = strengths[name]['fc0']
    ft0 = strengths[name]['ft0']
    comp, tens, dc, dt = build_cdp_tables(fc0, ft0)

    # Abaqus attend :
    # - CompressionHardening : (stress, inelastic_strain)
    # - TensionStiffening    : (stress, cracking_strain)
    # - Damage tables        : (strain, damage)
    mat.concreteDamagedPlasticity.ConcreteCompressionHardening(table=comp)
    mat.concreteDamagedPlasticity.ConcreteTensionStiffening(table=tens)
    mat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=dc)
    mat.concreteDamagedPlasticity.ConcreteTensionDamage(table=dt)

    mdl.HomogeneousSolidSection(name='Section-%d' % i, material=name)

# ============================================================
# 10) MAILLAGE SOL
# ============================================================
soil.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
cells = soil.cells[:]
meshed_as = None

if PREFER_HEX:
    try:
        soil.setMeshControls(regions=cells, technique=SWEEP, elemShape=HEX)
        et = mesh.ElemType(elemCode=ELEM_CODE_HEX, elemLibrary=STANDARD)
        soil.setElementType(regions=(cells,), elemTypes=(et,))
        soil.generateMesh()
        meshed_as = 'HEX'
    except Exception as e:
        print('--- HEX failed -> TET. Reason:', str(e))
        meshed_as = None

if meshed_as is None:
    soil.deleteMesh()
    soil.seedPart(size=ELEMENT_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    soil.setMeshControls(regions=cells, technique=FREE, elemShape=TET)
    et = mesh.ElemType(elemCode=ELEM_CODE_TET, elemLibrary=STANDARD)
    soil.setElementType(regions=(cells,), elemTypes=(et,))
    soil.generateMesh()
    meshed_as = 'TET'

elems = soil.elements[:]
print('--- Soil meshed as:', meshed_as, ' Nelems=', len(elems))

# ============================================================
# 11) HETEROGENEITE (3 phases)
# ============================================================
random.seed(RANDOM_SEED)
N_SEEDS = estimate_n_seeds_block(Lx, Ly, Lz, GRAIN_MEAN_DIAM)
N_SEEDS = max(N_SEEDS_MIN, min(N_SEEDS_MAX, N_SEEDS))
counts = safe_counts(N_SEEDS, p_vol)
labels = [0]*counts[0] + [1]*counts[1] + [2]*counts[2]
random.shuffle(labels)

seeds=[]
for lab in labels:
    x,y,z = rand_point_in_block(Lx, Ly, Lz)
    seeds.append((x,y,z,lab))
grid = build_seed_hash(seeds, HASH_CELL)

elsets=[[],[],[]]
for e in elems:
    cx,cy,cz = element_centroid(soil,e)
    lab = nearest_seed_label(cx,cy,cz,grid,HASH_CELL)
    elsets[lab].append(e)

names = ['ELSET_MIN1','ELSET_MIN2','ELSET_MIN3']
for n in names:
    if n in soil.sets.keys():
        del soil.sets[n]

for k in range(3):
    lbls=[ee.label for ee in elsets[k]]
    seq = soil.elements.sequenceFromLabels(labels=lbls)
    soil.Set(name=names[k], elements=seq)
    soil.SectionAssignment(region=regionToolset.Region(elements=seq),
                           sectionName='Section-%d'%(k+1))

# ============================================================
# 12) ASSEMBLY + POSITION INSERT
# ============================================================
a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

for nm in (SOIL_INST, INS_INST):
    if nm in a.instances.keys():
        del a.instances[nm]

a.Instance(name=SOIL_INST, part=soil, dependent=ON)
a.Instance(name=INS_INST, part=insert, dependent=ON)

IMPACT_GAP = 2.0         # mm au-dessus de la surface

# Position désirée de la POINTE (tip) en global
TIP_X = 5.0
TIP_Y = 6
TIP_Z = 4.0

a.translate(instanceList=(INS_INST,), vector=(TIP_X, TIP_Y, TIP_Z))
print('--- Insert placed: tip z=', Lz + IMPACT_GAP, ' surface z=', Lz)
print('--- DONE (insert equation + 3 minerals CDP).')
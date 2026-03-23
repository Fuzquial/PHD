# -*- coding: utf-8 -*-
"""
build_gaussian.py  —  Python pur (hors Abaqus)
===============================================
Genere percussion_gaussian.inp : soumission directe ou import CAE.

Specimen : parallelepiped 5x5x10 mm, heterogeneite par seuillage gaussien.
3 phases : Feldspath 30%, Quartz 50%, Biotite 20%  (materiaux CDP).
Impacteur : surface analytique de revolution rigide (ref. Emad).
Convention Emad : axe de chargement = y, face d'impact a y=0.

Remplace l'ancienne version CAE (BaseSolidRevolution echouait).

Usage :
  python build_gaussian.py
  -> genere percussion_gaussian.inp dans le meme dossier

Pour soumettre :
  abaqus job=percussion_gaussian inp=percussion_gaussian.inp cpus=4
"""

import os
import math
import random

# ─────────────────────────────────────────────────────────────
# PARAMETRES
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_INP  = os.path.join(BASE_DIR, 'percussion_gaussian.inp')

# Geometrie specimen (mm) — Convention Emad
BOX_LX    = 5.0
BOX_LY    = 5.0
BOX_LZ    = 10.0   # axe de chargement (y)

# Maillage
MESH_SIZE = 0.3    # mm

# Champ gaussien
RANDOM_SEED = 42
A_CORR      = 10.0   # longueur de correlation (mm)
N_WAVES     = 300
P_FRAC      = (0.30, 0.50, 0.20)   # Feldspath, Quartz, Biotite

# Impacteur
IMP_R_SHOULDER = 8.0
IMP_Y_SHOULDER = 11.5
IMP_R_CONE_END = 3.63730669587198
IMP_Y_CONE_END = 2.09999999996075
IMP_SPHERE_R   = 4.2
IMP_MASS       = 0.0069   # tonne

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
F_IMPACT    = -2880.0
P_CONF      =  30.0
STRESS_INIT = -30.0

# Step Abaqus/Explicit
STEP_DURATION = 1.0e-3
N_INC_MAX     = 500000


# ─────────────────────────────────────────────────────────────
# CHAMP GAUSSIEN
# ─────────────────────────────────────────────────────────────

def build_gaussian_waves(nwaves, seed, a):
    """Superposition de nwaves ondes isotropes de longueur de correlation a."""
    random.seed(seed)
    waves = []
    for _ in range(nwaves):
        kx = random.gauss(0.0, 1.0)
        ky = random.gauss(0.0, 1.0)
        kz = random.gauss(0.0, 1.0)
        norm = math.sqrt(kx*kx + ky*ky + kz*kz) + 1e-12
        kx /= norm; ky /= norm; kz /= norm
        phi = random.random() * 2.0 * math.pi
        waves.append((a*kx, a*ky, a*kz, phi))
    return waves


def field_value(x, y, z, waves):
    s = 0.0
    for (kx, ky, kz, phi) in waves:
        s += math.cos(kx*x + ky*y + kz*z + phi)
    return s / float(len(waves))


def weighted_quantile_thresholds(values, weights, fracs):
    """
    Retourne (t1, t2) tels que :
      fraction ponderee < t1  =  fracs[0]
      fraction ponderee < t2  =  fracs[0] + fracs[1]
    """
    f1 = fracs[0]
    f2 = fracs[0] + fracs[1]
    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total_w = sum(w for _, w in pairs) + 1e-30
    cum = 0.0
    t1 = pairs[-1][0]
    t2 = pairs[-1][0]
    got1 = False
    for (v, w) in pairs:
        cum += w
        p = cum / total_w
        if (not got1) and p >= f1:
            t1 = v
            got1 = True
        if p >= f2:
            t2 = v
            break
    return t1, t2


# ─────────────────────────────────────────────────────────────
# MAILLAGE HEX STRUCTURE
# ─────────────────────────────────────────────────────────────
# Identique a build_homogeneous.py
# Convention Emad : y = axe de chargement, y in [-LZ, 0]

def make_hex_mesh(lx, ly_lat, lz_load, h):
    NX = max(1, int(round(lx      / h)))
    NY = max(1, int(round(lz_load / h)))
    NZ = max(1, int(round(ly_lat  / h)))
    dx = lx      / NX
    dy = lz_load / NY
    dz = ly_lat  / NZ

    nodes = {}
    for iz in range(NZ + 1):
        for iy in range(NY + 1):
            for ix in range(NX + 1):
                nid = 1 + ix + iy*(NX+1) + iz*(NX+1)*(NY+1)
                nodes[nid] = (
                    -lx/2.0      + ix * dx,
                    -lz_load     + iy * dy,
                    -ly_lat/2.0  + iz * dz,
                )

    def nid(ix, iy, iz):
        return 1 + ix + iy*(NX+1) + iz*(NX+1)*(NY+1)

    elements = {}
    surfaces = {
        'BOTTOM': [], 'CONTACT': [],
        'LAT_X0': [], 'LAT_X1': [],
        'LAT_Z0': [], 'LAT_Z1': [],
    }
    # Centroides pour le champ gaussien (calcul direct)
    centroids = {}   # eid -> (x, y, z)

    for iz in range(NZ):
        for iy in range(NY):
            for ix in range(NX):
                eid = 1 + ix + iy*NX + iz*NX*NY
                n1 = nid(ix,   iy,   iz  )
                n2 = nid(ix+1, iy,   iz  )
                n3 = nid(ix+1, iy,   iz+1)
                n4 = nid(ix,   iy,   iz+1)
                n5 = nid(ix,   iy+1, iz  )
                n6 = nid(ix+1, iy+1, iz  )
                n7 = nid(ix+1, iy+1, iz+1)
                n8 = nid(ix,   iy+1, iz+1)
                elements[eid] = [n1, n2, n3, n4, n5, n6, n7, n8]

                # Centroide (centre geometrique de l element)
                centroids[eid] = (
                    -lx/2.0     + (ix + 0.5) * dx,
                    -lz_load    + (iy + 0.5) * dy,
                    -ly_lat/2.0 + (iz + 0.5) * dz,
                )

                if iy == 0:       surfaces['BOTTOM'].append((eid, 'S1'))
                if iy == NY - 1:  surfaces['CONTACT'].append((eid, 'S2'))
                if ix == 0:       surfaces['LAT_X0'].append((eid, 'S6'))
                if ix == NX - 1:  surfaces['LAT_X1'].append((eid, 'S4'))
                if iz == 0:       surfaces['LAT_Z0'].append((eid, 'S3'))
                if iz == NZ - 1:  surfaces['LAT_Z1'].append((eid, 'S5'))

    return nodes, elements, surfaces, centroids, NX, NY, NZ


# ─────────────────────────────────────────────────────────────
# SEUILLAGE GAUSSIEN → PHASES
# ─────────────────────────────────────────────────────────────

def assign_phases(centroids, waves, fracs):
    """
    Calcule le champ gaussien a chaque centroide et seuille en 3 phases.
    Retourne trois listes d'element IDs : (ids_m1, ids_m2, ids_m3).
    """
    fvals = {}
    for eid, (cx, cy, cz) in centroids.items():
        fvals[eid] = field_value(cx, cy, cz, waves)

    sorted_eids = sorted(fvals.keys(), key=lambda e: fvals[e])
    n = len(sorted_eids)

    # Seuillage uniforme (elements de meme taille : poids = 1)
    t1, t2 = weighted_quantile_thresholds(
        [fvals[e] for e in sorted_eids],
        [1.0] * n,
        fracs
    )

    ids_m1, ids_m2, ids_m3 = [], [], []
    for eid in sorted(centroids.keys()):
        v = fvals[eid]
        if v <= t1:
            ids_m1.append(eid)
        elif v <= t2:
            ids_m2.append(eid)
        else:
            ids_m3.append(eid)

    return ids_m1, ids_m2, ids_m3, t1, t2


# ─────────────────────────────────────────────────────────────
# HELPERS D'ECRITURE
# ─────────────────────────────────────────────────────────────

def write_surface_block(f, name, face_list):
    f.write('*Surface, type=ELEMENT, name={}\n'.format(name))
    for eid, slabel in face_list:
        f.write('{}, {}\n'.format(eid, slabel))


def write_elset_ids(f, name, eids):
    f.write('*Elset, elset={}\n'.format(name))
    buf = []
    for eid in eids:
        buf.append(str(eid))
        if len(buf) == 16:
            f.write(', '.join(buf) + ',\n')
            buf = []
    if buf:
        f.write(', '.join(buf) + '\n')


def material_block(f, m):
    fc0   = m['cdp']['sigc0']
    ft0   = m['cdp']['sigt0']
    f1    = R_F1   * fc0
    cres  = R_CRES * fc0
    tres  = max(0.01 * ft0, R_TRES * ft0)
    dt    = m['cdp'].get('dt', 0.98)
    dilat = m['cdp'].get('dilat', 35.0)

    f.write('*Material, name={}\n'.format(m['name']))
    f.write('*Density\n')
    f.write(' {:.3e},\n'.format(m['rho']))
    f.write('*Elastic\n')
    f.write('{:.1f}, {:.2f}\n'.format(m['E'], m['nu']))
    f.write('*Concrete Damaged Plasticity\n')
    f.write('{:.1f}, {:.3f}, {:.4f}, {:.4f}, {:.2e}\n'.format(
        dilat, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC))
    f.write('*Concrete Compression Hardening\n')
    f.write(' {:.4f},      0.\n'.format(f1))
    f.write(' {:.4f}, {:.5f}\n'.format(fc0, EIN_PIC))
    f.write(' {:.4f},  {:.4f}\n'.format(cres, EIN_END))
    f.write('*Concrete Tension Stiffening\n')
    f.write('  {:.4f},      0.\n'.format(ft0))
    f.write(' {:.5f}, {:.5f}\n'.format(tres, ETIN_END))
    f.write('*Concrete Compression Damage\n')
    f.write(' 0., 0.\n')
    f.write('*Concrete Tension Damage\n')
    f.write(' 0., 0.\n')
    f.write(' {:.4f}, {:.5f}\n'.format(dt, ETIN_END))


# ─────────────────────────────────────────────────────────────
# ECRITURE DU FICHIER .INP
# ─────────────────────────────────────────────────────────────

def write_inp(out_path):
    # Maillage
    nodes, elements, surfaces, centroids, NX, NY, NZ = make_hex_mesh(
        BOX_LX, BOX_LY, BOX_LZ, MESH_SIZE
    )
    n_elem = NX * NY * NZ
    print("[INP] Maillage {}x{}x{} = {} elements, {} noeuds".format(
        NX, NY, NZ, n_elem, len(nodes)))

    # Champ gaussien
    print("[INP] Calcul du champ gaussien ({} ondes, seed={}, a={} mm)...".format(
        N_WAVES, RANDOM_SEED, A_CORR))
    waves = build_gaussian_waves(N_WAVES, RANDOM_SEED, A_CORR)
    ids_feld, ids_qtz, ids_bio, t1, t2 = assign_phases(
        centroids, waves, P_FRAC)
    print("[INP] Seuils : t1={:.4f}  t2={:.4f}".format(t1, t2))
    print("[INP] Feldspath: {}  Quartz: {}  Biotite: {}".format(
        len(ids_feld), len(ids_qtz), len(ids_bio)))

    print("[INP] Ecriture de {}...".format(out_path))

    with open(out_path, 'w') as f:

        # ── En-tete ────────────────────────────────────────────
        f.write('*Heading\n')
        f.write('** Percussion dynamique - Granite Seuillage Gaussien (CDP)\n')
        f.write('** Generated by build_gaussian.py\n')
        f.write('** Champ : N_WAVES={}, seed={}, A_CORR={} mm\n'.format(
            N_WAVES, RANDOM_SEED, A_CORR))
        f.write('** Phases : Feldspath={:.0f}%  Quartz={:.0f}%  Biotite={:.0f}%\n'.format(
            P_FRAC[0]*100, P_FRAC[1]*100, P_FRAC[2]*100))
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n** PARTS\n**\n')

        # ── Part GRANITE ───────────────────────────────────────
        f.write('*Part, name=GRANITE\n')

        f.write('*Node\n')
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write('{:8d}, {:14.6f}, {:14.6f}, {:14.6f}\n'.format(nid, x, y, z))

        f.write('*Element, type=C3D8R\n')
        for eid in sorted(elements):
            conn = elements[eid]
            f.write('{}, {}\n'.format(eid, ', '.join(str(n) for n in conn)))

        # Elsets par phase
        write_elset_ids(f, 'ELSET_FELDSPATH', sorted(ids_feld))
        write_elset_ids(f, 'ELSET_QUARTZ',    sorted(ids_qtz))
        write_elset_ids(f, 'ELSET_BIOTITE',   sorted(ids_bio))

        # Elset global pour condition initiale de contrainte
        f.write('*Elset, elset=ALL, generate\n')
        f.write('1, {}, 1\n'.format(n_elem))

        # Surfaces
        for sname, flist in surfaces.items():
            write_surface_block(f, sname, flist)

        # Sections
        f.write('** Sections\n')
        f.write('*Solid Section, elset=ELSET_FELDSPATH, material=Feldspath\n,\n')
        f.write('*Solid Section, elset=ELSET_QUARTZ, material=Quartz\n,\n')
        f.write('*Solid Section, elset=ELSET_BIOTITE, material=Biotite\n,\n')
        f.write('*End Part\n')

        # ── Part impacteur vide ─────────────────────────────────
        f.write('**\n')
        f.write('*Part, name=Part-2\n')
        f.write('*End Part\n')

        # ── Assemblage ─────────────────────────────────────────
        f.write('**\n**\n** ASSEMBLY\n**\n')
        f.write('*Assembly, name=Assembly\n**\n')

        f.write('*Instance, name=GRANITE-1, part=GRANITE\n')
        f.write('*End Instance\n**\n')

        f.write('*Instance, name=Part-2-1, part=Part-2\n')
        f.write('*Node\n')
        f.write('      1,           0.,      {:7.3f},           0.\n'.format(
            IMP_Y_SHOULDER))
        f.write('*Nset, nset=Part-2-1-RefPt_, internal\n')
        f.write('1,\n')
        f.write('*Nset, nset=RP_IMP\n')
        f.write(' 1,\n')
        f.write('*Surface, type=REVOLUTION, name=m_Surf-1\n')
        f.write('START, {:.4f}, {:.4f}\n'.format(IMP_R_SHOULDER, IMP_Y_SHOULDER))
        f.write(' LINE, {:.10f}, {:.10f}\n'.format(IMP_R_CONE_END, IMP_Y_CONE_END))
        f.write(' CIRCL, 0., 0., 0., {:.4f}\n'.format(IMP_SPHERE_R))
        f.write('*Rigid Body, ref node=Part-2-1-RefPt_, analytical surface=m_Surf-1\n')
        f.write('*Element, type=MASS, elset=Set-1_Inertia-1_\n')
        f.write('1, 1\n')
        f.write('*Mass, elset=Set-1_Inertia-1_\n')
        f.write('{},\n'.format(IMP_MASS))
        f.write('*End Instance\n**\n')

        f.write('*Nset, nset=RP_IMP, instance=Part-2-1\n')
        f.write(' 1,\n')
        f.write('*End Assembly\n')

        # ── Materiaux ──────────────────────────────────────────
        f.write('**\n** MATERIALS\n**\n')
        for m in minerals:
            material_block(f, m)

        # ── Proprietes de contact ──────────────────────────────
        f.write('**\n** INTERACTION PROPERTIES\n**\n')
        f.write('*Surface Interaction, name=IntProp-1\n')
        f.write('1.,\n')
        f.write('*Friction\n')
        f.write('0.,\n')

        # ── Conditions aux limites ─────────────────────────────
        f.write('**\n** BOUNDARY CONDITIONS\n**\n')
        f.write('** Name: BC-Bottom  Type: Encastre\n')
        f.write('*Boundary\n')
        f.write('GRANITE-1.BOTTOM, ENCASTRE\n')
        f.write('** Name: BC-IMP-Guide\n')
        f.write('*Boundary\n')
        f.write('RP_IMP, 1, 1\n')
        f.write('RP_IMP, 3, 3\n')
        f.write('RP_IMP, 4, 6\n')

        # ── Conditions initiales ───────────────────────────────
        f.write('**\n** PREDEFINED FIELDS\n**\n')
        f.write('** Vitesse initiale impacteur\n')
        f.write('*Initial Conditions, type=VELOCITY\n')
        f.write('RP_IMP, 1, 0.\n')
        f.write('RP_IMP, 2, {:.4f}\n'.format(V_IMPACT))
        f.write('RP_IMP, 3, 0.\n')
        f.write('** Precontrainte isotrope initiale\n')
        f.write('*Initial Conditions, type=STRESS\n')
        f.write('GRANITE-1.ALL, {s}, {s}, {s}, 0., 0., 0.\n'.format(s=STRESS_INIT))

        # ── Step Abaqus/Explicit ───────────────────────────────
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n** STEP: Step-1\n**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc={}\n'.format(N_INC_MAX))
        f.write('*Dynamic, Explicit\n')
        f.write(', {:.4e}\n'.format(STEP_DURATION))

        f.write('**\n** INTERACTIONS\n**\n')
        f.write('** Impact impacteur / face superieure specimen\n')
        f.write('*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE\n')
        f.write('GRANITE-1.CONTACT, Part-2-1.m_Surf-1\n')

        f.write('**\n** LOADS\n**\n')
        f.write('** Force concentree sur impacteur (axe y)\n')
        f.write('*Cload\n')
        f.write('RP_IMP, 2, {:.4f}\n'.format(F_IMPACT))
        f.write('** Pression de confinement (top + 4 faces laterales, ref. Emad TOP_AND_LAT)\n')
        f.write('*Dsload\n')
        for sname in ('CONTACT', 'LAT_X0', 'LAT_X1', 'LAT_Z0', 'LAT_Z1'):
            f.write('GRANITE-1.{}, P, {:.4f}\n'.format(sname, P_CONF))

        f.write('**\n** OUTPUT REQUESTS\n**\n')
        f.write('*Restart, write, frequency=0\n')
        f.write('**\n** FIELD OUTPUT\n**\n')
        f.write('*Output, field, number interval=100\n')
        f.write('*Node Output\n')
        f.write('A, CF, RF, U, V\n')
        f.write('*Element Output, directions=YES\n')
        f.write('E, EVOL, MISES, PEEQ, PEEQT, S, STATUS\n')
        f.write('**\n** HISTORY OUTPUT\n**\n')
        f.write('*Output, history, frequency=1\n')
        f.write('*Node Output, nset=RP_IMP\n')
        f.write('U2, V2, A2, RF2\n')
        f.write('*End Step\n')

    print("[INP] Termine : {}".format(out_path))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    t0 = time.time()

    print('=' * 60)
    print('  build_gaussian.py  (Python pur, sans Abaqus)')
    print('  Specimen  : {}x{}x{} mm  |  h={} mm'.format(
        BOX_LX, BOX_LY, BOX_LZ, MESH_SIZE))
    print('  Champ : N_WAVES={}, seed={}, A_CORR={} mm'.format(
        N_WAVES, RANDOM_SEED, A_CORR))
    print('  Phases : Feldspath={:.0f}%  Quartz={:.0f}%  Biotite={:.0f}%'.format(
        P_FRAC[0]*100, P_FRAC[1]*100, P_FRAC[2]*100))
    print('=' * 60)

    write_inp(OUT_INP)

    print()
    print('Fichier genere en {:.1f}s'.format(time.time() - t0))
    print('  -> {}'.format(OUT_INP))
    print()
    print('Pour soumettre :')
    print('  abaqus job=percussion_gaussian inp=percussion_gaussian.inp cpus=4')

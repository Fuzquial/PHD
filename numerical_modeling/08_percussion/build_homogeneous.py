# -*- coding: utf-8 -*-
"""
build_homogeneous.py  —  Python pur (hors Abaqus)
===================================================
Genere percussion_homogeneous.inp : soumission directe ou import CAE.

Specimen : parallelepiped 5x5x10 mm, granite homogene (CDP).
Impacteur : surface analytique de revolution rigide (ref. Emad).
Convention Emad : axe de chargement = y, face d'impact a y=0.

Remplace l'ancienne version CAE (BaseSolidRevolution echouait).

Usage :
  python build_homogeneous.py
  -> genere percussion_homogeneous.inp dans le meme dossier

Pour soumettre :
  abaqus job=percussion_homogeneous inp=percussion_homogeneous.inp cpus=4
"""

import os
import math

# ─────────────────────────────────────────────────────────────
# PARAMETRES
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_INP  = os.path.join(BASE_DIR, 'percussion_homogeneous.inp')

# Geometrie specimen (mm)
# Convention Emad : x in [-LX/2,+LX/2], y in [-LZ,0], z in [-LY/2,+LY/2]
BOX_LX    = 5.0    # lateral x
BOX_LY    = 5.0    # lateral z
BOX_LZ    = 10.0   # axe de chargement (y)

# Maillage structure C3D8R
MESH_SIZE = 0.3    # mm

# Impacteur (ref. Emad)
IMP_R_SHOULDER = 8.0
IMP_Y_SHOULDER = 11.5
IMP_R_CONE_END = 3.63730669587198
IMP_Y_CONE_END = 2.09999999996075
IMP_SPHERE_R   = 4.2
IMP_MASS       = 0.0069     # tonne

# Materiau granite homogene
MAT_E   = 52000.0    # MPa
MAT_NU  = 0.25
MAT_RHO = 2.62e-9    # tonne/mm3

# CDP homogene
CDP_SIGC0 = 196.4
CDP_SIGT0 = 8.78
CDP_DILAT = 30.0
CDP_ECC   = 0.1
CDP_FBFC  = 1.16
CDP_K     = 0.667
CDP_VISC  = 5e-4
EIN_PIC   = 0.00065
EIN_END   = 0.0094
ETIN_END  = 0.00035
R_F1      = 153.7 / 196.4
R_CRES    = 4.27  / 196.4
R_TRES    = 0.034 / 8.78

# Chargement
V_IMPACT    = -2950.0    # mm/s (axe y)
F_IMPACT    = -2880.0    # N    (axe y)
P_CONF      =  30.0      # MPa (confinement lateral)
STRESS_INIT = -30.0      # MPa (isotrope)

# Step Abaqus/Explicit
STEP_DURATION = 1.0e-3   # s
N_INC_MAX     = 500000


# ─────────────────────────────────────────────────────────────
# GENERATION DU MAILLAGE HEX STRUCTURE
# ─────────────────────────────────────────────────────────────
# Domaine : x in [-LX/2,+LX/2], y in [-LZ,0], z in [-LY/2,+LY/2]
# Impact face : y=0  |  Bottom face : y=-LZ
#
# Numerotation des noeuds : nid = 1 + ix + iy*(NX+1) + iz*(NX+1)*(NY+1)
#
# Faces C3D8R (avec l'ordering ci-dessous) :
#   n1=node(ix,iy,iz)  n2=node(ix+1,iy,iz)  n3=node(ix+1,iy,iz+1)  n4=node(ix,iy,iz+1)
#   n5=node(ix,iy+1,iz) n6=node(ix+1,iy+1,iz) n7=node(ix+1,iy+1,iz+1) n8=node(ix,iy+1,iz+1)
#   S1 = {1,2,3,4}  = face y- (BOTTOM quand iy=0)
#   S2 = {5,8,7,6}  = face y+ (CONTACT quand iy=NY-1)
#   S3 = {1,5,6,2}  = face z- (LAT_Z0 quand iz=0)
#   S4 = {2,6,7,3}  = face x+ (LAT_X1 quand ix=NX-1)
#   S5 = {3,7,8,4}  = face z+ (LAT_Z1 quand iz=NZ-1)
#   S6 = {4,8,5,1}  = face x- (LAT_X0 quand ix=0)

def make_hex_mesh(lx, ly_lat, lz_load, h):
    """
    Retourne (nodes, elements, surfaces, NX, NY, NZ).
    nodes    : dict {nid: (x, y, z)}
    elements : dict {eid: [n1..n8]}  (C3D8R ordering)
    surfaces : dict {name: [(eid, 'Sx'), ...]}
    """
    NX = max(1, int(round(lx     / h)))
    NY = max(1, int(round(lz_load/ h)))   # y = axe de chargement
    NZ = max(1, int(round(ly_lat / h)))
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
                    -lz_load     + iy * dy,   # y=0 = face de contact
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

                if iy == 0:       surfaces['BOTTOM'].append((eid, 'S1'))
                if iy == NY - 1:  surfaces['CONTACT'].append((eid, 'S2'))
                if ix == 0:       surfaces['LAT_X0'].append((eid, 'S6'))
                if ix == NX - 1:  surfaces['LAT_X1'].append((eid, 'S4'))
                if iz == 0:       surfaces['LAT_Z0'].append((eid, 'S3'))
                if iz == NZ - 1:  surfaces['LAT_Z1'].append((eid, 'S5'))

    return nodes, elements, surfaces, NX, NY, NZ


# ─────────────────────────────────────────────────────────────
# HELPERS D'ECRITURE
# ─────────────────────────────────────────────────────────────

def write_surface_block(f, name, face_list):
    f.write('*Surface, type=ELEMENT, name={}\n'.format(name))
    for eid, slabel in face_list:
        f.write('{}, {}\n'.format(eid, slabel))


# ─────────────────────────────────────────────────────────────
# ECRITURE DU FICHIER .INP
# ─────────────────────────────────────────────────────────────

def write_inp(out_path):
    nodes, elements, surfaces, NX, NY, NZ = make_hex_mesh(
        BOX_LX, BOX_LY, BOX_LZ, MESH_SIZE
    )
    n_elem = NX * NY * NZ

    fc0  = CDP_SIGC0
    ft0  = CDP_SIGT0
    f1   = R_F1   * fc0
    cres = R_CRES * fc0
    tres = max(0.01 * ft0, R_TRES * ft0)

    print("[INP] Maillage {}x{}x{} = {} elements C3D8R, {} noeuds".format(
        NX, NY, NZ, n_elem, len(nodes)))
    print("[INP] Ecriture de {}...".format(out_path))

    with open(out_path, 'w') as f:

        # ── En-tete ────────────────────────────────────────────
        f.write('*Heading\n')
        f.write('** Percussion dynamique - Granite Homogene (CDP)\n')
        f.write('** Generated by build_homogeneous.py\n')
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

        # Elset global (generate pour compacite)
        f.write('*Elset, elset=ALL, generate\n')
        f.write('1, {}, 1\n'.format(n_elem))

        # Surfaces
        for sname, flist in surfaces.items():
            write_surface_block(f, sname, flist)

        # Section
        f.write('** Section: Granite Homogene\n')
        f.write('*Solid Section, elset=ALL, material=Granite-Hom\n')
        f.write(',\n')
        f.write('*End Part\n')

        # ── Part impacteur (vide, geometrie en ligne dans l instance) ──
        f.write('**\n')
        f.write('*Part, name=Part-2\n')
        f.write('*End Part\n')

        # ── Assemblage ─────────────────────────────────────────
        f.write('**\n**\n** ASSEMBLY\n**\n')
        f.write('*Assembly, name=Assembly\n**\n')

        f.write('*Instance, name=GRANITE-1, part=GRANITE\n')
        f.write('*End Instance\n**\n')

        # Instance impacteur : surface analytique de revolution rigide
        # Pointe a y=0 (contact face du specimen), epaule a y=IMP_Y_SHOULDER
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

        # Nset assemblage pour RP impacteur
        f.write('*Nset, nset=RP_IMP, instance=Part-2-1\n')
        f.write(' 1,\n')

        f.write('*End Assembly\n')

        # ── Materiaux ──────────────────────────────────────────
        f.write('**\n** MATERIALS\n**\n')
        f.write('*Material, name=Granite-Hom\n')
        f.write('*Density\n')
        f.write(' {:.3e},\n'.format(MAT_RHO))
        f.write('*Elastic\n')
        f.write('{:.1f}, {:.2f}\n'.format(MAT_E, MAT_NU))
        f.write('*Concrete Damaged Plasticity\n')
        f.write('{:.1f}, {:.3f}, {:.4f}, {:.4f}, {:.2e}\n'.format(
            CDP_DILAT, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC))
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
        f.write(' 0.9800, {:.5f}\n'.format(ETIN_END))

        # ── Proprietes de contact ──────────────────────────────
        f.write('**\n** INTERACTION PROPERTIES\n**\n')
        f.write('*Surface Interaction, name=IntProp-1\n')
        f.write('1.,\n')
        f.write('*Friction\n')
        f.write('0.,\n')

        # ── Conditions aux limites (Initial step) ──────────────
        f.write('**\n** BOUNDARY CONDITIONS\n**\n')
        f.write('** Name: BC-Bottom  Type: Encastre\n')
        f.write('*Boundary\n')
        f.write('GRANITE-1.BOTTOM, ENCASTRE\n')
        f.write('** Name: BC-IMP-Guide  (libre en y uniquement)\n')
        f.write('*Boundary\n')
        f.write('RP_IMP, 1, 1\n')   # u1 = 0
        f.write('RP_IMP, 3, 3\n')   # u3 = 0
        f.write('RP_IMP, 4, 6\n')   # rotations bloquees

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

        # Contact
        f.write('**\n** INTERACTIONS\n**\n')
        f.write('** Impact impacteur / face superieure specimen\n')
        f.write('*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE\n')
        f.write('GRANITE-1.CONTACT, Part-2-1.m_Surf-1\n')

        # Chargement
        f.write('**\n** LOADS\n**\n')
        f.write('** Force concentree sur impacteur (axe y)\n')
        f.write('*Cload\n')
        f.write('RP_IMP, 2, {:.4f}\n'.format(F_IMPACT))
        f.write('** Pression de confinement (top + 4 faces laterales, ref. Emad TOP_AND_LAT)\n')
        f.write('*Dsload\n')
        for sname in ('CONTACT', 'LAT_X0', 'LAT_X1', 'LAT_Z0', 'LAT_Z1'):
            f.write('GRANITE-1.{}, P, {:.4f}\n'.format(sname, P_CONF))

        # Sorties
        f.write('**\n** OUTPUT REQUESTS\n**\n')
        f.write('*Restart, write, frequency=0\n')
        f.write('**\n** FIELD OUTPUT: toutes les 100 images\n**\n')
        f.write('*Output, field, number interval=100\n')
        f.write('*Node Output\n')
        f.write('A, CF, RF, U, V\n')
        f.write('*Element Output, directions=YES\n')
        f.write('E, EVOL, MISES, PEEQ, PEEQT, S, STATUS\n')
        f.write('**\n** HISTORY OUTPUT: impacteur\n**\n')
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
    print('  build_homogeneous.py  (Python pur, sans Abaqus)')
    print('  Specimen  : {}x{}x{} mm  |  h={} mm'.format(
        BOX_LX, BOX_LY, BOX_LZ, MESH_SIZE))
    print('  V0={} mm/s  |  F={} N'.format(V_IMPACT, F_IMPACT))
    print('  Pconf={} MPa  |  Sinit={} MPa'.format(P_CONF, STRESS_INIT))
    print('=' * 60)

    write_inp(OUT_INP)

    print()
    print('Fichier genere en {:.1f}s'.format(time.time() - t0))
    print('  -> {}'.format(OUT_INP))
    print()
    print('Pour soumettre :')
    print('  abaqus job=percussion_homogeneous inp=percussion_homogeneous.inp cpus=4')
    print('  OU : Abaqus/CAE > File > Import > Model')

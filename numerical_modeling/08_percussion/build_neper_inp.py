# -*- coding: utf-8 -*-
"""
build_neper_inp.py  —  Python pur (hors Abaqus)
=================================================
Genere percussion_neper.inp en combinant :
  - Le maillage granite Neper (granite_box/granite_box_abaqus.inp)
  - L'impacteur rigide de revolution (ref. Emad)
  - Conditions aux limites, chargement, step explicite

Ce script remplace build_neper.py (CAE) qui echouait sur BaseSolidRevolution.
Lancer prep_neper_box.py d'abord pour generer le maillage Neper.

Convention Emad :
  Neper :  x[0,LX]   y[0,LY]   z[0,LZ]   (z = axe de chargement Neper)
  Emad  :  x[-LX/2,+LX/2]   y[-LZ,0]   z[-LY/2,+LY/2]
  => x_new = x_neper - LX/2
     y_new = z_neper - LZ    (face sup Neper z=LZ -> y=0 = impact)
     z_new = y_neper - LY/2

Usage :
  python build_neper_inp.py
  -> genere percussion_neper.inp dans le meme dossier

Pour soumettre :
  abaqus job=percussion_neper inp=percussion_neper.inp cpus=4
"""

import os
import re

# ─────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
NEPER_INP = os.path.join(BASE_DIR, 'granite_box', 'granite_box_abaqus.inp')
OUT_INP   = os.path.join(BASE_DIR, 'percussion_neper.inp')

# ─────────────────────────────────────────────────────────────
# GEOMETRIE (doit correspondre a prep_neper_box.py)
# ─────────────────────────────────────────────────────────────

BOX_LX = 5.0
BOX_LY = 5.0
BOX_LZ = 10.0

# ─────────────────────────────────────────────────────────────
# PARAMETRES IMPACTEUR (ref. Emad)
# ─────────────────────────────────────────────────────────────

IMP_R_SHOULDER = 8.0
IMP_Y_SHOULDER = 11.5
IMP_R_CONE_END = 3.63730669587198
IMP_Y_CONE_END = 2.09999999996075
IMP_SPHERE_R   = 4.2
IMP_MASS       = 0.0069   # tonne

# ─────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────

V_IMPACT    = -2950.0   # mm/s (axe y)
F_IMPACT    = -2880.0   # N    (axe y)
P_CONF      =  30.0     # MPa
STRESS_INIT = -30.0     # MPa

STEP_DURATION = 1.0e-3
N_INC_MAX     = 500000

# ─────────────────────────────────────────────────────────────
# MATERIAUX CDP (Feldspath / Quartz / Biotite)
# ─────────────────────────────────────────────────────────────

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

minerals = [
    dict(name='Feldspath', E=48000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=180.0, sigt0=8.0,  dilat=30.0, dt=0.98)),
    dict(name='Quartz',    E=85000.0, nu=0.25, rho=2.2e-9,
         cdp=dict(sigc0=350.0, sigt0=10.0, dilat=35.0, dt=0.98)),
    dict(name='Biotite',   E=30000.0, nu=0.25, rho=2.1e-9,
         cdp=dict(sigc0=250.0, sigt0=5.0,  dilat=30.0, dt=0.98)),
]


# ─────────────────────────────────────────────────────────────
# 1. LECTURE DU MAILLAGE NEPER
# ─────────────────────────────────────────────────────────────

def parse_neper_inp(filepath):
    """
    Lit granite_box_abaqus.inp et retourne :
      nodes    : {id: (x,y,z)} coords transformees (convention Emad)
      elements : {id: [n1,...]}
      elem_type: str
      elsets   : {name: [label_or_id, ...]}
      surfaces : {name: [(eid, face_label), ...]}
    """
    nodes     = {}
    elements  = {}
    elem_type = 'C3D8'
    elsets    = {}
    surfaces  = {}

    current_block = None
    current_name  = None

    with open(filepath, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                low = line.lower()
                if low.startswith('*node'):
                    current_block = 'node'
                elif low.startswith('*element'):
                    m = re.search(r'type\s*=\s*(\S+)', line, re.IGNORECASE)
                    if m:
                        elem_type = m.group(1).rstrip(',')
                    current_block = 'element'
                elif low.startswith('*elset'):
                    m = re.search(r'elset\s*=\s*([^,\s]+)', line, re.IGNORECASE)
                    current_name = m.group(1) if m else 'UNKNOWN'
                    elsets.setdefault(current_name, [])
                    current_block = 'elset'
                elif low.startswith('*surface'):
                    m = re.search(r'name\s*=\s*([^,\s]+)', line, re.IGNORECASE)
                    current_name = m.group(1) if m else 'SURF'
                    surfaces.setdefault(current_name, [])
                    current_block = 'surface'
                elif low.startswith('*end part'):
                    current_block = None
                else:
                    current_block = None
                continue

            if current_block == 'node':
                parts = [p.strip() for p in line.split(',')]
                nid = int(parts[0])
                xn, yn, zn = float(parts[1]), float(parts[2]), float(parts[3])
                # Transformation Neper -> Emad
                x_new = xn - BOX_LX / 2.0
                y_new = zn - BOX_LZ           # z=LZ -> y=0 (impact face)
                z_new = yn - BOX_LY / 2.0
                nodes[nid] = (x_new, y_new, z_new)

            elif current_block == 'element':
                parts = [p.strip() for p in line.split(',')]
                eid  = int(parts[0])
                conn = [int(p) for p in parts[1:] if p]
                elements[eid] = conn

            elif current_block == 'elset':
                for p in line.split(','):
                    p = p.strip()
                    if p:
                        elsets[current_name].append(p)

            elif current_block == 'surface':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    surfaces[current_name].append((parts[0], parts[1]))

    return nodes, elements, elem_type, elsets, surfaces


# ─────────────────────────────────────────────────────────────
# 2. BLOC MATERIAU CDP
# ─────────────────────────────────────────────────────────────

def write_material(f, m):
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
# 3. ECRITURE DU FICHIER .INP
# ─────────────────────────────────────────────────────────────

def write_elset_block(f, name, elems):
    f.write('*Elset, elset={}\n'.format(name))
    buf = []
    for e in elems:
        buf.append(str(e))
        if len(buf) == 16:
            f.write(', '.join(buf) + ',\n')
            buf = []
    if buf:
        f.write(', '.join(buf) + '\n')


def write_inp(out_path, nodes, elements, elem_type, elsets, surfaces):
    print("[INP] Ecriture de {}...".format(out_path))

    # Renommage surfaces : convention Neper -> convention Emad
    surf_rename = {
        'SURF_TOP': 'CONTACT',
        'SURF_BOT': 'BOTTOM',
        'SURF_X0':  'LAT_X0',
        'SURF_X1':  'LAT_X1',
        'SURF_Y0':  'LAT_Z0',
        'SURF_Y1':  'LAT_Z1',
    }

    with open(out_path, 'w') as f:

        # ── En-tete ────────────────────────────────────────────
        f.write('*Heading\n')
        f.write('** Percussion dynamique - Granite Neper+CRF (CDP)\n')
        f.write('** Generated by build_neper_inp.py\n')
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n** PARTS\n**\n')

        # ── Part GRANITE ───────────────────────────────────────
        f.write('*Part, name=GRANITE\n')

        f.write('*Node\n')
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write('{:8d}, {:14.6f}, {:14.6f}, {:14.6f}\n'.format(nid, x, y, z))

        f.write('*Element, type={}\n'.format(elem_type))
        for eid in sorted(elements):
            conn = elements[eid]
            f.write('{}, {}\n'.format(eid, ', '.join(str(n) for n in conn)))

        # Elsets phases
        for ename, elems in elsets.items():
            write_elset_block(f, ename, elems)

        # Surfaces (apres transformation des coords, les labels de face restent valides)
        for sname, face_list in surfaces.items():
            out_name = surf_rename.get(sname, sname)
            f.write('*Surface, type=ELEMENT, name={}\n'.format(out_name))
            for eid, slabel in face_list:
                f.write('{}, {}\n'.format(eid, slabel))

        # Sections par phase
        for phase, matname in [('ELSET_FELDSPATH', 'Feldspath'),
                                ('ELSET_QUARTZ',    'Quartz'),
                                ('ELSET_BIOTITE',   'Biotite')]:
            if phase in elsets:
                f.write('** Section: {}\n'.format(phase))
                f.write('*Solid Section, elset={}, material={}\n'.format(phase, matname))
                f.write(',\n')

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

        # Instance impacteur avec surface analytique de revolution
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
            write_material(f, m)

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
        f.write('** Precontrainte isotrope\n')
        f.write('*Initial Conditions, type=STRESS\n')
        for phase in ('ELSET_FELDSPATH', 'ELSET_QUARTZ', 'ELSET_BIOTITE'):
            if phase in elsets:
                f.write('GRANITE-1.{}, {s}, {s}, {s}, 0., 0., 0.\n'.format(
                    phase, s=STRESS_INIT))

        # ── Step Abaqus/Explicit ───────────────────────────────
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n** STEP: Step-1\n**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc={}\n'.format(N_INC_MAX))
        f.write('*Dynamic, Explicit\n')
        f.write(', {:.4e}\n'.format(STEP_DURATION))

        f.write('**\n** INTERACTIONS\n**\n')
        f.write('** Impact\n')
        f.write('*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE\n')
        f.write('GRANITE-1.CONTACT, Part-2-1.m_Surf-1\n')

        f.write('**\n** LOADS\n**\n')
        f.write('** Force concentree sur impacteur\n')
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
    print('  build_neper_inp.py  (Python pur, sans Abaqus)')
    print('  Specimen  : {}x{}x{} mm'.format(BOX_LX, BOX_LY, BOX_LZ))
    print('  V0={} mm/s  |  F={} N'.format(V_IMPACT, F_IMPACT))
    print('  Pconf={} MPa  |  Sinit={} MPa'.format(P_CONF, STRESS_INIT))
    print('=' * 60)

    if not os.path.isfile(NEPER_INP):
        raise IOError(
            "Fichier .inp Neper introuvable : {}\n"
            "Lancer d'abord : python prep_neper_box.py".format(NEPER_INP)
        )

    print("\n[1/2] Lecture du maillage Neper : {}".format(NEPER_INP))
    nodes, elements, elem_type, elsets, surfaces = parse_neper_inp(NEPER_INP)
    print("  -> {} noeuds, {} elements ({})".format(
        len(nodes), len(elements), elem_type))
    print("  -> Elsets  : {}".format(list(elsets.keys())))
    print("  -> Surfaces: {}".format(list(surfaces.keys())))

    print("\n[2/2] Generation du fichier .inp...")
    write_inp(OUT_INP, nodes, elements, elem_type, elsets, surfaces)

    print()
    print('Fichier genere en {:.1f}s'.format(time.time() - t0))
    print('  -> {}'.format(OUT_INP))
    print()
    print('Pour soumettre :')
    print('  abaqus job=percussion_neper inp=percussion_neper.inp cpus=4')
    print('  OU : Abaqus/CAE > File > Import > Model')
    print('=' * 60)

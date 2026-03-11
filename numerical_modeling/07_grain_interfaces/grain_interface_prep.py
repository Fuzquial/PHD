# -*- coding: utf-8 -*-
"""
grain_interface_prep.py  —  Preprocessing avec joints de grains différenciés
=============================================================================
1. Charge le maillage Neper (depuis 02_CRF/granite_abaqus/ ou en génère un)
2. Assigne les phases minérales aux grains via CRF
3. Duplique les noeuds aux joints de grains
4. Insère des éléments cohésifs COH3D6 à chaque interface grain-grain
5. Identifie le TYPE de contact pour chaque COH3D6 (ex. Q-Feld, Q-Bi, ...)
6. Exporte un .inp Abaqus avec :
      - Éléments bulk C3D4 par phase (ELSET_FELDSPATH, ELSET_QUARTZ, ELSET_BIOTITE)
      - 6 ELSETS de COH3D6 par type de contact minéral :
          IF_QRTZ_QRTZ, IF_FELS_FELS, IF_BIOT_BIOT,
          IF_FELS_QRTZ, IF_BIOT_QRTZ, IF_BIOT_FELS

Références bibliographiques des paramètres (voir build_model_interfaces.py) :
  Pan et al. 2023 (Computers & Geotechnics) — GBM Barre granite UDEC
  Liu et al. 2023 (Rock Mech Rock Eng)     — nanoindentation grain boundaries
  Nasseri & Young 2009 (IJRM)              — K_Ic Barre granite 0.71–1.89 MPa·m⁰·⁵

Prérequis : pip install meshio numpy
            Neper dans WSL2 (si maillage non présent)
"""

import os
import json
import sys
import time
import subprocess
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR  = os.path.join(THIS_DIR, "granite_interfaces")
BASENAME = "granite_cyl"

# Réutilise le maillage généré par granite_abaqus_prep.py si présent
NEPER_MESH_FILE = os.path.join(
    THIS_DIR, "..", "02_CRF", "granite_abaqus", BASENAME + "_mesh.msh"
)

# Géométrie du cylindre
CYL_R = 2.5    # mm
CYL_H = 10.0   # mm

# Neper (si le maillage n'existe pas)
N_GRAINS    = 200
MESH_CL     = 0.15
RANDOM_SEED = 42

# CRF
LAMBDA_RF = 0.8
SIGMA_RF  = 0.35
MU_RF     = 1.0

# Fractions volumiques (Feldspath, Quartz, Biotite)
P_FRAC      = (0.30, 0.50, 0.20)
PHASE_NAMES = ['Feldspath', 'Quartz', 'Biotite']

# Abréviations pour nommage des interfaces (ordre alphabétique pour la clé)
PHASE_ABBR = {'Feldspath': 'FELS', 'Quartz': 'QRTZ', 'Biotite': 'BIOT'}

# Liste ordonnée des 6 types d'interface possibles
INTERFACE_TYPES = [
    'IF_BIOT_BIOT',
    'IF_BIOT_FELS',
    'IF_BIOT_QRTZ',
    'IF_FELS_FELS',
    'IF_FELS_QRTZ',
    'IF_QRTZ_QRTZ',
]


def interface_key(phase_A, phase_B):
    """Retourne la clé d'interface canonique (ordre alphabétique des abréviations)."""
    a0 = PHASE_ABBR[phase_A]
    a1 = PHASE_ABBR[phase_B]
    if a0 > a1:
        a0, a1 = a1, a0
    return 'IF_{}_{}'.format(a0, a1)


# ─────────────────────────────────────────────────────────────
# UTILITAIRES WSL / NEPER
# ─────────────────────────────────────────────────────────────

def win_to_wsl(path):
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path  = "/mnt/" + drive + path[2:]
    return path


def run_neper(workdir, basename, n_grains, mesh_cl, cyl_r, cyl_h, seed):
    os.makedirs(workdir, exist_ok=True)
    tess_file = os.path.join(workdir, basename + ".tess")
    mesh_file = os.path.join(workdir, basename + "_mesh.msh")

    wsl_dir  = win_to_wsl(workdir)
    wsl_tess = win_to_wsl(tess_file)
    wsl_out  = wsl_dir + "/" + basename
    wsl_mout = wsl_dir + "/" + basename + "_mesh"

    if not os.path.exists(tess_file):
        print("[Neper 1/2] Tessellation ({} grains)...".format(n_grains))
        diam = 2.0 * cyl_r
        cmd = ("neper -T -n {n} -id {seed} "
               "-domain 'cylinder({h},{d})' "
               "-morpho graingrowth "
               "-morpho_optiini_iter 10000 "
               "-o '{o}'"
               ).format(n=n_grains, seed=seed, h=cyl_h, d=diam, o=wsl_out)
        proc = subprocess.Popen(["wsl", "bash", "-c", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            if line.strip():
                print("  " + line.rstrip()); sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Neper -T echoue (code {})".format(proc.returncode))
        print("  -> tessellation generee")
    else:
        print("[Neper 1/2] Tessellation deja presente.")

    if not os.path.exists(mesh_file):
        print("[Neper 2/2] Maillage (cl={})...".format(mesh_cl))
        cmd = "neper -M '{t}' -cl {cl} -format msh -o '{o}'".format(
              t=wsl_tess, cl=mesh_cl, o=wsl_mout)
        proc = subprocess.Popen(["wsl", "bash", "-c", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            if line.strip():
                print("  " + line.rstrip()); sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Neper -M echoue (code {})".format(proc.returncode))
        print("  -> maillage genere")
    else:
        print("[Neper 2/2] Maillage deja present.")

    return tess_file, mesh_file


# ─────────────────────────────────────────────────────────────
# LECTURE DU MAILLAGE
# ─────────────────────────────────────────────────────────────

def load_mesh(mesh_file):
    try:
        import meshio
    except ImportError:
        raise ImportError("Installe meshio : pip install meshio")

    print("[Mesh] Chargement de {}...".format(mesh_file))
    mesh = meshio.read(mesh_file)

    TETRA_TYPES = ("tetra", "tetra4", "tetra10")
    TAG_KEYS    = ("gmsh:physical", "medit:ref", "cell_tags")
    tag_key     = next((k for k in TAG_KEYS if k in mesh.cell_data), None)

    tetras = grain_ids = None
    for i, blk in enumerate(mesh.cells):
        if blk.type in TETRA_TYPES:
            tetras = blk.data
            if tag_key and i < len(mesh.cell_data[tag_key]):
                grain_ids = mesh.cell_data[tag_key][i]
            break

    if tetras is None:
        raise ValueError("Aucun tetraedre trouve dans le maillage.")
    if grain_ids is None:
        grain_ids = np.ones(len(tetras), dtype=int)

    nodes = mesh.points
    print("  -> {} noeuds, {} tets, {} grains".format(
        len(nodes), len(tetras), len(np.unique(grain_ids))))
    return nodes, tetras.astype(np.int64), grain_ids.astype(np.int64)


# ─────────────────────────────────────────────────────────────
# CRF LOG-NORMAL (Gutjahr 1989)
# ─────────────────────────────────────────────────────────────

def generate_crf(nx, ny, nz, lambda_rf, sigma_rf, mu_rf,
                 x_range, y_range, z_range, seed=42):
    np.random.seed(seed)
    Lx = x_range[1] - x_range[0]
    Ly = y_range[1] - y_range[0]
    Lz = z_range[1] - z_range[0]
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz

    kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2*np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    S      = (lambda_rf**3 * np.pi**1.5) * np.exp(-K2 * lambda_rf**2 / 4.0)
    noise  = (np.random.randn(nx,ny,nz) + 1j*np.random.randn(nx,ny,nz)) / np.sqrt(2)
    fg     = np.real(np.fft.ifftn(np.sqrt(S) * noise))
    fg     = (fg - fg.mean()) / (fg.std() + 1e-12)

    sigma_ln = np.sqrt(np.log(1 + (sigma_rf/mu_rf)**2))
    mu_ln    = np.log(mu_rf) - 0.5*sigma_ln**2
    field_ln = np.exp(mu_ln + sigma_ln * fg)

    xs = np.linspace(x_range[0] + 0.5*dx, x_range[1] - 0.5*dx, nx)
    ys = np.linspace(y_range[0] + 0.5*dy, y_range[1] - 0.5*dy, ny)
    zs = np.linspace(z_range[0] + 0.5*dz, z_range[1] - 0.5*dz, nz)
    return field_ln, xs, ys, zs


def crf_at_point(px, py, pz, field_ln, xs, ys, zs):
    ix = int(np.argmin(np.abs(xs - px)))
    iy = int(np.argmin(np.abs(ys - py)))
    iz = int(np.argmin(np.abs(zs - pz)))
    return float(field_ln[ix, iy, iz])


# ─────────────────────────────────────────────────────────────
# ATTRIBUTION DES PHASES AUX GRAINS
# ─────────────────────────────────────────────────────────────

def assign_phases_to_grains(nodes, tetras, grain_ids, field_ln, xs, ys, zs,
                             p_frac, phase_names):
    unique_grains = np.unique(grain_ids)

    grain_centroids = {}
    for gid in unique_grains:
        mask  = grain_ids == gid
        elems = tetras[mask]
        pts   = nodes[elems].mean(axis=1)
        grain_centroids[gid] = pts.mean(axis=0)

    grain_crf = {gid: crf_at_point(c[0], c[1], c[2], field_ln, xs, ys, zs)
                 for gid, c in grain_centroids.items()}
    grain_counts = {gid: int(np.sum(grain_ids == gid)) for gid in unique_grains}
    total = sum(grain_counts.values())

    crf_vals = [(grain_crf[gid], grain_counts[gid], gid) for gid in unique_grains]
    crf_vals.sort(key=lambda x: x[0])

    t1_target = p_frac[0] * total
    t2_target = (p_frac[0] + p_frac[1]) * total
    cum = 0
    t1_val = t2_val = crf_vals[-1][0]
    got_t1 = False
    for (val, count, gid) in crf_vals:
        cum += count
        if not got_t1 and cum >= t1_target:
            t1_val = val; got_t1 = True
        if cum >= t2_target:
            t2_val = val; break

    grain_phase = {}
    counts = [0, 0, 0]
    for gid in unique_grains:
        v = grain_crf[gid]
        if v <= t1_val:
            grain_phase[gid] = phase_names[0]; counts[0] += 1
        elif v <= t2_val:
            grain_phase[gid] = phase_names[1]; counts[1] += 1
        else:
            grain_phase[gid] = phase_names[2]; counts[2] += 1

    for i, name in enumerate(phase_names):
        print("  {} : {} grains".format(name, counts[i]))
    return grain_phase


# ─────────────────────────────────────────────────────────────
# DUPLICATION DES NOEUDS + INSERTION COH3D6
# (avec identification du type de contact minéral)
# ─────────────────────────────────────────────────────────────

def split_mesh_and_insert_cohesive(nodes, tetras, grain_ids, grain_phase):
    """
    Duplique les noeuds aux joints de grains et insère des COH3D6.
    Pour chaque élément cohésif, identifie le type de contact minéral
    (ex. 'IF_FELS_QRTZ') à partir des phases des deux grains adjacents.

    Retourne :
        new_nodes      (N_new, 3)
        new_tetras     (N_tet, 4)
        coh_elems      (N_coh, 6)   — 3 noeuds bas + 3 noeuds haut
        coh_if_types   [N_coh]      — type d'interface de chaque COH3D6
    """
    n_nodes  = len(nodes)
    n_tetras = len(tetras)

    # ── 1. Carte face → éléments ──
    print("  Construction de la carte face->elements ({} tets)...".format(n_tetras))
    sys.stdout.flush()

    TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_to_elems = defaultdict(list)

    for e_idx in range(n_tetras):
        n0, n1, n2, n3 = tetras[e_idx]
        tet = [int(n0), int(n1), int(n2), int(n3)]
        for fi, fj, fk in TET_FACES:
            face = tuple(sorted([tet[fi], tet[fj], tet[fk]]))
            face_to_elems[face].append(e_idx)
        if (e_idx + 1) % 100000 == 0:
            print("    {}/{} ({:.0f}%)".format(e_idx+1, n_tetras,
                                               100.*(e_idx+1)/n_tetras))
            sys.stdout.flush()

    # ── 2. Faces de joints de grains ──
    print("  Identification des faces de joints...")
    sys.stdout.flush()

    gb_faces = []   # (face_sorted, e_low, e_high, g_low, g_high)
    for face, elems in face_to_elems.items():
        if len(elems) == 2:
            e0, e1 = elems
            g0, g1 = int(grain_ids[e0]), int(grain_ids[e1])
            if g0 != g1:
                if g0 > g1:
                    e0, e1 = e1, e0
                    g0, g1 = g1, g0
                gb_faces.append((face, e0, e1, g0, g1))

    print("  {} faces de joints de grains trouvees".format(len(gb_faces)))
    sys.stdout.flush()

    # ── 3. Carte noeud → grains ──
    node_to_grains = defaultdict(set)
    for e_idx in range(n_tetras):
        g = int(grain_ids[e_idx])
        for n in tetras[e_idx]:
            node_to_grains[int(n)].add(g)

    # ── 4. Duplication des noeuds ──
    print("  Duplication des noeuds aux joints...")
    sys.stdout.flush()

    new_nodes_list = list(nodes)
    node_grain_map = {}   # (orig_node, grain) -> new_index

    for orig_n in range(n_nodes):
        grains = sorted(node_to_grains[orig_n])
        if not grains:
            continue
        node_grain_map[(orig_n, grains[0])] = orig_n
        for g in grains[1:]:
            new_idx = len(new_nodes_list)
            new_nodes_list.append(nodes[orig_n].copy())
            node_grain_map[(orig_n, g)] = new_idx

    new_nodes = np.array(new_nodes_list, dtype=np.float64)
    n_dup = len(new_nodes) - n_nodes
    print("  {} noeuds dupliques -> total {} noeuds".format(n_dup, len(new_nodes)))
    sys.stdout.flush()

    # ── 5. Mise à jour de la connectivité des tétraèdres ──
    print("  Mise a jour connectivite des tetraedres...")
    sys.stdout.flush()

    new_tetras = np.empty_like(tetras)
    for e_idx in range(n_tetras):
        g = int(grain_ids[e_idx])
        for k in range(4):
            new_tetras[e_idx, k] = node_grain_map[(int(tetras[e_idx, k]), g)]

    # ── 6. Création des COH3D6 avec type de contact ──
    print("  Creation des elements cohesifs COH3D6...")
    sys.stdout.flush()

    coh_list     = []
    coh_if_types = []

    if_counts = defaultdict(int)

    for face, e0, e1, g0, g1 in gb_faces:
        f0, f1, f2 = face
        bot = [node_grain_map[(f0, g0)],
               node_grain_map[(f1, g0)],
               node_grain_map[(f2, g0)]]
        top = [node_grain_map[(f0, g1)],
               node_grain_map[(f1, g1)],
               node_grain_map[(f2, g1)]]
        coh_list.append(bot + top)

        # Type d'interface : phases des deux grains adjacents
        p0 = grain_phase.get(g0, PHASE_NAMES[0])
        p1 = grain_phase.get(g1, PHASE_NAMES[0])
        if_type = interface_key(p0, p1)
        coh_if_types.append(if_type)
        if_counts[if_type] += 1

    coh_elems = (np.array(coh_list, dtype=np.int64)
                 if coh_list else np.empty((0, 6), dtype=np.int64))

    print("  {} elements COH3D6 crees :".format(len(coh_elems)))
    for itype in INTERFACE_TYPES:
        if if_counts[itype] > 0:
            print("    {:20s} : {}".format(itype, if_counts[itype]))
    sys.stdout.flush()

    return new_nodes, new_tetras, coh_elems, coh_if_types


# ─────────────────────────────────────────────────────────────
# EXPORT .INP ABAQUS
# ─────────────────────────────────────────────────────────────

def write_inp_with_interfaces(filepath, nodes, tetras, grain_ids, grain_phase,
                               phase_names, coh_elems, coh_if_types):
    """
    Écrit le .inp avec C3D4 (bulk) + COH3D6 (joints).
    6 ELSETS de COH3D6, un par type de contact minéral.
    """
    print("[Export] Ecriture de {}...".format(filepath))
    sys.stdout.flush()

    n_nodes  = len(nodes)
    n_tetras = len(tetras)
    n_coh    = len(coh_elems)
    node_off = 1
    elem_off = 1
    coh_off  = n_tetras + elem_off   # IDs COH3D6 après les C3D4

    # Tri des éléments bulk par phase
    phase_elems = {p: [] for p in phase_names}
    for i, gid in enumerate(grain_ids):
        phase = grain_phase.get(int(gid), phase_names[0])
        phase_elems[phase].append(i + elem_off)

    # Tri des COH3D6 par type d'interface
    if_elems = {itype: [] for itype in INTERFACE_TYPES}
    for i, itype in enumerate(coh_if_types):
        if_elems[itype].append(coh_off + i)

    def write_elset(f, name, eids):
        f.write("*Elset, elset={}\n".format(name))
        buf = []
        for eid in eids:
            buf.append(str(eid))
            if len(buf) == 16:
                f.write(", ".join(buf) + ",\n"); buf = []
        if buf:
            f.write(", ".join(buf) + "\n")

    with open(filepath, 'w') as f:
        f.write("** Abaqus orphan mesh — Granite avec joints COH3D6 differencies\n")
        f.write("** grain_interface_prep.py\n")
        f.write("** Ref: Pan et al. 2023 (C&G), Liu et al. 2023 (RMRE),\n")
        f.write("**      Nasseri & Young 2009 (IJRM)\n")
        f.write("*Part, name=GRANITE\n")

        # ── Noeuds ──
        f.write("*Node\n")
        print("  Noeuds ({})...".format(n_nodes)); sys.stdout.flush()
        for i, (x, y, z) in enumerate(nodes):
            f.write("{}, {:.6f}, {:.6f}, {:.6f}\n".format(
                i + node_off, x, y, z))
            if (i + 1) % 100000 == 0:
                print("    {}/{} ({:.0f}%)".format(i+1, n_nodes, 100.*(i+1)/n_nodes))
                sys.stdout.flush()

        # ── Éléments bulk C3D4 ──
        f.write("*Element, type=C3D4\n")
        print("  Elements C3D4 ({})...".format(n_tetras)); sys.stdout.flush()
        for i, (n0, n1, n2, n3) in enumerate(tetras):
            f.write("{}, {}, {}, {}, {}\n".format(
                i + elem_off,
                int(n0)+node_off, int(n1)+node_off,
                int(n2)+node_off, int(n3)+node_off))
            if (i + 1) % 100000 == 0:
                print("    {}/{} ({:.0f}%)".format(i+1, n_tetras, 100.*(i+1)/n_tetras))
                sys.stdout.flush()

        # ── Éléments cohésifs COH3D6 ──
        if n_coh > 0:
            f.write("*Element, type=COH3D6\n")
            print("  Elements COH3D6 ({})...".format(n_coh)); sys.stdout.flush()
            for i, row in enumerate(coh_elems):
                f.write("{}, {}, {}, {}, {}, {}, {}\n".format(
                    coh_off + i,
                    int(row[0])+node_off, int(row[1])+node_off, int(row[2])+node_off,
                    int(row[3])+node_off, int(row[4])+node_off, int(row[5])+node_off))
                if (i + 1) % 50000 == 0:
                    print("    {}/{} ({:.0f}%)".format(i+1, n_coh, 100.*(i+1)/n_coh))
                    sys.stdout.flush()

        # ── ELSETS bulk ──
        for phase in phase_names:
            write_elset(f, 'ELSET_{}'.format(phase.upper()), phase_elems[phase])

        # ── ELSETS par type d'interface ──
        for itype in INTERFACE_TYPES:
            if if_elems[itype]:
                write_elset(f, itype, if_elems[itype])

        f.write("*End Part\n")

    # Résumé
    for p in phase_names:
        print("  ELSET_{:10s} : {} elements".format(p.upper(), len(phase_elems[p])))
    for itype in INTERFACE_TYPES:
        if if_elems[itype]:
            print("  {:20s} : {} elements COH3D6".format(itype, len(if_elems[itype])))
    print("  -> {} ecrit".format(filepath))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(WORKDIR, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  grain_interface_prep.py — Joints de grains differencies")
    print("=" * 65)
    sys.stdout.flush()

    # ── ÉTAPE 1 : Maillage Neper ──
    print("\n[ETAPE 1/5] Maillage Neper")
    mesh_file = os.path.normpath(NEPER_MESH_FILE)
    if os.path.exists(mesh_file):
        print("  Reutilisation du maillage existant :")
        print("  {}".format(mesh_file))
    else:
        print("  Maillage non trouve, generation via WSL...")
        neper_workdir = os.path.join(WORKDIR, "neper")
        _, mesh_file = run_neper(
            neper_workdir, BASENAME, N_GRAINS, MESH_CL, CYL_R, CYL_H, RANDOM_SEED)
    sys.stdout.flush()

    # ── ÉTAPE 2 : Lecture maillage ──
    print("\n[ETAPE 2/5] Lecture du maillage")
    nodes, tetras, grain_ids = load_mesh(mesh_file)
    sys.stdout.flush()

    # ── ÉTAPE 3 : CRF + attribution des phases ──
    print("\n[ETAPE 3/5] CRF + attribution des phases minerales")
    margin  = LAMBDA_RF
    x_range = (-CYL_R - margin, CYL_R + margin)
    y_range = (-CYL_R - margin, CYL_R + margin)
    z_range = (-margin,          CYL_H + margin)
    field_ln, xs, ys, zs = generate_crf(
        32, 32, 64, LAMBDA_RF, SIGMA_RF, MU_RF,
        x_range, y_range, z_range, seed=RANDOM_SEED)
    print("  CRF : mean={:.3f}  std={:.3f}".format(
        float(field_ln.mean()), float(field_ln.std())))
    grain_phase = assign_phases_to_grains(
        nodes, tetras, grain_ids, field_ln, xs, ys, zs, P_FRAC, PHASE_NAMES)
    sys.stdout.flush()

    json_file = os.path.join(WORKDIR, BASENAME + "_grain_phases.json")
    with open(json_file, 'w') as fj:
        json.dump({str(k): v for k, v in grain_phase.items()}, fj, indent=2)
    print("  -> JSON : {}".format(json_file))

    # ── ÉTAPE 4 : Duplication noeuds + insertion COH3D6 ──
    print("\n[ETAPE 4/5] Insertion des COH3D6 differencies par type de contact")
    new_nodes, new_tetras, coh_elems, coh_if_types = \
        split_mesh_and_insert_cohesive(nodes, tetras, grain_ids, grain_phase)
    sys.stdout.flush()

    # ── ÉTAPE 5 : Export .inp ──
    print("\n[ETAPE 5/5] Export .inp Abaqus")
    inp_file = os.path.join(WORKDIR, BASENAME + "_interfaces.inp")
    write_inp_with_interfaces(
        inp_file, new_nodes, new_tetras, grain_ids, grain_phase,
        PHASE_NAMES, coh_elems, coh_if_types)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  Termine en {:.0f}s ({:.1f} min)".format(elapsed, elapsed / 60.))
    print("  .inp : {}".format(inp_file))
    print("  -> Lancez build_model_interfaces.py dans Abaqus/CAE")
    print("=" * 65)


if __name__ == "__main__":
    main()

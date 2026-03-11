# -*- coding: utf-8 -*-
"""
granite_abaqus_prep.py  —  Etape 1 : Preprocessing (Windows Python)
=====================================================================
1. Génère une tessellation Voronoï cylindrique via Neper (WSL)
2. Maille en tétraèdres (C3D4)
3. Lit le maillage avec meshio
4. Assigne les phases minérales aux grains via CRF log-normal (Gutjahr 1989)
5. Exporte un fichier .inp Abaqus avec les elsets par phase
6. Exporte un fichier JSON de correspondance grain -> phase

Prérequis : pip install meshio numpy scipy
            Neper installé dans WSL2
"""

import os
import json
import subprocess
import numpy as np
import meshio

# ─────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────

WORKDIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "granite_abaqus")
BASENAME = "granite_cyl"

# Géométrie du cylindre (identique à build_model_5MPa)
CYL_R = 2.5   # mm
CYL_H = 10.0  # mm

# Neper
N_GRAINS  = 200
MESH_CL   = 0.15   # taille de maille [mm]
RANDOM_SEED = 42

# CRF
LAMBDA_RF = 0.8    # longueur de corrélation [mm]
SIGMA_RF  = 0.35
MU_RF     = 1.0

# Fractions volumiques (Feldspath, Quartz, Biotite)
P_FRAC = (0.30, 0.50, 0.20)

# Noms des phases (ordre = quantiles croissants du CRF)
PHASE_NAMES = ['Feldspath', 'Quartz', 'Biotite']


# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────

def win_to_wsl(path):
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path  = "/mnt/" + drive + path[2:]
    return path


# ─────────────────────────────────────────────────────────────
# 1. NEPER : TESSELLATION + MAILLAGE CYLINDRIQUE
# ─────────────────────────────────────────────────────────────

def run_neper(workdir, basename, n_grains, mesh_cl, cyl_r, cyl_h, seed):
    os.makedirs(workdir, exist_ok=True)

    tess_file = os.path.join(workdir, basename + ".tess")
    mesh_file = os.path.join(workdir, basename + "_mesh.msh")

    wsl_dir   = win_to_wsl(workdir)
    wsl_tess  = win_to_wsl(tess_file)
    wsl_out   = wsl_dir + "/" + basename
    wsl_mout  = wsl_dir + "/" + basename + "_mesh"

    # ── Tessellation sur domaine cylindrique ──
    if not os.path.exists(tess_file):
        print("[Neper 1/2] Tessellation cylindrique ({} grains)...".format(n_grains))
        print("  (peut prendre 30s-2min selon le nombre de grains)")
        import sys
        sys.stdout.flush()
        # Syntaxe Neper 4.x : cylinder(height,diameter)
        diam = 2.0 * cyl_r
        cmd = (
            "neper -T -n {n} -id {seed} "
            "-domain 'cylinder({h},{d})' "
            "-morpho graingrowth "
            "-morpho_optiini_iter 10000 "
            "-o '{o}'"
        ).format(n=n_grains, seed=seed, h=cyl_h, d=diam, o=wsl_out)
        # Streaming de la sortie Neper ligne par ligne
        proc = subprocess.Popen(
            ["wsl", "bash", "-c", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print("  [Neper-T] " + line)
                sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Neper -T echoue (code {})".format(proc.returncode))
        print("  -> tessellation generee")
    else:
        print("[Neper 1/2] Tessellation deja presente.")

    # ── Maillage tétraédrique ──
    if not os.path.exists(mesh_file):
        import sys
        print("[Neper 2/2] Maillage (cl={})...".format(mesh_cl))
        print("  (peut prendre 1-5min selon la taille de maille)")
        sys.stdout.flush()
        cmd = (
            "neper -M '{t}' -cl {cl} -format msh -o '{o}'"
        ).format(t=wsl_tess, cl=mesh_cl, o=wsl_mout)
        proc = subprocess.Popen(
            ["wsl", "bash", "-c", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print("  [Neper-M] " + line)
                sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Neper -M echoue (code {})".format(proc.returncode))
        print("  -> maillage genere")
    else:
        print("[Neper 2/2] Maillage deja present.")

    return tess_file, mesh_file


# ─────────────────────────────────────────────────────────────
# 2. LECTURE DU MAILLAGE
# ─────────────────────────────────────────────────────────────

def load_mesh(mesh_file):
    print("[Mesh] Chargement de {}...".format(mesh_file))
    mesh = meshio.read(mesh_file)

    print("  Types de cellules : {}".format([b.type for b in mesh.cells]))
    print("  Cles cell_data    : {}".format(list(mesh.cell_data.keys())))

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
        print("  Avertissement : pas de tag grain, tous = 1")
        grain_ids = np.ones(len(tetras), dtype=int)

    nodes = mesh.points
    print("  -> {} noeuds, {} elements, {} grains".format(
        nodes.shape[0], tetras.shape[0], len(np.unique(grain_ids))
    ))
    return nodes, tetras, grain_ids


# ─────────────────────────────────────────────────────────────
# 3. CRF LOG-NORMAL (Gutjahr 1989)
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

    S       = (lambda_rf**3 * np.pi**1.5) * np.exp(-K2 * lambda_rf**2 / 4.0)
    noise   = (np.random.randn(nx,ny,nz) + 1j*np.random.randn(nx,ny,nz)) / np.sqrt(2)
    field_g = np.real(np.fft.ifftn(np.sqrt(S) * noise))
    field_g = (field_g - field_g.mean()) / (field_g.std() + 1e-12)

    sigma_ln = np.sqrt(np.log(1 + (sigma_rf/mu_rf)**2))
    mu_ln    = np.log(mu_rf) - 0.5*sigma_ln**2
    field_ln = np.exp(mu_ln + sigma_ln * field_g)

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
# 4. ATTRIBUTION DES PHASES AUX GRAINS (via CRF)
# ─────────────────────────────────────────────────────────────

def assign_phases_to_grains(nodes, tetras, grain_ids, field_ln, xs, ys, zs,
                             p_frac, phase_names):
    """
    Assigne une phase minérale à chaque grain en se basant
    sur la valeur CRF au centroïde du grain (approche grain-level).
    """
    unique_grains = np.unique(grain_ids)

    # Centroïde de chaque grain = moyenne des centroïdes de ses éléments
    grain_centroids = {}
    for gid in unique_grains:
        mask  = grain_ids == gid
        elems = tetras[mask]
        pts   = nodes[elems].mean(axis=1)   # centroïdes des éléments du grain
        grain_centroids[gid] = pts.mean(axis=0)

    # Valeur CRF au centroïde de chaque grain
    grain_crf = {}
    for gid, centroid in grain_centroids.items():
        grain_crf[gid] = crf_at_point(
            centroid[0], centroid[1], centroid[2], field_ln, xs, ys, zs
        )

    # Seuils pondérés par nombre d'éléments par grain
    grain_counts = {gid: int(np.sum(grain_ids == gid)) for gid in unique_grains}
    total = sum(grain_counts.values())

    crf_vals = [(grain_crf[gid], grain_counts[gid], gid) for gid in unique_grains]
    crf_vals.sort(key=lambda x: x[0])

    t1_target = p_frac[0] * total
    t2_target = (p_frac[0] + p_frac[1]) * total

    cum = 0
    t1_gid = t2_gid = crf_vals[-1][2]
    got_t1 = False
    for (val, count, gid) in crf_vals:
        cum += count
        if not got_t1 and cum >= t1_target:
            t1_gid = gid
            got_t1 = True
        if cum >= t2_target:
            t2_gid = gid
            break

    t1 = grain_crf[t1_gid]
    t2 = grain_crf[t2_gid]
    print("  Seuils CRF grain : t1={:.4f}  t2={:.4f}".format(t1, t2))

    # Attribution grain -> phase
    grain_phase = {}
    counts = [0, 0, 0]
    for gid in unique_grains:
        v = grain_crf[gid]
        if v <= t1:
            grain_phase[gid] = phase_names[0]
            counts[0] += 1
        elif v <= t2:
            grain_phase[gid] = phase_names[1]
            counts[1] += 1
        else:
            grain_phase[gid] = phase_names[2]
            counts[2] += 1

    for i, name in enumerate(phase_names):
        print("  {} : {} grains".format(name, counts[i]))

    return grain_phase


# ─────────────────────────────────────────────────────────────
# 5. EXPORT ABAQUS .INP
# ─────────────────────────────────────────────────────────────

def write_abaqus_inp(filepath, nodes, tetras, grain_ids, grain_phase, phase_names):
    """
    Ecrit un fichier .inp Abaqus avec :
    - Noeuds (1-indexés)
    - Elements C3D4 (1-indexés)
    - Elsets par phase minérale
    """
    print("[Export] Ecriture de {}...".format(filepath))

    # Renumérotation 1-based
    node_offset = 1   # nodes sont déjà 0-indexés dans meshio
    elem_offset = 1

    phase_elems = {p: [] for p in phase_names}
    for i, gid in enumerate(grain_ids):
        phase = grain_phase.get(int(gid), phase_names[0])
        phase_elems[phase].append(i + elem_offset)

    import sys
    n_nodes = len(nodes)
    n_elems = len(tetras)

    with open(filepath, 'w') as f:
        f.write("** Abaqus orphan mesh - Granite microstructure (Neper + CRF)\n")
        f.write("** Generated by granite_abaqus_prep.py\n")
        f.write("*Part, name=GRANITE\n")

        # Noeuds
        f.write("*Node\n")
        print("  Ecriture des noeuds ({})...".format(n_nodes))
        sys.stdout.flush()
        for i, (x, y, z) in enumerate(nodes):
            f.write("{}, {:.6f}, {:.6f}, {:.6f}\n".format(
                i + node_offset, x, y, z
            ))
            if (i + 1) % 50000 == 0:
                print("    {}/{} noeuds ({:.0f}%)".format(
                    i + 1, n_nodes, 100.0*(i+1)/n_nodes))
                sys.stdout.flush()

        # Elements C3D4
        f.write("*Element, type=C3D4\n")
        print("  Ecriture des elements ({})...".format(n_elems))
        sys.stdout.flush()
        for i, (n1, n2, n3, n4) in enumerate(tetras):
            f.write("{}, {}, {}, {}, {}\n".format(
                i + elem_offset,
                n1 + node_offset,
                n2 + node_offset,
                n3 + node_offset,
                n4 + node_offset
            ))
            if (i + 1) % 100000 == 0:
                print("    {}/{} elements ({:.0f}%)".format(
                    i + 1, n_elems, 100.0*(i+1)/n_elems))
                sys.stdout.flush()

        # Elsets par phase (max 16 par ligne pour Abaqus)
        for phase in phase_names:
            elems = phase_elems[phase]
            f.write("*Elset, elset=ELSET_{}\n".format(phase.upper()))
            line_buf = []
            for eid in elems:
                line_buf.append(str(eid))
                if len(line_buf) == 16:
                    f.write(", ".join(line_buf) + ",\n")
                    line_buf = []
            if line_buf:
                f.write(", ".join(line_buf) + "\n")

        f.write("*End Part\n")

    # Compte les éléments par phase
    for phase in phase_names:
        print("  ELSET_{} : {} elements".format(
            phase.upper(), len(phase_elems[phase])
        ))
    print("  -> {} ecrit".format(filepath))
    return phase_elems


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    import sys
    import time
    np.random.seed(RANDOM_SEED)
    os.makedirs(WORKDIR, exist_ok=True)

    print("=" * 60)
    print("  granite_abaqus_prep.py — Démarrage")
    print("  {} grains | maille cl={} | {} elements".format(
        N_GRAINS, MESH_CL, "estim."))
    print("=" * 60)
    sys.stdout.flush()

    t0 = time.time()

    # 1. Neper
    print("\n[ETAPE 1/4] Tessellation + maillage Neper")
    sys.stdout.flush()
    _, mesh_file = run_neper(
        WORKDIR, BASENAME, N_GRAINS, MESH_CL, CYL_R, CYL_H, RANDOM_SEED
    )

    # 2. Lecture maillage
    print("\n[ETAPE 2/4] Lecture du maillage")
    sys.stdout.flush()
    nodes, tetras, grain_ids = load_mesh(mesh_file)
    print("  Fait en {:.1f}s".format(time.time() - t0))
    sys.stdout.flush()

    # 3. CRF sur le domaine cylindrique
    print("\n[ETAPE 3/4] Génération du CRF log-normal (Gutjahr 1989)...")
    margin = LAMBDA_RF
    x_range = (-CYL_R - margin, CYL_R + margin)
    y_range = (-CYL_R - margin, CYL_R + margin)
    z_range = (-margin,          CYL_H + margin)
    nx, ny, nz = 32, 32, 64

    field_ln, xs, ys, zs = generate_crf(
        nx, ny, nz, LAMBDA_RF, SIGMA_RF, MU_RF,
        x_range, y_range, z_range, seed=RANDOM_SEED
    )
    print("  CRF : mean={:.3f}  std={:.3f}".format(
        float(field_ln.mean()), float(field_ln.std())
    ))

    # 4. Attribution phases aux grains
    print("\n[ETAPE 4/4] Attribution des phases minérales + export .inp")
    grain_phase = assign_phases_to_grains(
        nodes, tetras, grain_ids, field_ln, xs, ys, zs, P_FRAC, PHASE_NAMES
    )

    # Sauvegarde JSON grain -> phase
    json_file = os.path.join(WORKDIR, BASENAME + "_grain_phases.json")
    grain_phase_str = {str(k): v for k, v in grain_phase.items()}
    with open(json_file, 'w') as f:
        json.dump(grain_phase_str, f, indent=2)
    print("  -> {} ecrit".format(json_file))

    # 5. Export Abaqus .inp
    inp_file = os.path.join(WORKDIR, BASENAME + "_abaqus.inp")
    write_abaqus_inp(inp_file, nodes, tetras, grain_ids, grain_phase, PHASE_NAMES)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  ✓ Preprocessing termine en {:.0f}s ({:.1f}min)".format(
        elapsed, elapsed / 60.0))
    print("  Fichier .inp : {}".format(inp_file))
    print("  Lancez ensuite build_model_neper.py dans Abaqus/CAE.")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# Script Python standard (hors Abaqus)
# =====================================================================
# prep_neper_box.py
# Genere le maillage polycristallin Neper pour le specimen parallelepiped
# 5x5x10 mm destine au modele de percussion (08_percussion).
#
# Etapes :
#   1. Tessellation Voronoi (Neper -T) — domaine cuboid
#   2. Maillage hexaedrique (Neper -M, elttype hex)
#   3. Lecture du .msh, assignation des phases via CRF (Gutjahr 1989)
#      Feldspath 30%, Quartz 50%, Biotite 20%
#   4. Export granite_box_abaqus.inp (pret pour build_neper.py)
#
# Prerequis : Neper installe sous WSL
# Utilisation : python prep_neper_box.py
# =====================================================================

import os
import subprocess
import struct
import math
import numpy as np

# ─────────────────────────────────────────────────────────────
# PARAMETRES
# ─────────────────────────────────────────────────────────────

BOX_LX   = 5.0     # mm
BOX_LY   = 5.0     # mm
BOX_LZ   = 10.0    # mm (axe de chargement z)

N_GRAINS  = 200
MESH_CL   = 0.3    # taille cible des elements [mm]
RANDOM_SEED = 42

# Phases CRF
P_FRAC   = (0.30, 0.50, 0.20)   # Feldspath, Quartz, Biotite
LAMBDA_C = 0.8                   # longueur de correlation [mm]
N_MODES  = 2000                  # nb de modes FFT Gutjahr

# Dossier de sortie (dans 08_percussion)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'granite_box')
OUT_BASE   = os.path.join(OUT_DIR, 'granite_box')
INP_OUT    = OUT_BASE + '_abaqus.inp'

# ─────────────────────────────────────────────────────────────
# HELPERS WSL
# ─────────────────────────────────────────────────────────────

def win_to_wsl(win_path):
    """Convertit un chemin Windows en chemin WSL."""
    p = win_path.replace('\\', '/')
    if len(p) >= 2 and p[1] == ':':
        drive = p[0].lower()
        p = '/mnt/' + drive + p[2:]
    return p


def run_neper(args, cwd):
    """Lance une commande Neper sous WSL."""
    cmd = ['wsl', 'neper'] + args
    print("  CMD: " + ' '.join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError("Neper a echoue (code {})".format(result.returncode))


# ─────────────────────────────────────────────────────────────
# CRF (Champ aleatoire correle, Gutjahr 1989)
# ─────────────────────────────────────────────────────────────

def generate_crf_lognormal(nx, ny, nz, dx, dy, dz, lam, n_modes, seed):
    """
    Champ gaussien lognormal 3D par FFT (methode de Gutjahr 1989).
    Retourne un tableau (nz, ny, nx) de valeurs lognormales.
    """
    rng = np.random.default_rng(seed)

    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2.0 * np.pi

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    # Spectre de puissance gaussien
    S = np.exp(-K2 * lam**2 / (4.0 * np.pi**2))
    S[0, 0, 0] = 0.0

    # Champ gaussien en espace de Fourier
    amp   = np.sqrt(S * nx * ny * nz / (dx * dy * dz))
    phase = rng.standard_normal((nz, ny, nx)) + 1j * rng.standard_normal((nz, ny, nx))
    gfield = np.real(np.fft.ifftn(amp * phase))

    # Normalisation -> lognormal
    mu    = np.mean(gfield)
    sigma = np.std(gfield) + 1e-12
    gfield = (gfield - mu) / sigma
    return np.exp(gfield)


def weighted_quantile_thresholds(values, weights, fracs):
    f1 = fracs[0]
    f2 = fracs[0] + fracs[1]
    pairs = sorted(zip(values.ravel(), weights.ravel()), key=lambda vw: vw[0])
    total_w = sum(w for v, w in pairs) + 1e-30
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


def interpolate_crf(crf_field, centroid, bbox, shape):
    """Interpolation trilineaire du champ CRF a un point centroid."""
    nz, ny, nx = shape
    xmin, ymin, zmin = bbox[0]
    xmax, ymax, zmax = bbox[1]
    cx, cy, cz = centroid

    fx = (cx - xmin) / (xmax - xmin + 1e-12) * (nx - 1)
    fy = (cy - ymin) / (ymax - ymin + 1e-12) * (ny - 1)
    fz = (cz - zmin) / (zmax - zmin + 1e-12) * (nz - 1)

    ix = int(max(0, min(nx - 2, math.floor(fx))))
    iy = int(max(0, min(ny - 2, math.floor(fy))))
    iz = int(max(0, min(nz - 2, math.floor(fz))))

    tx = fx - ix
    ty = fy - iy
    tz = fz - iz

    v000 = crf_field[iz,   iy,   ix  ]
    v100 = crf_field[iz,   iy,   ix+1]
    v010 = crf_field[iz,   iy+1, ix  ]
    v110 = crf_field[iz,   iy+1, ix+1]
    v001 = crf_field[iz+1, iy,   ix  ]
    v101 = crf_field[iz+1, iy,   ix+1]
    v011 = crf_field[iz+1, iy+1, ix  ]
    v111 = crf_field[iz+1, iy+1, ix+1]

    val = (v000*(1-tx)*(1-ty)*(1-tz) + v100*tx*(1-ty)*(1-tz) +
           v010*(1-tx)*ty*(1-tz)     + v110*tx*ty*(1-tz) +
           v001*(1-tx)*(1-ty)*tz     + v101*tx*(1-ty)*tz +
           v011*(1-tx)*ty*tz         + v111*tx*ty*tz)
    return val


# ─────────────────────────────────────────────────────────────
# LECTURE MAILLAGE .msh (Gmsh format 2)
# ─────────────────────────────────────────────────────────────

def read_gmsh_msh(msh_file):
    """Lit un fichier .msh Gmsh format 2, retourne nodes et elements hex/tet."""
    nodes = {}      # nid -> (x, y, z)
    elements = {}   # eid -> (etype, tags, connectivity)
    # etype : 5=hex8, 4=tet4, 2=tri3, 3=quad4

    with open(msh_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '$Nodes':
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                nid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[nid] = (x, y, z)
                i += 1
        elif line == '$Elements':
            i += 1
            n_elems = int(lines[i].strip())
            i += 1
            for _ in range(n_elems):
                parts = lines[i].strip().split()
                eid   = int(parts[0])
                etype = int(parts[1])
                n_tags = int(parts[2])
                tags  = [int(parts[3 + j]) for j in range(n_tags)]
                conn  = [int(parts[3 + n_tags + j])
                         for j in range(len(parts) - 3 - n_tags)]
                elements[eid] = (etype, tags, conn)
                i += 1
        else:
            i += 1

    return nodes, elements


# ─────────────────────────────────────────────────────────────
# ECRITURE DU FICHIER .INP
# ─────────────────────────────────────────────────────────────

def write_abaqus_inp(out_path, nodes, vol_elements, phase_labels, surf_sets):
    """
    Ecrit un fichier .inp Abaqus avec :
      - Part GRANITE (orphan mesh)
      - Elsets par phase (ELSET_FELDSPATH, ELSET_QUARTZ, ELSET_BIOTITE)
      - Surfaces de bord (SURF_BOT, SURF_TOP, SURF_X0/X1/Y0/Y1)
    """
    CHUNK = 16  # labels par ligne

    def write_set(f, set_type, name, labels):
        f.write('*{}, {}={}\n'.format(set_type, set_type.lower(), name))
        for k in range(0, len(labels), CHUNK):
            f.write(', '.join(str(lb) for lb in labels[k:k+CHUNK]) + ',\n')

    with open(out_path, 'w') as f:
        f.write('** Granite polycristallin - genere par prep_neper_box.py\n')
        f.write('*Part, name=GRANITE\n')
        f.write('*Node\n')
        for nid in sorted(nodes.keys()):
            x, y, z = nodes[nid]
            f.write('{:d}, {:.8e}, {:.8e}, {:.8e}\n'.format(nid, x, y, z))

        # Elements (uniquement 3D volumiques)
        hex_elems = {eid: e for eid, e in vol_elements.items() if e[0] == 5}
        tet_elems = {eid: e for eid, e in vol_elements.items() if e[0] == 4}

        if hex_elems:
            f.write('*Element, type=C3D8R\n')
            for eid in sorted(hex_elems.keys()):
                conn = hex_elems[eid][2]
                f.write('{:d}, '.format(eid) + ', '.join(str(n) for n in conn) + '\n')
        if tet_elems:
            f.write('*Element, type=C3D4\n')
            for eid in sorted(tet_elems.keys()):
                conn = tet_elems[eid][2]
                f.write('{:d}, '.format(eid) + ', '.join(str(n) for n in conn) + '\n')

        # Elsets par phase
        for phase_name, labels in phase_labels.items():
            if labels:
                write_set(f, 'Elset', 'ELSET_' + phase_name, labels)

        # Elset ALL
        all_labels = list(hex_elems.keys()) + list(tet_elems.keys())
        write_set(f, 'Elset', 'ALL', all_labels)

        # Surfaces
        for surf_name, face_pairs in surf_sets.items():
            f.write('*Surface, type=ELEMENT, name={}\n'.format(surf_name))
            for eid, face_id in face_pairs:
                f.write('{:d}, S{:d}\n'.format(eid, face_id))

        f.write('*End Part\n')

    print("  -> Fichier .inp ecrit : {}".format(out_path))


# ─────────────────────────────────────────────────────────────
# IDENTIFICATION DES FACES DE BORD
# ─────────────────────────────────────────────────────────────

def find_surface_faces_hex(nodes, hex_elems, bbox, tol=1e-2):
    """
    Pour les elements hex (C3D8R, 8 noeuds), identifie les faces sur les 6 surfaces.
    Face locale d'un C3D8R : S1(1-2-3-4), S2(5-8-7-6), S3(1-5-6-2),
                              S4(2-6-7-3), S5(3-7-8-4), S6(4-8-5-1)
    indices Python (0-based): S1=(0,1,2,3), S2=(4,7,6,5), S3=(0,4,5,1),
                               S4=(1,5,6,2), S5=(2,6,7,3), S6=(3,7,4,0)
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox

    # Faces locales (1-indexed connectivity positions, 0-based)
    face_nodes = {
        1: [0, 1, 2, 3],
        2: [4, 7, 6, 5],
        3: [0, 4, 5, 1],
        4: [1, 5, 6, 2],
        5: [2, 6, 7, 3],
        6: [3, 7, 4, 0],
    }

    surfaces = {
        'SURF_BOT': [],  # z = zmin
        'SURF_TOP': [],  # z = zmax
        'SURF_X0':  [],  # x = xmin
        'SURF_X1':  [],  # x = xmax
        'SURF_Y0':  [],  # y = ymin
        'SURF_Y1':  [],  # y = ymax
    }

    def face_centroid(conn, fn_indices):
        xs = [nodes[conn[k]][0] for k in fn_indices]
        ys = [nodes[conn[k]][1] for k in fn_indices]
        zs = [nodes[conn[k]][2] for k in fn_indices]
        return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

    for eid, (etype, tags, conn) in hex_elems.items():
        for fid, fn_idx in face_nodes.items():
            cx, cy, cz = face_centroid(conn, fn_idx)
            if abs(cz - zmin) < tol:
                surfaces['SURF_BOT'].append((eid, fid))
            elif abs(cz - zmax) < tol:
                surfaces['SURF_TOP'].append((eid, fid))
            elif abs(cx - xmin) < tol:
                surfaces['SURF_X0'].append((eid, fid))
            elif abs(cx - xmax) < tol:
                surfaces['SURF_X1'].append((eid, fid))
            elif abs(cy - ymin) < tol:
                surfaces['SURF_Y0'].append((eid, fid))
            elif abs(cy - ymax) < tol:
                surfaces['SURF_Y1'].append((eid, fid))

    return surfaces


# ─────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir_wsl = win_to_wsl(OUT_DIR)

    # ── 1. Tessellation Neper ─────────────────────────────────
    print("=== Etape 1 : Tessellation Neper ===")
    tess_file = OUT_BASE + '.tess'

    run_neper(
        ['-T',
         '-n', str(N_GRAINS),
         '-domain', 'cube({},{},{})'.format(BOX_LX, BOX_LY, BOX_LZ),
         '-id', str(RANDOM_SEED),
         '-o', win_to_wsl(OUT_BASE)],
        cwd=out_dir_wsl
    )
    print("  -> Tessellation : {}".format(tess_file))

    # ── 2. Maillage Neper (-elttype hex) ─────────────────────
    print("=== Etape 2 : Maillage Neper (hex) ===")
    msh_file = OUT_BASE + '.msh'

    run_neper(
        ['-M', win_to_wsl(tess_file),
         '-elttype', 'hex',
         '-cl', str(MESH_CL),
         '-order', '1',
         '-format', 'msh',
         '-o', win_to_wsl(OUT_BASE)],
        cwd=out_dir_wsl
    )
    print("  -> Maillage : {}".format(msh_file))

    # ── 3. Lecture du maillage ────────────────────────────────
    print("=== Etape 3 : Lecture et assignation CRF ===")
    nodes, elements = read_gmsh_msh(msh_file)

    vol_elements = {eid: e for eid, e in elements.items() if e[0] in (4, 5)}
    print("  -> Noeuds : {}  Elements 3D : {}".format(len(nodes), len(vol_elements)))

    # Bounding box du maillage
    all_x = [nodes[n][0] for n in nodes]
    all_y = [nodes[n][1] for n in nodes]
    all_z = [nodes[n][2] for n in nodes]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    zmin, zmax = min(all_z), max(all_z)
    bbox = ((xmin, ymin, zmin), (xmax, ymax, zmax))
    print("  Bbox : x[{:.2f},{:.2f}] y[{:.2f},{:.2f}] z[{:.2f},{:.2f}]".format(
        xmin, xmax, ymin, ymax, zmin, zmax))

    # ── 4. Champ CRF ─────────────────────────────────────────
    print("  Calcul du champ CRF lognormal...")
    nx = max(16, int(round((xmax - xmin) / LAMBDA_C * 4)))
    ny = max(16, int(round((ymax - ymin) / LAMBDA_C * 4)))
    nz = max(16, int(round((zmax - zmin) / LAMBDA_C * 4)))
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    crf = generate_crf_lognormal(nx, ny, nz, dx, dy, dz, LAMBDA_C, N_MODES, RANDOM_SEED)

    # Calcul des valeurs et volumes par element
    fvals = []
    vols  = []
    centroids = []
    for eid, (etype, tags, conn) in vol_elements.items():
        xs = [nodes[n][0] for n in conn]
        ys = [nodes[n][1] for n in conn]
        zs = [nodes[n][2] for n in conn]
        cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
        centroids.append((eid, cx, cy, cz))
        fvals.append(interpolate_crf(crf, (cx, cy, cz), bbox, (nz, ny, nx)))
        vols.append(1.0)  # volume uniforme (hex regulier)

    fvals_arr = np.array(fvals)
    vols_arr  = np.ones(len(fvals))

    t1, t2 = weighted_quantile_thresholds(fvals_arr, vols_arr, P_FRAC)
    print("  Seuils CRF : t1={:.4f}  t2={:.4f}".format(t1, t2))

    phase_labels = {'FELDSPATH': [], 'QUARTZ': [], 'BIOTITE': []}
    eids = list(vol_elements.keys())
    for i, eid in enumerate(eids):
        f = fvals[i]
        if f <= t1:
            phase_labels['FELDSPATH'].append(eid)
        elif f <= t2:
            phase_labels['QUARTZ'].append(eid)
        else:
            phase_labels['BIOTITE'].append(eid)

    print("  Feldspath: {}  Quartz: {}  Biotite: {}".format(
        len(phase_labels['FELDSPATH']),
        len(phase_labels['QUARTZ']),
        len(phase_labels['BIOTITE'])))

    # ── 5. Surfaces de bord ───────────────────────────────────
    print("  Identification des surfaces de bord...")
    hex_elems = {eid: e for eid, e in vol_elements.items() if e[0] == 5}
    surf_sets = find_surface_faces_hex(nodes, hex_elems, bbox)
    for sname, faces in surf_sets.items():
        print("  {} : {} faces".format(sname, len(faces)))

    # ── 6. Ecriture .inp ──────────────────────────────────────
    print("=== Etape 4 : Ecriture du fichier .inp ===")
    write_abaqus_inp(INP_OUT, nodes, vol_elements, phase_labels, surf_sets)

    print("\n=== prep_neper_box.py termine. ===")
    print("Fichier genere : {}".format(INP_OUT))
    print("Lancer ensuite build_neper.py dans Abaqus CAE.")

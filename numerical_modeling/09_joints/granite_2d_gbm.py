#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
granite_2d_gbm.py — Modèle 2D Grain-Based avec éléments cohésifs
=================================================================
Implémente les concepts de :
  - Zhang et al. (2023)        : GBM multi-échelle, contacts inter/intra-grain
  - Simonovski & Cizelj (2013) : Éléments cohésifs aux joints de grains, viscous reg.
  - Simonovski & Cizelj (2015) : Bonnes pratiques convergence Abaqus

Génère un fichier .inp Abaqus 2D (plane strain) avec :
  - Grains Voronoï maillés en CPE3 (triangles plane strain)
  - Éléments cohésifs COH2D4 (épaisseur zéro) aux joints de grains
  - Matériaux CDP pour les grains (Feldspath / Quartz / Biotite)
  - Loi traction-séparation bilinéaire pour les joints
  - Compression uniaxiale ou triaxiale

Dépendances : numpy, scipy
Sortie       : fichier .inp Abaqus + figure matplotlib (optionnel)

Usage :
    python granite_2d_gbm.py
"""

import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree
import os
import sys
import time

# =================================================================
# 1. PARAMÈTRES
# =================================================================

WORKDIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "granite_2d")
BASENAME = "granite_2d_gbm"

# -- Géométrie du spécimen rectangulaire --
SPEC_W = 50.0   # mm (largeur)
SPEC_H = 100.0  # mm (hauteur)

# -- Grains Voronoï --
N_GRAINS  = 80
SEED      = 42

# -- Maillage --
MESH_SIZE = 1.2  # mm — taille cible des éléments

# -- Phases minérales --
PHASE_FRACTIONS = (0.30, 0.50, 0.20)
PHASE_NAMES     = ['Feldspath', 'Quartz', 'Biotite']

# -- Matériaux CDP (valeurs CDP.xlsx) --
MINERALS = [
    dict(name='Feldspath', E=45000.0, nu=0.29, rho=2.69e-9,
         sigc0=120.0, sigt0=5.0, dilat=35.0, dt=0.98),
    dict(name='Quartz',    E=80000.0, nu=0.17, rho=2.69e-9,
         sigc0=285.0, sigt0=10.0, dilat=35.0, dt=0.98),
    dict(name='Biotite',   E=20000.0, nu=0.20, rho=2.89e-9,
         sigc0=125.0, sigt0=7.0, dilat=35.0, dt=0.98),
]

# Paramètres CDP globaux
CDP_ECC  = 0.1
CDP_FBFC = 1.16
CDP_K    = 0.554
CDP_VISC = 5e-4   # régul. visqueuse — ≤10% step time (Simonovski 2013)

# Ratios hardening/softening (identiques à build_model_5MPa.py)
R_F1   = 153.7 / 196.4
R_CRES = 4.27  / 196.4
R_TRES = 0.034 / 8.78
EIN_PIC  = 0.00065
EIN_END  = 0.0094
ETIN_END = 0.00035

# -- Propriétés cohésives des joints de grains --
# Basé sur Zhang 2023 Table 3 + Simonovski 2013
# T0 = 1.0 mm (épaisseur constitutive, cf. Simonovski 2015)
COH_T0 = 1.0

# Joints entre grains de MÊME phase (plus résistants)
COH_SAME = dict(
    Knn=1.0e6, Kss=1.0e6, Ktt=1.0e6,   # MPa — rigidité pénalité
    sigma_n=12.0, sigma_s=20.0,          # MPa — résistances pic
    GfI=0.10, GfII=0.20,                 # N/mm — énergie fracture
)
# Joints entre grains de phase DIFFÉRENTE (plus faibles)
COH_DIFF = dict(
    Knn=1.0e6, Kss=1.0e6, Ktt=1.0e6,
    sigma_n=5.0,  sigma_s=12.0,
    GfI=0.005, GfII=0.05,
)

# -- Chargement --
TOP_DISP    = -2.0   # mm (compression axiale)
CONFINING   = 0.0    # MPa (pression latérale, 0 = uniaxial)
STEP_TIME   = 1.0    # s (temps fictif pour l'analyse quasi-statique)


# =================================================================
# 2. GÉNÉRATION DU MAILLAGE VORONOÏ
# =================================================================

def generate_mesh(width, height, mesh_size, n_grains, seed=42):
    """
    Génère un maillage triangulaire 2D et assigne chaque élément
    à un grain Voronoï (approche nearest-seed).

    Retourne : nodes (N,2), triangles (M,3), grain_ids (M,), seeds (n_grains,2)
    """
    rng = np.random.default_rng(seed)

    # --- Graines Voronoï ---
    seeds = np.column_stack([
        rng.uniform(0, width, n_grains),
        rng.uniform(0, height, n_grains),
    ])

    # --- Points de maillage : grille + perturbation ---
    nx = int(round(width / mesh_size)) + 1
    ny = int(round(height / mesh_size)) + 1
    xs = np.linspace(0, width, nx)
    ys = np.linspace(0, height, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Perturbation aléatoire des nœuds intérieurs (meilleure qualité)
    interior = ((pts[:, 0] > 1e-6) & (pts[:, 0] < width - 1e-6) &
                (pts[:, 1] > 1e-6) & (pts[:, 1] < height - 1e-6))
    pts[interior] += rng.uniform(-mesh_size * 0.2, mesh_size * 0.2,
                                 size=(interior.sum(), 2))

    # --- Triangulation de Delaunay ---
    tri = Delaunay(pts)
    triangles = tri.simplices.copy()

    # Filtrer les triangles dégénérés (aire < seuil)
    def tri_area(p1, p2, p3):
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) -
                         (p3[0]-p1[0])*(p2[1]-p1[1]))

    keep = []
    for i, t in enumerate(triangles):
        a = tri_area(pts[t[0]], pts[t[1]], pts[t[2]])
        if a > 1e-8:
            keep.append(i)
    triangles = triangles[keep]

    # --- Attribution grain : centroïde → graine la plus proche ---
    centroids = pts[triangles].mean(axis=1)
    tree = cKDTree(seeds)
    _, grain_ids = tree.query(centroids)

    print("  Maillage : {} noeuds, {} elements CPE3, {} grains".format(
        len(pts), len(triangles), n_grains))
    return pts, triangles, grain_ids, seeds


# =================================================================
# 3. INSERTION DES ÉLÉMENTS COHÉSIFS (COH2D4)
# =================================================================

def insert_cohesive_elements(nodes, triangles, grain_ids):
    """
    Insère des éléments cohésifs COH2D4 (épaisseur zéro)
    aux joints de grains.

    Algorithme (cf. Simonovski 2013, Section 2) :
      1. Identifier les nœuds aux joints de grains (multi-grain)
      2. Dupliquer ces nœuds : le grain primaire garde l'original,
         les autres grains reçoivent des copies
      3. Mettre à jour la connectivité des triangles
      4. Créer les COH2D4 aux arêtes inter-grain

    Retourne :
      new_nodes (N',2), new_triangles (M,3), coh_elems list of
      (n1_A, n2_A, n2_B, n1_B, grainA, grainB)
    """
    n_nodes_orig = len(nodes)

    # --- Étape 1 : nœud → set de grains qui l'utilisent ---
    node_grains = {}
    for ti, tri in enumerate(triangles):
        gid = grain_ids[ti]
        for ni in tri:
            if ni not in node_grains:
                node_grains[ni] = set()
            node_grains[ni].add(gid)

    # --- Étape 2 : nœuds de joint (partagés par 2+ grains) ---
    boundary_nodes = {ni: sorted(gs)
                      for ni, gs in node_grains.items() if len(gs) > 1}

    # Duplication : grain[0] garde l'original, les autres reçoivent des copies
    extra_nodes = []
    # (node_original, grain_id) → node_index effectif
    node_grain_map = {}

    for ni, grains_list in boundary_nodes.items():
        # Le premier grain garde l'original
        node_grain_map[(ni, grains_list[0])] = ni
        # Les autres reçoivent des copies
        for g in grains_list[1:]:
            new_idx = n_nodes_orig + len(extra_nodes)
            extra_nodes.append(nodes[ni].copy())
            node_grain_map[(ni, g)] = new_idx

    new_nodes = np.vstack([nodes, np.array(extra_nodes)]) if extra_nodes else nodes.copy()

    # --- Étape 3 : MAJ connectivité des triangles ---
    new_triangles = triangles.copy()
    for ti in range(len(new_triangles)):
        gid = grain_ids[ti]
        for k in range(3):
            ni = triangles[ti][k]
            key = (ni, gid)
            if key in node_grain_map:
                new_triangles[ti][k] = node_grain_map[key]

    # --- Étape 4 : trouver les arêtes inter-grain (maillage ORIGINAL) ---
    edge_tris = {}
    for ti, tri in enumerate(triangles):
        for j in range(3):
            n1, n2 = sorted([tri[j], tri[(j + 1) % 3]])
            edge = (n1, n2)
            if edge not in edge_tris:
                edge_tris[edge] = []
            edge_tris[edge].append(ti)

    # --- Étape 5 : créer les COH2D4 ---
    coh_elems = []
    for (n1_orig, n2_orig), tri_list in edge_tris.items():
        if len(tri_list) != 2:
            continue  # arête frontière (1 seul triangle)
        t1, t2 = tri_list
        gA, gB = grain_ids[t1], grain_ids[t2]
        if gA == gB:
            continue  # arête interne au grain

        # Nœuds effectifs pour chaque grain
        n1_A = node_grain_map.get((n1_orig, gA), n1_orig)
        n2_A = node_grain_map.get((n2_orig, gA), n2_orig)
        n1_B = node_grain_map.get((n1_orig, gB), n1_orig)
        n2_B = node_grain_map.get((n2_orig, gB), n2_orig)

        # COH2D4 : (bottom1, bottom2, top2, top1)
        # bottom = côté grain A, top = côté grain B
        coh_elems.append((n1_A, n2_A, n2_B, n1_B, gA, gB))

    print("  Cohésif : {} nœuds dupliqués, {} éléments COH2D4".format(
        len(extra_nodes), len(coh_elems)))
    return new_nodes, new_triangles, coh_elems


# =================================================================
# 4. ATTRIBUTION DES PHASES MINÉRALES
# =================================================================

def assign_phases(n_grains, fractions, phase_names, seed=42):
    """
    Attribution aléatoire des phases en respectant les fractions
    volumiques cibles.

    Retourne : dict grain_id → phase_name
    """
    rng = np.random.default_rng(seed)
    counts = np.round(np.array(fractions) * n_grains).astype(int)
    counts[-1] = n_grains - counts[:-1].sum()

    labels = np.concatenate([np.full(c, p) for p, c in zip(phase_names, counts)])
    rng.shuffle(labels)

    grain_phase = {i: labels[i] for i in range(n_grains)}
    for p in phase_names:
        n = sum(1 for v in grain_phase.values() if v == p)
        print("  {} : {} grains ({:.0f}%)".format(p, n, 100 * n / n_grains))
    return grain_phase


# =================================================================
# 5. ÉCRITURE DU FICHIER .INP ABAQUS
# =================================================================

def write_inp(filepath, nodes, triangles, grain_ids, coh_elems,
              grain_phase, minerals, coh_same, coh_diff,
              spec_w, spec_h, top_disp, confining, step_time):
    """
    Écrit un fichier .inp complet pour Abaqus/Standard 2D plane strain.
    """
    print("[INP] Écriture de {}...".format(filepath))

    phase_names = sorted(set(grain_phase.values()))
    n_tri = len(triangles)
    n_coh = len(coh_elems)

    # --- Pré-calcul des elsets ---
    # Par phase (triangles)
    elset_phase = {p: [] for p in phase_names}
    for ti in range(n_tri):
        gid = grain_ids[ti]
        phase = grain_phase.get(gid, phase_names[0])
        elset_phase[phase].append(ti + 1)  # 1-based

    # Cohésifs : même phase vs phase différente
    elset_coh_same = []
    elset_coh_diff = []
    for ci, (_, _, _, _, gA, gB) in enumerate(coh_elems):
        eid = n_tri + ci + 1  # 1-based, après les triangles
        pA = grain_phase.get(gA, phase_names[0])
        pB = grain_phase.get(gB, phase_names[0])
        if pA == pB:
            elset_coh_same.append(eid)
        else:
            elset_coh_diff.append(eid)

    # Node sets pour BCs
    tol = 1e-4
    nset_bot = [i + 1 for i, n in enumerate(nodes) if n[1] < tol]
    nset_top = [i + 1 for i, n in enumerate(nodes) if abs(n[1] - spec_h) < tol]
    # Nœud coin bas-gauche pour empêcher translation en x
    corner_node = None
    best_dist = None
    for i, n in enumerate(nodes):
        if n[1] < tol:
            d = abs(n[0])
            if best_dist is None or d < best_dist:
                corner_node = i + 1
                best_dist = d

    # Nsets latéraux pour pression de confinement
    nset_left  = [i + 1 for i, n in enumerate(nodes) if n[0] < tol]
    nset_right = [i + 1 for i, n in enumerate(nodes) if abs(n[0] - spec_w) < tol]

    with open(filepath, 'w') as f:
        # ── HEADING ──
        f.write("** ===================================================\n")
        f.write("** Granite 2D GBM — Voronoi + Cohesive Grain Boundaries\n")
        f.write("** Zhang 2023 / Simonovski 2013, 2015\n")
        f.write("** Generated by granite_2d_gbm.py\n")
        f.write("** ===================================================\n")
        f.write("*HEADING\n")
        f.write("Granite 2D Grain-Based Model with Cohesive Elements\n")

        # ── NODES ──
        f.write("**\n*NODE\n")
        for i, (x, y) in enumerate(nodes):
            f.write("{}, {:.8f}, {:.8f}\n".format(i + 1, x, y))

        # ── ELEMENTS CPE3 (grains) ──
        f.write("**\n*ELEMENT, TYPE=CPE3, ELSET=ALL_GRAINS\n")
        for i, (n1, n2, n3) in enumerate(triangles):
            f.write("{}, {}, {}, {}\n".format(i + 1, n1 + 1, n2 + 1, n3 + 1))

        # ── ELEMENTS COH2D4 (joints de grains) ──
        if n_coh > 0:
            f.write("**\n*ELEMENT, TYPE=COH2D4, ELSET=ALL_COH\n")
            for ci, (nA1, nA2, nB2, nB1, _, _) in enumerate(coh_elems):
                eid = n_tri + ci + 1
                f.write("{}, {}, {}, {}, {}\n".format(
                    eid, nA1 + 1, nA2 + 1, nB2 + 1, nB1 + 1))

        # ── ELSETS par phase ──
        for phase, eids in elset_phase.items():
            f.write("**\n*ELSET, ELSET=ELSET_{}\n".format(phase.upper()))
            _write_id_list(f, eids)

        if elset_coh_same:
            f.write("**\n*ELSET, ELSET=ELSET_COH_SAME\n")
            _write_id_list(f, elset_coh_same)
        if elset_coh_diff:
            f.write("**\n*ELSET, ELSET=ELSET_COH_DIFF\n")
            _write_id_list(f, elset_coh_diff)

        # ── NSETS pour BCs ──
        f.write("**\n*NSET, NSET=NSET_BOT\n")
        _write_id_list(f, nset_bot)
        f.write("*NSET, NSET=NSET_TOP\n")
        _write_id_list(f, nset_top)
        if corner_node:
            f.write("*NSET, NSET=NSET_CORNER\n")
            f.write("{}\n".format(corner_node))
        if confining > 0:
            f.write("*NSET, NSET=NSET_LEFT\n")
            _write_id_list(f, nset_left)
            f.write("*NSET, NSET=NSET_RIGHT\n")
            _write_id_list(f, nset_right)

        # ── MATÉRIAUX CDP (grains) ──
        for m in minerals:
            f.write("**\n** Material: {}\n".format(m['name']))
            f.write("*MATERIAL, NAME={}\n".format(m['name']))
            f.write("*DENSITY\n{},\n".format(m['rho']))
            f.write("*ELASTIC\n{}, {}\n".format(m['E'], m['nu']))

            dilat = m.get('dilat', 35.0)
            f.write("*CONCRETE DAMAGED PLASTICITY\n")
            f.write("{}, {}, {}, {}, {}\n".format(
                dilat, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC))

            fc0 = m['sigc0']
            ft0 = m['sigt0']
            f1   = R_F1 * fc0
            cres = R_CRES * fc0
            tres = max(0.01 * ft0, R_TRES * ft0)

            f.write("*CONCRETE COMPRESSION HARDENING\n")
            f.write("{:.4f}, 0.0\n".format(f1))
            f.write("{:.4f}, {}\n".format(fc0, EIN_PIC))
            f.write("{:.4f}, {}\n".format(cres, EIN_END))

            f.write("*CONCRETE TENSION STIFFENING\n")
            f.write("{:.4f}, 0.0\n".format(ft0))
            f.write("{:.6f}, {}\n".format(tres, ETIN_END))

            f.write("*CONCRETE COMPRESSION DAMAGE\n")
            f.write("0.0, 0.0\n")

            dt = m.get('dt', 0.98)
            f.write("*CONCRETE TENSION DAMAGE\n")
            f.write("0.0, 0.0\n")
            f.write("{}, {}\n".format(dt, ETIN_END))

        # ── MATÉRIAUX COHÉSIFS (joints de grains) ──
        for label, props in [('Coh_Same', coh_same), ('Coh_Diff', coh_diff)]:
            f.write("**\n** Material: {} (traction-separation)\n".format(label))
            f.write("*MATERIAL, NAME={}\n".format(label))
            f.write("*ELASTIC, TYPE=TRACTION\n")
            f.write("{}, {}, {}\n".format(
                props['Knn'], props['Kss'], props['Ktt']))
            f.write("*DAMAGE INITIATION, CRITERION=MAXS\n")
            f.write("{}, {}, {}\n".format(
                props['sigma_n'], props['sigma_s'], props['sigma_s']))
            f.write("*DAMAGE EVOLUTION, TYPE=ENERGY, "
                    "MIXED MODE BEHAVIOR=BK, POWER=1.0\n")
            f.write("{}, {}, {}\n".format(
                props['GfI'], props['GfII'], props['GfII']))

        # ── SECTIONS SOLIDES (grains) ──
        for phase in phase_names:
            f.write("**\n*SOLID SECTION, ELSET=ELSET_{}, MATERIAL={}\n".format(
                phase.upper(), phase))
            f.write("1.0,\n")  # épaisseur out-of-plane

        # ── SECTIONS COHÉSIVES ──
        if elset_coh_same:
            f.write("**\n*COHESIVE SECTION, ELSET=ELSET_COH_SAME, "
                    "MATERIAL=Coh_Same, RESPONSE=TRACTION SEPARATION\n")
            f.write("{},\n".format(COH_T0))
        if elset_coh_diff:
            f.write("*COHESIVE SECTION, ELSET=ELSET_COH_DIFF, "
                    "MATERIAL=Coh_Diff, RESPONSE=TRACTION SEPARATION\n")
            f.write("{},\n".format(COH_T0))

        # ── STEP 1 : Compression ──
        f.write("**\n** ===== STEP 1 : Compression =====\n")
        f.write("*STEP, NAME=Compress, NLGEOM=YES, INC=10000\n")
        f.write("*STATIC, STABILIZE=0.0002\n")
        dt_init = step_time / 100.0
        dt_min  = step_time / 1.0e6
        dt_max  = step_time / 10.0
        f.write("{}, {}, {}, {}\n".format(dt_init, step_time, dt_min, dt_max))

        # BCs
        f.write("**\n*BOUNDARY\n")
        f.write("NSET_BOT, 2, 2, 0.0\n")        # u2 = 0 en bas
        f.write("NSET_CORNER, 1, 1, 0.0\n")      # u1 = 0 coin
        f.write("NSET_TOP, 2, 2, {}\n".format(top_disp))  # déplacement imposé

        # Pression de confinement latérale (si triaxial)
        if confining > 0:
            # Surfaces latérales — on utilise *DSLOAD sur les arêtes
            # Pour un modèle flat, on identifie les faces latérales
            f.write("**\n** Confinement lateral : {} MPa\n".format(confining))
            # On applique la pression via *DLOAD sur les éléments latéraux
            # Alternative simple : boundary pressure via *CLOAD
            # (à adapter selon le modèle réel)

        # Sorties
        f.write("**\n*OUTPUT, FIELD, FREQUENCY=10\n")
        f.write("*NODE OUTPUT\n")
        f.write("U, RF\n")
        f.write("*ELEMENT OUTPUT, ELSET=ALL_GRAINS\n")
        f.write("S, E, PE, PEEQ, SDEG, DAMAGEC, DAMAGET, STATUS\n")
        if n_coh > 0:
            if elset_coh_same:
                f.write("*ELEMENT OUTPUT, ELSET=ELSET_COH_SAME\n")
                f.write("S, E, SDEG, STATUS\n")
            if elset_coh_diff:
                f.write("*ELEMENT OUTPUT, ELSET=ELSET_COH_DIFF\n")
                f.write("S, E, SDEG, STATUS\n")

        f.write("**\n*OUTPUT, HISTORY, FREQUENCY=1\n")
        f.write("*NODE OUTPUT, NSET=NSET_TOP\n")
        f.write("RF2, U2\n")

        f.write("**\n*END STEP\n")

    print("  -> {} écrit ({} CPE3 + {} COH2D4)".format(filepath, n_tri, n_coh))
    return filepath


def _write_id_list(f, ids, per_line=16):
    """Écrit une liste d'IDs au format Abaqus (max 16 par ligne)."""
    buf = []
    for eid in ids:
        buf.append(str(eid))
        if len(buf) == per_line:
            f.write(", ".join(buf) + ",\n")
            buf = []
    if buf:
        f.write(", ".join(buf) + "\n")


# =================================================================
# 6. VISUALISATION (matplotlib)
# =================================================================

def visualize(nodes, triangles, grain_ids, grain_phase, coh_elems,
              seeds, outdir):
    """
    Génère 2 figures :
    1. Microstructure (phases minérales + joints de grains)
    2. Maillage avec éléments cohésifs mis en évidence
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.collections import LineCollection
    except ImportError:
        print("[Viz] matplotlib non disponible, pas de figure.")
        return

    phase_colors = {
        'Feldspath': '#E8A87C',
        'Quartz':    '#D4E157',
        'Biotite':   '#607D8B',
    }

    # --- Figure 1 : Phases minérales ---
    fig1, ax1 = plt.subplots(figsize=(5, 10))

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)

    # Map phases to numeric IDs for tripcolor
    phase_to_id = {'Feldspath': 0, 'Quartz': 1, 'Biotite': 2}
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([phase_colors['Feldspath'],
                           phase_colors['Quartz'],
                           phase_colors['Biotite']])
    facevals = np.array([phase_to_id.get(grain_phase.get(grain_ids[ti], 'Feldspath'), 0)
                         for ti in range(len(triangles))], dtype=float)
    ax1.tripcolor(triang, facecolors=facevals, cmap=cmap,
                  vmin=-0.5, vmax=2.5, edgecolors='none')

    # Dessiner les joints de grains (arêtes cohésives)
    coh_lines = []
    for nA1, nA2, _, _, _, _ in coh_elems:
        p1 = nodes[nA1]
        p2 = nodes[nA2]
        coh_lines.append([p1, p2])
    if coh_lines:
        lc = LineCollection(coh_lines, colors='black', linewidths=0.3, alpha=0.6)
        ax1.add_collection(lc)

    # Légende
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=phase_colors[p], label=p)
                      for p in ['Feldspath', 'Quartz', 'Biotite']]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # Graines
    ax1.plot(seeds[:, 0], seeds[:, 1], 'k.', ms=1, alpha=0.3)

    ax1.set_xlim(0, nodes[:, 0].max())
    ax1.set_ylim(0, nodes[:, 1].max())
    ax1.set_aspect('equal')
    ax1.set_title("Microstructure 2D — Phases minérales")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    plt.tight_layout()
    fig1_path = os.path.join(outdir, "microstructure_2d.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> {} sauvegardée".format(fig1_path))

    # --- Figure 2 : Maillage + cohésifs ---
    fig2, ax2 = plt.subplots(figsize=(5, 10))

    # Triangles en gris clair
    ax2.triplot(triang, 'k-', lw=0.1, alpha=0.3)

    # Cohésifs colorés par type
    coh_same_lines = []
    coh_diff_lines = []
    for nA1, nA2, _, _, gA, gB in coh_elems:
        p1 = nodes[nA1]
        p2 = nodes[nA2]
        pA = grain_phase.get(gA, 'X')
        pB = grain_phase.get(gB, 'X')
        if pA == pB:
            coh_same_lines.append([p1, p2])
        else:
            coh_diff_lines.append([p1, p2])

    if coh_same_lines:
        lc1 = LineCollection(coh_same_lines, colors='blue',
                             linewidths=0.8, alpha=0.7, label='Same-phase GB')
        ax2.add_collection(lc1)
    if coh_diff_lines:
        lc2 = LineCollection(coh_diff_lines, colors='red',
                             linewidths=0.8, alpha=0.7, label='Diff-phase GB')
        ax2.add_collection(lc2)

    ax2.set_xlim(0, nodes[:, 0].max())
    ax2.set_ylim(0, nodes[:, 1].max())
    ax2.set_aspect('equal')
    ax2.set_title("Maillage + éléments cohésifs")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2_path = os.path.join(outdir, "mesh_cohesive_2d.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> {} sauvegardée".format(fig2_path))


# =================================================================
# MAIN
# =================================================================

def main():
    t0 = time.time()
    os.makedirs(WORKDIR, exist_ok=True)

    print("=" * 60)
    print("  granite_2d_gbm.py — Démarrage")
    print("  {} grains | maille ~{} mm | {}×{} mm".format(
        N_GRAINS, MESH_SIZE, SPEC_W, SPEC_H))
    print("=" * 60)

    # 1. Maillage Voronoï
    print("\n[1/5] Génération du maillage...")
    nodes, triangles, grain_ids, seeds = generate_mesh(
        SPEC_W, SPEC_H, MESH_SIZE, N_GRAINS, SEED)

    # 2. Insertion des éléments cohésifs
    print("\n[2/5] Insertion des éléments cohésifs...")
    nodes, triangles, coh_elems = insert_cohesive_elements(
        nodes, triangles, grain_ids)

    # 3. Attribution des phases
    print("\n[3/5] Attribution des phases minérales...")
    grain_phase = assign_phases(N_GRAINS, PHASE_FRACTIONS, PHASE_NAMES, SEED)

    # 4. Écriture .inp
    print("\n[4/5] Écriture du fichier .inp...")
    inp_path = os.path.join(WORKDIR, BASENAME + ".inp")
    write_inp(inp_path, nodes, triangles, grain_ids, coh_elems,
              grain_phase, MINERALS, COH_SAME, COH_DIFF,
              SPEC_W, SPEC_H, TOP_DISP, CONFINING, STEP_TIME)

    # 5. Visualisation
    print("\n[5/5] Visualisation...")
    visualize(nodes, triangles, grain_ids, grain_phase,
              coh_elems, seeds, WORKDIR)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  Terminé en {:.1f}s".format(elapsed))
    print("  Fichier .inp : {}".format(inp_path))
    print("  Figures      : {}".format(WORKDIR))
    print("=" * 60)


if __name__ == "__main__":
    main()

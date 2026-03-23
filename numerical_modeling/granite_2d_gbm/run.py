# -*- coding: utf-8 -*-
"""
granite_2d_neper.py  (v3)
=========================
Lit le maillage Neper 2D (.msh), insere des elements cohesifs COH2D4
aux joints de grains, et exporte un .inp Abaqus complet.

FIX v3 :
  - Detection automatique du plan 2D (Neper met Y=0, coords dans XZ)
  - Verification des aires AVANT ecriture
  - Format flat (pas de *Part/*Assembly)
  - THICKNESS=SPECIFIED pour cohesive sections

Prerequis :
  pip install meshio numpy matplotlib

Neper :
  neper -T -n 80 -id 1 -dim 2 -domain "square(50,100)" -morpho graingrowth -reg 1 -o granite_2d
  neper -M granite_2d.tess -cl 1.0 -order 1 -format msh -o granite_2d_mesh

Usage :
  python granite_2d_neper.py

Puis soumettre DIRECTEMENT (ne PAS importer dans CAE) :
  abaqus job=granite_2d input=granite_2d_cohesive cpus=4
"""

import numpy as np
import meshio
import os
import sys
import time


# =================================================================
# PARAMETRES
# =================================================================

MESH_FILE  = r"granite_2d_gbm\granite_2d_mesh.msh"
OUTPUT_INP = "granite_2d_cohesive.inp"

SPEC_W = 50.0
SPEC_H = 100.0

PHASE_FRACTIONS = (0.30, 0.50, 0.20)
PHASE_NAMES     = ['Feldspath', 'Quartz', 'Biotite']
SEED            = 42

MINERALS = [
    dict(name='Feldspath', E=45000.0, nu=0.29, rho=2.69e-9,
         sigc0=120.0, sigt0=5.0, dilat=35.0, dt=0.98),
    dict(name='Quartz',    E=80000.0, nu=0.17, rho=2.69e-9,
         sigc0=285.0, sigt0=10.0, dilat=35.0, dt=0.98),
    dict(name='Biotite',   E=20000.0, nu=0.20, rho=2.89e-9,
         sigc0=125.0, sigt0=7.0, dilat=35.0, dt=0.98),
]

CDP_ECC  = 0.1
CDP_FBFC = 1.16
CDP_K    = 0.554
CDP_VISC = 5e-4

R_F1     = 153.7 / 196.4
R_CRES   = 4.27  / 196.4
R_TRES   = 0.034 / 8.78
EIN_PIC  = 0.00065
EIN_END  = 0.0094
ETIN_END = 0.00035

COH_T0 = 1.0

COH_SAME = dict(Knn=1.0e6, Kss=1.0e6, Ktt=1.0e6,
                sigma_n=12.0, sigma_s=20.0,
                GfI=0.10, GfII=0.20)
COH_DIFF = dict(Knn=1.0e6, Kss=1.0e6, Ktt=1.0e6,
                sigma_n=5.0,  sigma_s=12.0,
                GfI=0.005, GfII=0.05)

TOP_DISP  = -2.0
STEP_TIME = 1.0


# =================================================================
# ETAPE 3 : Lecture du maillage Neper
# =================================================================

def read_neper_mesh(mesh_file):
    print("[3/6] Lecture de {}...".format(mesh_file))
    mesh = meshio.read(mesh_file)

    print("  Types de cellules : {}".format([b.type for b in mesh.cells]))
    print("  Cles cell_data    : {}".format(list(mesh.cell_data.keys())))

    pts = mesh.points
    print("  mesh.points shape : {}".format(pts.shape))
    for col in range(pts.shape[1]):
        cmin = pts[:, col].min()
        cmax = pts[:, col].max()
        print("    Col {} : min={:.4f}  max={:.4f}  range={:.4f}".format(
            col, cmin, cmax, cmax - cmin))

    # -----------------------------------------------------------
    # FIX : Neper 2D ecrit 3 colonnes (x, y, z) avec y=0 partout
    # On detecte automatiquement les 2 colonnes utiles
    # -----------------------------------------------------------
    if pts.shape[1] == 3:
        ranges = [pts[:, c].max() - pts[:, c].min() for c in range(3)]
        cols_sorted = sorted(range(3), key=lambda c: ranges[c], reverse=True)
        col_a, col_b = sorted(cols_sorted[:2])
        col_flat = cols_sorted[2]
        print("  >>> Plan detecte : colonnes {} et {} "
              "(col {} plate, range={:.6f})".format(
                  col_a, col_b, col_flat, ranges[col_flat]))
        nodes = pts[:, [col_a, col_b]].copy()
    elif pts.shape[1] == 2:
        nodes = pts.copy()
        print("  >>> Coordonnees 2D directes")
    else:
        raise ValueError("Format inattendu : shape={}".format(pts.shape))

    print("  Noeuds 2D : x=[{:.2f}, {:.2f}]  y=[{:.2f}, {:.2f}]".format(
        nodes[:, 0].min(), nodes[:, 0].max(),
        nodes[:, 1].min(), nodes[:, 1].max()))

    # --- Triangles ---
    TYPES = ("triangle", "triangle3", "tri", "tri3")
    TAG_KEYS = ("gmsh:physical", "medit:ref", "cell_tags")
    tag_key = next((k for k in TAG_KEYS if k in mesh.cell_data), None)

    triangles = None
    grain_ids = None

    for i, blk in enumerate(mesh.cells):
        if blk.type in TYPES:
            triangles = blk.data
            if tag_key and i < len(mesh.cell_data[tag_key]):
                grain_ids = mesh.cell_data[tag_key][i]
            break

    if triangles is None:
        raise ValueError("Aucun triangle. Types : {}".format(
            [b.type for b in mesh.cells]))

    if grain_ids is None:
        print("  ATTENTION : pas de tag grain")
        grain_ids = np.ones(len(triangles), dtype=int)

    # --- Verification des aires ---
    p0 = nodes[triangles[:, 0]]
    p1 = nodes[triangles[:, 1]]
    p2 = nodes[triangles[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
        (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

    n_zero = np.sum(areas < 1e-12)
    print("  Aires : min={:.6f}  max={:.4f}  mean={:.4f}".format(
        areas.min(), areas.max(), areas.mean()))

    if n_zero > 0:
        print("  *** ERREUR FATALE : {} triangles avec aire ~0 ***".format(n_zero))
        sys.exit(1)
    else:
        print("  OK : toutes les aires sont positives")

    n_grains = len(np.unique(grain_ids))
    print("  -> {} noeuds, {} triangles, {} grains".format(
        len(nodes), len(triangles), n_grains))

    return nodes, triangles, grain_ids


# =================================================================
# ETAPE 3b : Attribution des phases
# =================================================================

def assign_phases(grain_ids, fractions, phase_names, seed=42):
    print("\n[3b/6] Attribution des phases minerales...")
    rng = np.random.default_rng(seed)
    unique_grains = np.unique(grain_ids)
    n_grains = len(unique_grains)

    counts = np.round(np.array(fractions) * n_grains).astype(int)
    counts[-1] = n_grains - counts[:-1].sum()

    labels = np.concatenate([np.full(c, p)
                             for p, c in zip(phase_names, counts)])
    rng.shuffle(labels)

    grain_phase = {}
    for i, gid in enumerate(unique_grains):
        grain_phase[gid] = labels[i]

    for p in phase_names:
        n = sum(1 for v in grain_phase.values() if v == p)
        print("  {} : {} grains ({:.0f}%)".format(p, n, 100 * n / n_grains))

    return grain_phase


# =================================================================
# ETAPE 4 : Duplication des noeuds aux joints de grains
# =================================================================

def duplicate_boundary_nodes(nodes, triangles, grain_ids):
    print("\n[4/6] Duplication des noeuds aux joints...")

    n_orig = len(nodes)

    node_grains = {}
    for ti, tri in enumerate(triangles):
        gid = grain_ids[ti]
        for ni in tri:
            if ni not in node_grains:
                node_grains[ni] = set()
            node_grains[ni].add(gid)

    boundary_nodes = {ni: sorted(gs)
                      for ni, gs in node_grains.items() if len(gs) > 1}

    extra_nodes = []
    node_grain_map = {}

    for ni, grains_list in boundary_nodes.items():
        node_grain_map[(ni, grains_list[0])] = ni
        for g in grains_list[1:]:
            new_idx = n_orig + len(extra_nodes)
            extra_nodes.append(nodes[ni].copy())
            node_grain_map[(ni, g)] = new_idx

    if extra_nodes:
        new_nodes = np.vstack([nodes, np.array(extra_nodes)])
    else:
        new_nodes = nodes.copy()

    new_triangles = triangles.copy()
    for ti in range(len(new_triangles)):
        gid = grain_ids[ti]
        for k in range(3):
            ni = triangles[ti][k]
            key = (ni, gid)
            if key in node_grain_map:
                new_triangles[ti][k] = node_grain_map[key]

    print("  {} noeuds de joint".format(len(boundary_nodes)))
    print("  {} noeuds dupliques".format(len(extra_nodes)))
    print("  {} noeuds total (avant: {})".format(len(new_nodes), n_orig))

    return new_nodes, new_triangles, node_grain_map


# =================================================================
# ETAPE 5 : Creation des elements COH2D4
# =================================================================

def create_cohesive_elements(triangles_orig, grain_ids, node_grain_map):
    print("\n[5/6] Creation des COH2D4...")

    edge_tris = {}
    for ti, tri in enumerate(triangles_orig):
        for j in range(3):
            n1, n2 = sorted([tri[j], tri[(j + 1) % 3]])
            edge = (n1, n2)
            if edge not in edge_tris:
                edge_tris[edge] = []
            edge_tris[edge].append(ti)

    coh_elems = []
    for (n1_orig, n2_orig), tri_list in edge_tris.items():
        if len(tri_list) != 2:
            continue
        t1, t2 = tri_list
        gA, gB = grain_ids[t1], grain_ids[t2]
        if gA == gB:
            continue

        n1_A = node_grain_map.get((n1_orig, gA), n1_orig)
        n2_A = node_grain_map.get((n2_orig, gA), n2_orig)
        n1_B = node_grain_map.get((n1_orig, gB), n1_orig)
        n2_B = node_grain_map.get((n2_orig, gB), n2_orig)

        coh_elems.append((n1_A, n2_A, n2_B, n1_B, gA, gB))

    print("  {} COH2D4 crees".format(len(coh_elems)))
    return coh_elems


# =================================================================
# ETAPE 6 : Ecriture du .inp (format FLAT)
# =================================================================

def write_inp(filepath, nodes, triangles, grain_ids, coh_elems,
              grain_phase, minerals, coh_same, coh_diff,
              spec_w, spec_h, top_disp, step_time):

    print("\n[6/6] Ecriture .inp (format flat)...")

    phase_names = sorted(set(grain_phase.values()))
    n_tri = len(triangles)
    n_coh = len(coh_elems)

    elset_phase = {p: [] for p in phase_names}
    for ti in range(n_tri):
        gid = grain_ids[ti]
        phase = grain_phase.get(gid, phase_names[0])
        elset_phase[phase].append(ti + 1)

    elset_coh_same = []
    elset_coh_diff = []
    for ci, (_, _, _, _, gA, gB) in enumerate(coh_elems):
        eid = n_tri + ci + 1
        pA = grain_phase.get(gA, phase_names[0])
        pB = grain_phase.get(gB, phase_names[0])
        if pA == pB:
            elset_coh_same.append(eid)
        else:
            elset_coh_diff.append(eid)

    print("  COH same-phase : {}".format(len(elset_coh_same)))
    print("  COH diff-phase : {}".format(len(elset_coh_diff)))

    tol = 1e-4
    nset_bot = [i + 1 for i, n in enumerate(nodes) if n[1] < tol]
    nset_top = [i + 1 for i, n in enumerate(nodes)
                if abs(n[1] - spec_h) < tol]

    corner_node = None
    best_dist = None
    for i, n in enumerate(nodes):
        if n[1] < tol:
            d = abs(n[0])
            if best_dist is None or d < best_dist:
                corner_node = i + 1
                best_dist = d

    print("  NSET_BOT : {} noeuds".format(len(nset_bot)))
    print("  NSET_TOP : {} noeuds".format(len(nset_top)))
    print("  CORNER   : noeud {}".format(corner_node))

    if len(nset_bot) == 0 or len(nset_top) == 0:
        print("  *** ATTENTION : nset vide ! Verifier SPEC_W/SPEC_H ***")

    with open(filepath, 'w') as f:

        f.write("*HEADING\n")
        f.write("Granite 2D GBM - {} CPE3 + {} COH2D4 (Neper)\n".format(
            n_tri, n_coh))

        f.write("**\n*NODE\n")
        for i, (x, y) in enumerate(nodes):
            f.write("{}, {:.8f}, {:.8f}\n".format(i + 1, x, y))

        f.write("**\n*ELEMENT, TYPE=CPE3, ELSET=ALL_GRAINS\n")
        for i, (n1, n2, n3) in enumerate(triangles):
            f.write("{}, {}, {}, {}\n".format(
                i + 1, n1 + 1, n2 + 1, n3 + 1))

        if n_coh > 0:
            f.write("**\n*ELEMENT, TYPE=COH2D4, ELSET=ALL_COH\n")
            for ci, (nA1, nA2, nB2, nB1, _, _) in enumerate(coh_elems):
                eid = n_tri + ci + 1
                f.write("{}, {}, {}, {}, {}\n".format(
                    eid, nA1 + 1, nA2 + 1, nB2 + 1, nB1 + 1))

        for phase, eids in elset_phase.items():
            f.write("**\n*ELSET, ELSET=ELSET_{}\n".format(phase.upper()))
            _write_ids(f, eids)

        if elset_coh_same:
            f.write("**\n*ELSET, ELSET=ELSET_COH_SAME\n")
            _write_ids(f, elset_coh_same)
        if elset_coh_diff:
            f.write("**\n*ELSET, ELSET=ELSET_COH_DIFF\n")
            _write_ids(f, elset_coh_diff)

        f.write("**\n*NSET, NSET=NSET_BOT\n")
        _write_ids(f, nset_bot)
        f.write("*NSET, NSET=NSET_TOP\n")
        _write_ids(f, nset_top)
        if corner_node:
            f.write("*NSET, NSET=NSET_CORNER\n")
            f.write("{}\n".format(corner_node))

        for m in minerals:
            f.write("**\n*MATERIAL, NAME={}\n".format(m['name']))
            f.write("*DENSITY\n{},\n".format(m['rho']))
            f.write("*ELASTIC\n{}, {}\n".format(m['E'], m['nu']))
            f.write("*CONCRETE DAMAGED PLASTICITY\n")
            f.write("{}, {}, {}, {}, {}\n".format(
                m['dilat'], CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC))

            fc0  = m['sigc0']
            ft0  = m['sigt0']
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
            f.write("*CONCRETE TENSION DAMAGE\n")
            f.write("0.0, 0.0\n")
            f.write("{}, {}\n".format(m['dt'], ETIN_END))

        for label, props in [('Coh_Same', coh_same), ('Coh_Diff', coh_diff)]:
            f.write("**\n*MATERIAL, NAME={}\n".format(label))
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

        for phase in phase_names:
            f.write("**\n*SOLID SECTION, ELSET=ELSET_{}, MATERIAL={}\n".format(
                phase.upper(), phase))
            f.write("1.0,\n")

        if elset_coh_same:
            f.write("**\n*COHESIVE SECTION, ELSET=ELSET_COH_SAME, "
                    "MATERIAL=Coh_Same, RESPONSE=TRACTION SEPARATION, "
                    "THICKNESS=SPECIFIED\n")
            f.write("{},\n".format(COH_T0))
        if elset_coh_diff:
            f.write("**\n*COHESIVE SECTION, ELSET=ELSET_COH_DIFF, "
                    "MATERIAL=Coh_Diff, RESPONSE=TRACTION SEPARATION, "
                    "THICKNESS=SPECIFIED\n")
            f.write("{},\n".format(COH_T0))

        f.write("**\n*STEP, NAME=Compress, NLGEOM=YES, INC=10000\n")
        f.write("*STATIC, STABILIZE=0.0002\n")
        dt_init = step_time / 100.0
        dt_min  = step_time / 1.0e6
        dt_max  = step_time / 10.0
        f.write("{}, {}, {}, {}\n".format(dt_init, step_time, dt_min, dt_max))

        f.write("**\n*BOUNDARY\n")
        f.write("NSET_BOT, 2, 2, 0.0\n")
        if corner_node:
            f.write("NSET_CORNER, 1, 1, 0.0\n")
        f.write("NSET_TOP, 2, 2, {}\n".format(top_disp))

        f.write("**\n*OUTPUT, FIELD, FREQUENCY=10\n")
        f.write("*NODE OUTPUT\nU, RF\n")
        f.write("*ELEMENT OUTPUT, ELSET=ALL_GRAINS\n")
        f.write("S, E, PE, PEEQ, SDEG, DAMAGEC, DAMAGET, STATUS\n")
        if elset_coh_same:
            f.write("*ELEMENT OUTPUT, ELSET=ELSET_COH_SAME\n")
            f.write("S, E, SDEG, STATUS\n")
        if elset_coh_diff:
            f.write("*ELEMENT OUTPUT, ELSET=ELSET_COH_DIFF\n")
            f.write("S, E, SDEG, STATUS\n")
        f.write("**\n*OUTPUT, HISTORY, FREQUENCY=1\n")
        f.write("*NODE OUTPUT, NSET=NSET_TOP\nRF2, U2\n")
        f.write("**\n*END STEP\n")

    print("\n  -> {} ecrit".format(filepath))
    print("  {} CPE3 + {} COH2D4 = {} elements".format(
        n_tri, n_coh, n_tri + n_coh))


def _write_ids(f, ids, per_line=16):
    buf = []
    for eid in ids:
        buf.append(str(eid))
        if len(buf) == per_line:
            f.write(", ".join(buf) + ",\n")
            buf = []
    if buf:
        f.write(", ".join(buf) + "\n")


# =================================================================
# VISUALISATION
# =================================================================

def visualize(nodes, triangles, grain_ids, grain_phase, coh_elems):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Patch
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("[Viz] matplotlib non disponible.")
        return

    colors = {'Feldspath': '#E8A87C', 'Quartz': '#D4E157',
              'Biotite': '#607D8B'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    pid = {'Biotite': 0, 'Feldspath': 1, 'Quartz': 2}
    cmap = ListedColormap([colors['Biotite'], colors['Feldspath'],
                           colors['Quartz']])
    fv = np.array([pid.get(grain_phase.get(grain_ids[ti], 'Feldspath'), 1)
                   for ti in range(len(triangles))], dtype=float)

    ax1.tripcolor(triang, facecolors=fv, cmap=cmap,
                  vmin=-0.5, vmax=2.5, edgecolors='none')
    ax1.set_aspect('equal')
    ax1.set_title("Phases minerales (Neper)")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend(handles=[Patch(facecolor=colors[p], label=p)
               for p in ['Feldspath', 'Quartz', 'Biotite']],
               loc='upper right', fontsize=8)

    ax2.triplot(triang, 'k-', lw=0.1, alpha=0.3)
    same_l, diff_l = [], []
    for nA1, nA2, _, _, gA, gB in coh_elems:
        seg = [nodes[nA1], nodes[nA2]]
        if grain_phase.get(gA) == grain_phase.get(gB):
            same_l.append(seg)
        else:
            diff_l.append(seg)
    if same_l:
        ax2.add_collection(LineCollection(same_l, colors='blue', lw=0.8,
                                          alpha=0.7, label='Same-phase GB'))
    if diff_l:
        ax2.add_collection(LineCollection(diff_l, colors='red', lw=0.8,
                                          alpha=0.7, label='Diff-phase GB'))
    ax2.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax2.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax2.set_aspect('equal')
    ax2.set_title("Maillage + COH2D4")
    ax2.set_xlabel("x [mm]")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.splitext(OUTPUT_INP)[0] + ".png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Figure : {}".format(out))


# =================================================================
# MAIN
# =================================================================

def main():
    t0 = time.time()

    print("=" * 60)
    print("  granite_2d_neper.py  v3")
    print("  Neper 2D mesh -> COH2D4 -> Abaqus .inp")
    print("=" * 60)

    if not os.path.exists(MESH_FILE):
        print("\nERREUR : {} introuvable.".format(MESH_FILE))
        print("Lance d'abord Neper :")
        print("  neper -T -n 80 -id 1 -dim 2 "
              "-domain 'square(50,100)' -morpho graingrowth "
              "-reg 1 -o granite_2d")
        print("  neper -M granite_2d.tess -cl 1.0 "
              "-order 1 -format msh -o granite_2d_mesh")
        sys.exit(1)

    nodes, triangles, grain_ids = read_neper_mesh(MESH_FILE)
    grain_phase = assign_phases(grain_ids, PHASE_FRACTIONS, PHASE_NAMES, SEED)

    triangles_orig = triangles.copy()
    nodes, triangles, ngm = duplicate_boundary_nodes(
        nodes, triangles, grain_ids)

    coh_elems = create_cohesive_elements(triangles_orig, grain_ids, ngm)

    write_inp(OUTPUT_INP, nodes, triangles, grain_ids, coh_elems,
              grain_phase, MINERALS, COH_SAME, COH_DIFF,
              SPEC_W, SPEC_H, TOP_DISP, STEP_TIME)

    print("\n[Viz] Figure de controle...")
    visualize(nodes, triangles, grain_ids, grain_phase, coh_elems)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  Termine en {:.1f}s".format(elapsed))
    print("  .inp : {}".format(os.path.abspath(OUTPUT_INP)))
    print("")
    print("  >>> SOUMETTRE DIRECTEMENT (pas par CAE) :")
    print("  abaqus job=granite_2d input=granite_2d_cohesive cpus=4")
    print("=" * 60)


if __name__ == "__main__":
    main()
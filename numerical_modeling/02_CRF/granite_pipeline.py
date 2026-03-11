"""
Pipeline complet : Microstructure granitique + Champ Aléatoire Corrélé
=======================================================================
1. Génération de la tessellation Voronoï (Neper -T)
2. Maillage tétraédrique (Neper -M)
3. Attribution des phases minérales (quartz / feldspath / biotite)
4. Application d'un CRF log-normal intra-grain (méthode Gutjahr/Le Goc)
5. Visualisation

Prérequis (WSL2) :
    pip install meshio numpy matplotlib scipy
    sudo apt install neper
"""

import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────────────────────

WORKDIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "granite_sim")
BASENAME   = "granite"
N_GRAINS   = 300          # nombre de grains
MESH_CL    = 0.06         # taille de maille caractéristique [mm]
DOMAIN_MM  = 1.0          # domaine cubique [mm]

# Fractions volumiques des phases (doivent sommer à 1)
PHASE_FRACTIONS = {
    "quartz":     0.30,
    "feldspath":  0.55,
    "biotite":    0.15,
}

# Propriétés mécaniques par phase [GPa]
PHASE_PROPS = {
    "quartz":    {"E": 95.0,  "nu": 0.07, "UCS": 1000e6, "color": "#f0e68c"},
    "feldspath": {"E": 70.0,  "nu": 0.25, "UCS":  600e6, "color": "#d2b48c"},
    "biotite":   {"E": 30.0,  "nu": 0.30, "UCS":  200e6, "color": "#708090"},
}

# Paramètres CRF (hétérogénéité intra-grain)
LAMBDA_RF  = 0.15   # longueur de corrélation [mm]
SIGMA_RF   = 0.25   # écart-type du CRF
MU_RF      = 1.0    # moyenne (facteur multiplicatif sur la propriété)

# Grille pour le CRF
NX, NY, NZ = 64, 64, 64


# ─────────────────────────────────────────────────────────────
# 1. NEPER : TESSELLATION & MAILLAGE
# ─────────────────────────────────────────────────────────────

def win_to_wsl(path):
    """Convertit un chemin Windows en chemin WSL (/mnt/c/...)."""
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path  = f"/mnt/{drive}{path[2:]}"
    return path


def run_neper(workdir, basename, n_grains, mesh_cl, domain_mm):
    """Lance Neper -T puis -M via WSL (appelé depuis Windows)."""
    os.makedirs(workdir, exist_ok=True)

    tess_file = os.path.join(workdir, basename + ".tess")
    mesh_file = os.path.join(workdir, basename + "_mesh.msh")

    wsl_workdir  = win_to_wsl(workdir)
    wsl_tess     = win_to_wsl(tess_file)
    wsl_out_base = f"{wsl_workdir}/{basename}"
    wsl_mesh_out = f"{wsl_workdir}/{basename}_mesh"

    # ── Tessellation ──
    if not os.path.exists(tess_file):
        print(f"[Neper] Génération de la tessellation ({n_grains} grains)...")
        bash_cmd_T = (
            f"neper -T -n {n_grains} -id 1 "
            f"-domain 'cube({domain_mm},{domain_mm},{domain_mm})' "
            f"-morpho graingrowth "
            f"-o '{wsl_out_base}'"
        )
        result = subprocess.run(["wsl", "bash", "-c", bash_cmd_T],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Neper -T a échoué :\n{result.stderr}")
        print("  → tessellation générée")
    else:
        print("[Neper] Tessellation déjà présente, on passe.")

    # ── Maillage ──
    if not os.path.exists(mesh_file):
        print(f"[Neper] Maillage (cl={mesh_cl})...")
        bash_cmd_M = (
            f"neper -M '{wsl_tess}' "
            f"-cl {mesh_cl} "
            f"-format msh "
            f"-o '{wsl_mesh_out}'"
        )
        result = subprocess.run(["wsl", "bash", "-c", bash_cmd_M],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Neper -M a échoué :\n{result.stderr}")
        print("  → maillage généré")
    else:
        print("[Neper] Maillage déjà présent, on passe.")

    return tess_file, mesh_file


# ─────────────────────────────────────────────────────────────
# 2. LECTURE DU MAILLAGE
# ─────────────────────────────────────────────────────────────

def load_mesh(mesh_file):
    """Charge le maillage .msh avec meshio."""
    try:
        import meshio
    except ImportError:
        raise ImportError("Installe meshio : pip install meshio")

    print(f"[Mesh] Chargement de {mesh_file}...")
    mesh = meshio.read(mesh_file)

    # Diagnostic : types de cellules disponibles
    print(f"  Types de cellules : {[b.type for b in mesh.cells]}")
    print(f"  Clés cell_data    : {list(mesh.cell_data.keys())}")

    # Noms possibles selon la version GMSH / meshio
    TETRA_TYPES = ("tetra", "tetra4", "tetra10")

    # Clé de tag possible
    TAG_KEYS = ("gmsh:physical", "medit:ref", "cell_tags")

    tetras    = None
    grain_ids = None

    # Cherche la clé de tag disponible
    tag_key = next((k for k in TAG_KEYS if k in mesh.cell_data), None)

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in TETRA_TYPES:
            tetras = cell_block.data
            if tag_key is not None:
                tag_arrays = mesh.cell_data[tag_key]
                if i < len(tag_arrays):
                    grain_ids = tag_arrays[i]
            break

    if tetras is None:
        raise ValueError(
            f"Aucun tétraèdre trouvé. Types présents : {[b.type for b in mesh.cells]}"
        )

    if grain_ids is None:
        print("  Avertissement : pas de tag de grain trouvé, assignation par index.")
        grain_ids = np.zeros(len(tetras), dtype=int)

    nodes = mesh.points
    print(f"  → {nodes.shape[0]} nœuds, {tetras.shape[0]} tétraèdres, "
          f"{len(np.unique(grain_ids))} grains")
    return nodes, tetras, grain_ids


# ─────────────────────────────────────────────────────────────
# 3. ATTRIBUTION DES PHASES MINÉRALES
# ─────────────────────────────────────────────────────────────

def assign_phases(grain_ids, phase_fractions, seed=42):
    """
    Attribue aléatoirement une phase minérale à chaque grain
    en respectant les fractions volumiques cibles.

    Retour
    ------
    elem_phase : array (n_elem,) — nom de la phase pour chaque élément
    grain_phase : dict grain_id -> phase_name
    """
    rng = np.random.default_rng(seed)
    unique_grains = np.unique(grain_ids)
    n_grains      = len(unique_grains)

    phases  = list(phase_fractions.keys())
    fracs   = np.array(list(phase_fractions.values()))
    counts  = np.round(fracs * n_grains).astype(int)
    counts[-1] = n_grains - counts[:-1].sum()   # ajustement pour sommer à n_grains

    labels = np.concatenate([np.full(c, p) for p, c in zip(phases, counts)])
    rng.shuffle(labels)

    grain_phase = dict(zip(unique_grains, labels))
    elem_phase  = np.array([grain_phase[g] for g in grain_ids])

    for phase in phases:
        n = np.sum(elem_phase == phase)
        print(f"  {phase:12s} : {n} éléments ({100*n/len(elem_phase):.1f} %)")

    return elem_phase, grain_phase


# ─────────────────────────────────────────────────────────────
# 4. CHAMP ALÉATOIRE CORRÉLÉ (CRF) INTRA-GRAIN
# ─────────────────────────────────────────────────────────────

def generate_gaussian_crf(nx, ny, nz, lambda_rf, domain_size):
    """Champ gaussien corrélé via méthode spectrale (Gutjahr 1989)."""
    dx = domain_size / nx
    dy = domain_size / ny
    dz = domain_size / nz

    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2

    S      = (lambda_rf**3 * np.pi**1.5) * np.exp(-K2 * lambda_rf**2 / 4.0)
    noise  = (np.random.randn(nx, ny, nz) + 1j * np.random.randn(nx, ny, nz)) / np.sqrt(2)
    field  = np.real(np.fft.ifftn(np.sqrt(S) * noise))
    field  = (field - field.mean()) / (field.std() + 1e-12)
    return field


def gaussian_to_lognormal(field_gauss, mu_rf, sigma_rf):
    """Transformation log-normale préservant la structure de corrélation."""
    sigma_ln = np.sqrt(np.log(1 + (sigma_rf / mu_rf)**2))
    mu_ln    = np.log(mu_rf) - 0.5 * sigma_ln**2
    return np.exp(mu_ln + sigma_ln * field_gauss)


def apply_crf_to_mesh(nodes, tetras, nx, ny, nz, lambda_rf, sigma_rf, mu_rf, domain_mm):
    """
    Génère un CRF sur la grille régulière et interpole aux centres
    des éléments par recherche du voisin le plus proche.

    Retour
    ------
    crf_elem : array (n_elem,) — facteur CRF pour chaque élément
    """
    print("[CRF] Génération du champ aléatoire corrélé...")
    np.random.seed(42)
    field_g  = generate_gaussian_crf(nx, ny, nz, lambda_rf, domain_mm)
    field_ln = gaussian_to_lognormal(field_g, mu_rf, sigma_rf)

    # Grille régulière
    xs = np.linspace(0, domain_mm, nx)
    ys = np.linspace(0, domain_mm, ny)
    zs = np.linspace(0, domain_mm, nz)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    grid_val = field_ln.ravel()

    # Centres des éléments
    elem_centers = nodes[tetras].mean(axis=1)   # (n_elem, 3)

    # KD-tree pour interpolation au plus proche voisin
    print("[CRF] Interpolation sur les éléments du maillage...")
    tree = cKDTree(grid_pts)
    _, idx = tree.query(elem_centers, workers=-1)
    crf_elem = grid_val[idx]

    print(f"  → CRF interpolé : min={crf_elem.min():.3f}, "
          f"max={crf_elem.max():.3f}, mean={crf_elem.mean():.3f}")
    return crf_elem, field_ln


# ─────────────────────────────────────────────────────────────
# 5. PROPRIÉTÉS EFFECTIVES PAR ÉLÉMENT
# ─────────────────────────────────────────────────────────────

def compute_effective_properties(elem_phase, crf_elem, phase_props):
    """
    Propriétés effectives = propriété_phase × facteur_CRF.

    Retour
    ------
    E_eff, UCS_eff : arrays (n_elem,)
    """
    E_base   = np.array([phase_props[p]["E"]   for p in elem_phase])
    UCS_base = np.array([phase_props[p]["UCS"] for p in elem_phase])

    E_eff   = E_base   * crf_elem
    UCS_eff = UCS_base * crf_elem

    print(f"[Props] E_eff   : {E_eff.mean():.1f} ± {E_eff.std():.1f} GPa")
    print(f"[Props] UCS_eff : {UCS_eff.mean()/1e6:.1f} ± {UCS_eff.std()/1e6:.1f} MPa")
    return E_eff, UCS_eff


# ─────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────

def visualize(nodes, tetras, elem_phase, crf_elem, E_eff, UCS_eff,
              field_ln, domain_mm, phase_props, outdir):
    """Génère 3 figures de synthèse."""

    phase_colors = {p: phase_props[p]["color"] for p in phase_props}
    elem_colors  = np.array([phase_colors[p] for p in elem_phase])

    # ── Figure 1 : Phases minérales (coupe z = milieu) ──
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    centers   = nodes[tetras].mean(axis=1)
    z_mid     = domain_mm / 2
    tol       = domain_mm / (NZ * 2)
    mask      = np.abs(centers[:, 2] - z_mid) < tol * 8

    for phase, props in phase_props.items():
        m = mask & (elem_phase == phase)
        if m.any():
            ax1.scatter(centers[m, 0], centers[m, 1],
                        c=props["color"], s=4, alpha=0.7, label=phase)

    ax1.set_aspect('equal')
    ax1.set_title("Phases minérales — coupe z = 0.5 mm")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax1.legend(markerscale=4, fontsize=10)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "granite_phases.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → granite_phases.png sauvegardée")

    # ── Figure 2 : Champ CRF (coupes orthogonales 2D) ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Champ CRF log-normal — coupes orthogonales", fontsize=13)

    kw = dict(cmap='RdYlBu_r', origin='lower',
              extent=[0, domain_mm, 0, domain_mm], aspect='equal')
    im0 = axes2[0].imshow(field_ln[:, :, NZ//2].T,   **kw)
    im1 = axes2[1].imshow(field_ln[:, NY//2, :].T,   **kw)
    im2 = axes2[2].imshow(field_ln[NX//2, :, :].T,   **kw)

    for ax, title in zip(axes2, ["Plan XY (z=0.5)", "Plan XZ (y=0.5)", "Plan YZ (x=0.5)"]):
        plt.colorbar(ax.images[0], ax=ax, shrink=0.85)
        ax.set_title(title); ax.set_xlabel("[mm]"); ax.set_ylabel("[mm]")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "granite_crf.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → granite_crf.png sauvegardée")

    # ── Figure 3 : Distributions des propriétés effectives ──
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle("Distributions des propriétés effectives (phase × CRF)", fontsize=13)

    phases = list(phase_props.keys())
    colors = [phase_props[p]["color"] for p in phases]

    for phase, color in zip(phases, colors):
        m = elem_phase == phase
        ax3a.hist(E_eff[m],          bins=50, alpha=0.6, color=color, label=phase, density=True)
        ax3b.hist(UCS_eff[m] / 1e6, bins=50, alpha=0.6, color=color, label=phase, density=True)

    ax3a.set_xlabel("E [GPa]");    ax3a.set_ylabel("Densité"); ax3a.set_title("Module de Young effectif")
    ax3b.set_xlabel("UCS [MPa]");  ax3b.set_ylabel("Densité"); ax3b.set_title("Résistance UCS effective")
    ax3a.legend(); ax3b.legend()
    ax3a.grid(True, alpha=0.3); ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "granite_properties.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → granite_properties.png sauvegardée")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    outdir = WORKDIR
    os.makedirs(outdir, exist_ok=True)

    # 1. Neper
    tess_file, mesh_file = run_neper(
        WORKDIR, BASENAME, N_GRAINS, MESH_CL, DOMAIN_MM
    )

    # 2. Chargement maillage
    nodes, tetras, grain_ids = load_mesh(mesh_file)

    # 3. Attribution des phases
    print("\n[Phases] Attribution des phases minérales...")
    elem_phase, grain_phase = assign_phases(grain_ids, PHASE_FRACTIONS)

    # 4. Champ CRF
    print()
    crf_elem, field_ln = apply_crf_to_mesh(
        nodes, tetras, NX, NY, NZ, LAMBDA_RF, SIGMA_RF, MU_RF, DOMAIN_MM
    )

    # 5. Propriétés effectives
    print()
    E_eff, UCS_eff = compute_effective_properties(elem_phase, crf_elem, PHASE_PROPS)

    # 6. Visualisation
    print("\n[Viz] Génération des figures...")
    visualize(nodes, tetras, elem_phase, crf_elem, E_eff, UCS_eff,
              field_ln, DOMAIN_MM, PHASE_PROPS, outdir)

    print(f"\n✓ Pipeline terminé. Résultats dans : {outdir}")


if __name__ == "__main__":
    main()

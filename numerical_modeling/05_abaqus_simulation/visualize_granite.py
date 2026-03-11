"""
Visualisation 3D interactive de la microstructure granitique avec PyVista
=========================================================================
Prérequis :
    pip install pyvista meshio
"""

import os
import numpy as np
import pyvista as pv

MESH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "granite_sim", "granite_mesh.msh"
)

# ─────────────────────────────────────────────────────────────
# Chargement du maillage
# ─────────────────────────────────────────────────────────────

print(f"Chargement : {MESH_FILE}")
mesh = pv.read(MESH_FILE)
print(mesh)
print(f"  Tableaux disponibles : {mesh.array_names}")

# ─────────────────────────────────────────────────────────────
# Sélection du scalaire à afficher (tag de grain)
# ─────────────────────────────────────────────────────────────

# Cherche automatiquement un tableau de type "grain id"
CANDIDATES = ["gmsh:physical", "medit:ref", "cell_tags", "ElementBlockIds"]
scalar = next((c for c in CANDIDATES if c in mesh.array_names), None)

if scalar is None and mesh.array_names:
    scalar = mesh.array_names[0]
    print(f"  Aucun tag connu trouvé, utilisation de : '{scalar}'")
elif scalar:
    print(f"  Tag de grain utilisé : '{scalar}'")
else:
    print("  Aucun scalaire trouvé, affichage sans couleur.")

# ─────────────────────────────────────────────────────────────
# Vue 1 : Volume complet avec coupe
# ─────────────────────────────────────────────────────────────

plotter = pv.Plotter(shape=(1, 2), title="Microstructure granitique")

# --- Panneau gauche : vue externe ---
plotter.subplot(0, 0)
plotter.add_text("Vue externe — grains", font_size=10)
if scalar:
    plotter.add_mesh(mesh, scalars=scalar, cmap="tab20",
                     show_edges=False, opacity=1.0,
                     scalar_bar_args={"title": "ID grain"})
else:
    plotter.add_mesh(mesh, color="lightgray", show_edges=True)
plotter.add_axes()
plotter.view_isometric()

# --- Panneau droit : coupe au milieu ---
plotter.subplot(0, 1)
plotter.add_text("Coupe centrale (plan XY)", font_size=10)

# Coupe au milieu selon Z
bounds = mesh.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
z_mid  = (bounds[4] + bounds[5]) / 2
clip   = mesh.clip(normal="z", origin=(0, 0, z_mid))

if scalar:
    plotter.add_mesh(clip, scalars=scalar, cmap="tab20",
                     show_edges=True, edge_color="k", line_width=0.3,
                     scalar_bar_args={"title": "ID grain"})
else:
    plotter.add_mesh(clip, color="lightgray", show_edges=True)

plotter.add_axes()
plotter.view_xy()

print("\nFenêtre interactive — rotation : clic gauche | zoom : molette | fermer pour continuer")
plotter.show()

# ─────────────────────────────────────────────────────────────
# Vue 2 : Grains individuels (exploded view)
# ─────────────────────────────────────────────────────────────

if scalar:
    print("\nGénération de la vue éclatée des grains...")

    grain_ids = mesh.cell_data[scalar]
    unique_ids = np.unique(grain_ids)

    # Limite à 50 grains pour la performance
    display_ids = unique_ids[:min(50, len(unique_ids))]
    cmap = pv.plotting.colors.get_cmap_safe("tab20")

    plotter2 = pv.Plotter(title="Grains individuels (50 premiers)")
    plotter2.add_text("Grains individuels — couleur par ID", font_size=10)

    center = np.array(mesh.center)

    for i, gid in enumerate(display_ids):
        grain_mesh = mesh.extract_cells(grain_ids == gid)
        if grain_mesh.n_cells == 0:
            continue
        # Légère translation vers l'extérieur (vue éclatée)
        gc = np.array(grain_mesh.center)
        direction = gc - center
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            offset = direction / norm * 0.05
        else:
            offset = np.zeros(3)
        grain_mesh.translate(offset, inplace=True)
        color = cmap(i % 20)[:3]
        plotter2.add_mesh(grain_mesh, color=color, show_edges=True,
                          edge_color="k", line_width=0.2, opacity=0.9)

    plotter2.add_axes()
    plotter2.view_isometric()
    plotter2.show()

print("\n✓ Visualisation terminée.")

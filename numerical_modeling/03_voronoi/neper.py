import os
import sys
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def run(cmd, cwd=None):
    """Run a command, print it, raise if it fails."""
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_neper():
    if shutil.which("neper") is None:
        raise RuntimeError(
            "Neper n'est pas trouvé dans le PATH.\n"
            "Teste dans un terminal: neper -V\n"
            "Si ça ne marche pas, ajoute Neper au PATH ou utilise le chemin complet vers l'exécutable."
        )
    # petite vérif
    run(["neper", "-V"])


def main():
    # -----------------------------
    # Paramètres
    # -----------------------------
    # Domaine : rectangle long (mm ou unités arbitraires)
    Lx, Ly, Lz = 40.0, 5.0, 5.0
    N_GRAINS = 200

    # dossier de sortie
    outdir = Path("neper_demo_voronoi_3d").resolve()
    outdir.mkdir(exist_ok=True)

    # noms de fichiers
    base = "rect_voro"
    tess_file = outdir / f"{base}.tess"
    img_base = outdir / f"{base}_view"  # Neper ajoutera .png

    ensure_neper()

    # -----------------------------
    # 1) Génération tessellation Voronoï 3D
    # -----------------------------
    # Exemple doc Neper: neper -T -n 200 -morpho gg ... :contentReference[oaicite:1]{index=1}
    # Ici on impose un domaine box(Lx,Ly,Lz) (rectangle 3D)
    run([
        "neper",
        "-T",
        "-n", str(N_GRAINS),
        "-domain", f"box({Lx},{Ly},{Lz})",
        "-o", str(outdir / base)
    ])

    if not tess_file.exists():
        raise RuntimeError(f"Fichier tessellation introuvable: {tess_file}")

    # -----------------------------
    # 2) Rendu PNG (visualisation)
    # -----------------------------
    # Doc: neper -V n200.tess -print img1 -> img1.png :contentReference[oaicite:2]{index=2}
    # On peut aussi régler la transparence avec -datacelltrs (0 opaque, 1 transparent) :contentReference[oaicite:3]{index=3}
    run([
        "neper",
        "-V", str(tess_file),
        "-datacelltrs", "0.0",       # 0 = opaque
        "-showedge", "domtype==1",   # montre juste les arêtes du domaine (optionnel)
        "-print", str(img_base)
    ])

    img_file = Path(str(img_base) + ".png")
    if not img_file.exists():
        raise RuntimeError(f"Image PNG introuvable: {img_file}")

    # -----------------------------
    # 3) Affichage matplotlib
    # -----------------------------
    img = plt.imread(str(img_file))
    plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Neper Voronoï 3D — box({Lx},{Ly},{Lz}), n={N_GRAINS}")
    plt.tight_layout()
    plt.show()

    print("\nOK ✅")
    print("Tessellation :", tess_file)
    print("Image        :", img_file)
    print("Dossier      :", outdir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\nERREUR: Neper a échoué.")
        sys.exit(e.returncode)
    except Exception as e:
        print("\nERREUR:", e)
        sys.exit(1)
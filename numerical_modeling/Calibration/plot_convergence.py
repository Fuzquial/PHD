# -*- coding: utf-8 -*-
# Script de post-traitement - Etude de convergence de maillage
# Compare les courbes (-U3 , -RF3) pour chaque job
#
# Utilisation :
#   - Depuis Abaqus CAE : File > Run Script > plot_convergence.py
#   - Depuis le terminal : abaqus python plot_convergence.py
#
# Les fichiers .odb doivent etre dans le meme dossier que ce script
# (ou adapter WORK_DIR ci-dessous).

import os
import sys

# ============================================================
# Configuration
# ============================================================

# Dossier contenant les fichiers .odb
WORK_DIR = r'C:\Users\fuzquianoalricabi\Documents\python'

# (job_name, legende) dans l'ordre voulu sur le graphe
JOBS = [
    ('mesh005_5MPa_v2', 'maille 0.5 mm'),
    ('mesh003_5MPa',    'maille 0.3 mm'),
    ('mesh002_5MPa',    'maille 0.2 mm'),
    ('mesh_005',        'maille 0.1 mm'),
]

# Steps a concatener (dans l'ordre chronologique)
# Step-1 = confinement, Step-2 = compression axiale
STEPS = ['Step-1', 'Step-2']

OUTPUT_PNG = os.path.join(WORK_DIR, 'convergence_maillage.png')
OUTPUT_CSV = os.path.join(WORK_DIR, 'convergence_maillage.csv')

# ============================================================
# Extraction des donnees ODB
# ============================================================
try:
    from odbAccess import openOdb
except ImportError:
    print("ERREUR : odbAccess non disponible.")
    print("Lancer ce script via : abaqus python plot_convergence.py")
    sys.exit(1)


def extract_u3_rf3(odb_path):
    """
    Ouvre un ODB et extrait les series temporelles U3 et RF3
    depuis la premiere region d'historique qui les contient.
    Concatene tous les steps de STEPS.
    Retourne (list_u3, list_rf3) tries par temps croissant.
    """
    odb = openOdb(odb_path, readOnly=True)

    all_time = []
    all_u3   = []
    all_rf3  = []

    for step_name in STEPS:
        if step_name not in odb.steps.keys():
            continue
        step = odb.steps[step_name]

        # Chercher la region contenant a la fois U3 et RF3
        target_region = None
        for rkey in step.historyRegions.keys():
            hr = step.historyRegions[rkey]
            if 'U3' in hr.historyOutputs.keys() and 'RF3' in hr.historyOutputs.keys():
                target_region = hr
                break

        if target_region is None:
            print("  ATTENTION : RF3/U3 introuvables dans le step '%s'" % step_name)
            continue

        u3_data  = target_region.historyOutputs['U3'].data   # list of (time, value)
        rf3_data = target_region.historyOutputs['RF3'].data

        # Construire un dict temps -> valeur pour les deux
        rf3_dict = {}
        for (t, v) in rf3_data:
            rf3_dict[t] = v

        for (t, u3_val) in u3_data:
            if t in rf3_dict:
                all_time.append(t + step.totalTime)
                all_u3.append(u3_val)
                all_rf3.append(rf3_dict[t])

    odb.close()

    # Trier par temps croissant
    combined = sorted(zip(all_time, all_u3, all_rf3), key=lambda x: x[0])
    u3_out  = [-row[1] for row in combined]   # signe convention compression positive
    rf3_out = [-row[2] for row in combined]

    return u3_out, rf3_out


# Collecte des donnees pour tous les jobs
datasets = []

for (job_name, label) in JOBS:
    odb_path = os.path.join(WORK_DIR, job_name + '.odb')
    if not os.path.isfile(odb_path):
        print("ODB non trouve, ignore : %s" % odb_path)
        continue
    print("Lecture de %s ..." % job_name)
    u3_list, rf3_list = extract_u3_rf3(odb_path)
    datasets.append((label, u3_list, rf3_list))
    print("  -> %d points extraits" % len(u3_list))

if not datasets:
    print("Aucune donnee disponible. Verifier WORK_DIR et les noms de jobs.")
    sys.exit(1)

# ============================================================
# Export CSV
# ============================================================
with open(OUTPUT_CSV, 'w') as f:
    # En-tete
    header_parts = []
    for (label, u3, rf3) in datasets:
        safe = label.replace(' ', '_').replace('.', 'p')
        header_parts.append('-U3_%s' % safe)
        header_parts.append('-RF3_%s' % safe)
    f.write(','.join(header_parts) + '\n')

    # Nombre de lignes = max des longueurs
    max_len = max(len(u3) for (_, u3, _) in datasets)
    for i in range(max_len):
        row = []
        for (_, u3, rf3) in datasets:
            if i < len(u3):
                row.append('%.6e' % u3[i])
                row.append('%.6e' % rf3[i])
            else:
                row.append('')
                row.append('')
        f.write(','.join(row) + '\n')

print("CSV sauvegarde : %s" % OUTPUT_CSV)

# ============================================================
# Trace matplotlib
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')   # backend sans fenetre (compatible Abaqus)
    import matplotlib.pyplot as plt

    COLORS     = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
    LINESTYLES = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (label, u3, rf3) in enumerate(datasets):
        c  = COLORS[idx % len(COLORS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(u3, rf3, color=c, linestyle=ls, linewidth=1.8, label=label)

    ax.set_xlabel('-U3  [mm]  (deplacement axial)', fontsize=12)
    ax.set_ylabel('-RF3  [N]  (force axiale)', fontsize=12)
    ax.set_title('Convergence de maillage - Confinement 5 MPa', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print("Graphe sauvegarde : %s" % OUTPUT_PNG)

except ImportError:
    print("matplotlib non disponible dans cet environnement Abaqus.")
    print("Les donnees ont quand meme ete exportees dans le CSV.")
    print("Ouvrir le CSV dans Excel ou Python externe pour tracer.")

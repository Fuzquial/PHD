# -*- coding: utf-8 -*-
# Post-traitement essai triaxial - campagne 5/10/30/50 MPa
# A lancer dans Abaqus/CAE apres la fin des jobs (File -> Run Script)
# OU en ligne de commande : abaqus cae noGUI=postprocess_triaxial.py

from odbAccess import *
import os

# ============================================================
# Parametres
# ============================================================
PRESSURE_LIST = [5, 10, 30, 50]   # MPa
CYL_R = 2.5    # mm
CYL_H = 10.0   # mm
AREA  = 3.14159265358979 * CYL_R**2   # mm^2

# ============================================================
# Extraction des donnees pour un ODB
# ============================================================
def extract_rf3_u3(odb_path):
    """Retourne (times, rf3, u3) depuis l'historique RP_TOP."""
    odb = openOdb(path=odb_path, readOnly=True)
    rf3_data = {}
    u3_data  = {}

    for stepName in odb.steps.keys():
        step = odb.steps[stepName]
        for hRegionKey in step.historyRegions.keys():
            hRegion = step.historyRegions[hRegionKey]
            if 'RF3' in hRegion.historyOutputs:
                for (t, v) in hRegion.historyOutputs['RF3'].data:
                    rf3_data[t] = v
            if 'U3' in hRegion.historyOutputs:
                for (t, v) in hRegion.historyOutputs['U3'].data:
                    u3_data[t] = v

    odb.close()

    times = sorted(set(rf3_data.keys()) & set(u3_data.keys()))
    if not times:
        return [], [], []

    rf3_list = [rf3_data[t] for t in times]
    u3_list  = [u3_data[t]  for t in times]
    return times, rf3_list, u3_list

# ============================================================
# Boucle sur les pressions
# ============================================================
all_results = {}   # {pressure: {'eps': [...], 'sigma': [...], 'u3': [...], 'rf3': [...]}}

for p in PRESSURE_LIST:
    job_name = '%dMPa_hetero' % p
    odb_path = job_name + '.odb'
    csv_path = job_name + '_resultats.csv'

    if not os.path.exists(odb_path):
        print("ATTENTION: ODB introuvable -> %s (job non termine ?)" % odb_path)
        continue

    print("Traitement: %s ..." % odb_path)
    times, rf3_list, u3_list = extract_rf3_u3(odb_path)

    if not times:
        print("  ERREUR: aucune donnee RF3/U3 trouvee dans %s" % odb_path)
        continue

    # Calcul contrainte et deformation axiales
    sigma_list = [-rf3 / AREA   for rf3 in rf3_list]   # MPa, compression positive
    eps_list   = [-u3  / CYL_H  for u3  in u3_list]    # sans unite

    all_results[p] = {'times': times, 'rf3': rf3_list, 'u3': u3_list,
                      'sigma': sigma_list, 'eps': eps_list}

    # Export CSV individuel
    with open(csv_path, 'w') as f:
        f.write("Temps_s,RF3_N,U3_mm,Sigma_axiale_MPa,Epsilon_axiale\n")
        for i in range(len(times)):
            f.write("%.6g,%.6g,%.6g,%.6g,%.6g\n" % (
                times[i], rf3_list[i], u3_list[i], sigma_list[i], eps_list[i]))
    print("  CSV: %s  |  Sigma_max = %.1f MPa  |  U3_final = %.3f mm" % (
        csv_path, max(sigma_list), u3_list[-1]))

# ============================================================
# Graphiques avec matplotlib
# ============================================================
if not all_results:
    print("Aucun resultat disponible — verifiez les ODB.")
else:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        COLORS = {5: 'blue', 10: 'green', 30: 'orange', 50: 'red'}
        MARKERS = {5: 'o', 10: 's', 30: '^', 50: 'D'}

        # --- Figure 1 : Contrainte-Deformation (toutes pressions) ---
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        for p in sorted(all_results.keys()):
            r = all_results[p]
            ax1.plot(r['eps'], r['sigma'],
                     color=COLORS.get(p, 'black'),
                     linewidth=1.8,
                     label='Confinement %d MPa' % p)
        ax1.set_xlabel('Deformation axiale (-)')
        ax1.set_ylabel('Contrainte axiale (MPa)')
        ax1.set_title('Courbes contrainte-deformation — Essai triaxial')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        fig1.tight_layout()
        fig1.savefig('triaxial_sigma_eps.png', dpi=150)
        plt.close(fig1)
        print("Graphique sauvegarde: triaxial_sigma_eps.png")

        # --- Figure 2 : Force-Deplacement (toutes pressions) ---
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        for p in sorted(all_results.keys()):
            r = all_results[p]
            ax2.plot(r['u3'], r['rf3'],
                     color=COLORS.get(p, 'black'),
                     linewidth=1.8,
                     label='Confinement %d MPa' % p)
        ax2.set_xlabel('Deplacement U3 (mm)')
        ax2.set_ylabel('Force de reaction RF3 (N)')
        ax2.set_title('Courbes force-deplacement — Essai triaxial')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        fig2.tight_layout()
        fig2.savefig('triaxial_force_depl.png', dpi=150)
        plt.close(fig2)
        print("Graphique sauvegarde: triaxial_force_depl.png")

        # --- Figure 3 : Resistance pic vs pression de confinement ---
        pressions_done = sorted(all_results.keys())
        sigma_pic = [max(all_results[p]['sigma']) for p in pressions_done]

        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.plot(pressions_done, sigma_pic, 'ko-', linewidth=2, markersize=8)
        for p, s in zip(pressions_done, sigma_pic):
            ax3.annotate('%.1f MPa' % s, xy=(p, s),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Pression de confinement (MPa)')
        ax3.set_ylabel('Resistance en compression (MPa)')
        ax3.set_title('Resistance pic vs confinement')
        ax3.grid(True, linestyle='--', alpha=0.5)
        fig3.tight_layout()
        fig3.savefig('triaxial_resistance_pic.png', dpi=150)
        plt.close(fig3)
        print("Graphique sauvegarde: triaxial_resistance_pic.png")

    except ImportError:
        print("matplotlib non disponible — graphiques non generes.")
        print("Installer avec : pip install matplotlib")

print("\nPost-traitement termine.")
print("Fichiers produits :")
for p in sorted(all_results.keys()):
    print("  - %dMPa_hetero_resultats.csv" % p)
if all_results:
    print("  - triaxial_sigma_eps.png")
    print("  - triaxial_force_depl.png")
    print("  - triaxial_resistance_pic.png")

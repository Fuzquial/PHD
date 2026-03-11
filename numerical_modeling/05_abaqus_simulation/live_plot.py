# -*- coding: utf-8 -*-
# Live plot - courbe contrainte-deformation en temps reel
# Lancer depuis Abaqus VIEWER : File -> Run Script -> live_plot.py
# Le script relit l'ODB toutes les REFRESH_INTERVAL secondes

from visualization import *
import time

# ============================================================
# Parametres
# ============================================================
JOB_NAME         = '5MPa_hetero_test'   # nom du job sans .odb
ODB_PATH         = JOB_NAME + '.odb'
REFRESH_INTERVAL = 10     # secondes entre chaque mise a jour
MAX_ITERATIONS   = 720    # 720 x 10s = 2h max avant arret auto

CYL_R = 2.5
CYL_H = 10.0
AREA  = 3.14159265358979 * CYL_R**2

# ============================================================
# Ouverture ODB + creation du viewport XY
# ============================================================
print("Ouverture ODB: %s" % ODB_PATH)
if ODB_PATH in session.odbs.keys():
    session.odbs[ODB_PATH].close()
odb = session.openOdb(name=ODB_PATH)

# Creer le XY Plot dans le viewport
xyp       = session.XYPlot(name='LivePlot')
chartName = xyp.charts.keys()[0]
chart     = xyp.charts[chartName]

vp = session.viewports['Viewport: 1']
vp.setValues(displayedObject=xyp)

print("Viewport XY cree. Debut du suivi...")

# ============================================================
# Boucle de mise a jour
# ============================================================
for iteration in range(MAX_ITERATIONS):

    try:
        # Fermer et rouvrir l'ODB pour lire les nouvelles donnees
        if ODB_PATH in session.odbs.keys():
            session.odbs[ODB_PATH].close()
        odb = session.openOdb(name=ODB_PATH)

        # Extraction RF3 et U3
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

        times = sorted(set(rf3_data.keys()) & set(u3_data.keys()))

        if not times:
            print("  [%d] Pas encore de donnees..." % iteration)
        else:
            # Calcul contrainte et deformation
            xy_data = tuple(
                (-u3_data[t] / CYL_H, -rf3_data[t] / AREA)
                for t in times
            )

            # Mise a jour du XY data dans la session
            xy_name = 'sigma_eps_live'
            if xy_name in session.xyDataObjects.keys():
                del session.xyDataObjects[xy_name]
            session.XYData(name=xy_name, data=xy_data)

            # Mise a jour de la courbe dans le chart
            curve = session.Curve(xyData=session.xyDataObjects[xy_name])
            curve.lineStyle.setValues(color='#0000FF', thickness=VERY_THIN)
            chart.setValues(curvesToPlot=(curve,))

            # Mise en forme axes (apres premier tracé)
            try:
                chart.axes1[0].axisData.setValues(title='Deformation axiale (-)')
                chart.axes2[0].axisData.setValues(title='Contrainte axiale (MPa)')
                xyp.title.setValues(text='Contrainte-Deformation — %s (live)' % JOB_NAME)
            except Exception:
                pass

            # Rafraichir le viewport
            vp.setValues(displayedObject=xyp)

            sigma_max = max(v[1] for v in xy_data)
            eps_last  = xy_data[-1][0]
            print("  [%ds] Points: %d | Sigma_max: %.1f MPa | Eps: %.4f" % (
                iteration * REFRESH_INTERVAL, len(times), sigma_max, eps_last))

    except Exception as e:
        print("  [%d] Erreur: %s" % (iteration, str(e)))

    # Attendre avant la prochaine lecture
    time.sleep(REFRESH_INTERVAL)

print("Live plot termine apres %d iterations." % MAX_ITERATIONS)

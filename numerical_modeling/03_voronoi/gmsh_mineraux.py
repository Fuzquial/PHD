import gmsh
import sys
import os

def mesh_stl_to_volume(input_stl, output_msh):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("Mineral_Mesh")

    # 1. Importer le fichier STL
    try:
        gmsh.merge(input_stl)
    except:
        print(f"Erreur : Impossible de lire le fichier {input_stl}")
        return

    # 2. Créer une surface géométrique à partir des triangles du STL
    # Gmsh va classifier les triangles pour créer des surfaces discrètes
    angle = 40  # Seuil pour détecter les arêtes vives
    forceParametrizablePatches = False
    includeBoundary = True
    curveAngle = 180

    gmsh.model.mesh.classifySurfaces(angle * 3.14159 / 180., includeBoundary, 
                                      forceParametrizablePatches, curveAngle * 3.14159 / 180.)
    
    # Créer une géométrie à partir des surfaces classifiées
    gmsh.model.mesh.createGeometry()

    # 3. Créer un volume
    # Récupérer toutes les surfaces (dim=2) pour définir l'enveloppe du volume
    s = gmsh.model.getEntities(2)
    l = [it[1] for it in s]
    surface_loop = gmsh.model.geo.addSurfaceLoop(l)
    volume = gmsh.model.geo.addVolume([surface_loop])
    
    gmsh.model.geo.synchronize()

    # 4. Paramétrage du maillage
    # On définit une taille de maille cible (plus petit = plus précis)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)

    # 5. Générer le maillage (1D, 2D, puis 3D)
    print("Génération du maillage volumique...")
    gmsh.model.mesh.generate(3)

    # 6. Sauvegarder
    gmsh.write(output_msh)
    print(f"Maillage terminé : {output_msh}")

    # Lancer l'interface graphique pour vérifier le résultat
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    # Remplacez par le nom de votre fichier exporté de MATLAB
    input_file = "mine/mineral_1.stl" 
    if os.path.exists(input_file):
        mesh_stl_to_volume(input_file, "mineral_1.msh")
    else:
        print(f"Fichier {input_file} introuvable. Exportez-le d'abord depuis MATLAB.")
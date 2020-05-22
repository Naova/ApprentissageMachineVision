import renommer_fichiers
import brut_to_png
import Etiquetage.main

print("Renommer les fichiers")
renommer_fichiers.main()
print("Generation des fichiers PNG")
brut_to_png.main()
print("Lancement du programme d'etiquetage...")
Etiquetage.main.main()
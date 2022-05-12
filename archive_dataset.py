from tqdm import tqdm
import zipfile
import pathlib

dataset_subfolders = ["Genere", "Robot", "Simulation", "HardNegative"]
cameras = ["lower", "upper"]
things_to_zip = ["labels.json", "Brut"]

with zipfile.ZipFile('Dataset.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for subfolder in dataset_subfolders:
        for camera in cameras:
            for thing_to_zip in things_to_zip:
                s = f'Dataset/{subfolder}/{camera}/{thing_to_zip}'
                print(s)
                if thing_to_zip == "Brut":
                    files = pathlib.Path(f'Dataset/{subfolder}/{camera}/Brut/').glob('**/*')
                    for file in tqdm(files):
                        archive.write(str(file), arcname=str(file))
                else:
                    archive.write(s)

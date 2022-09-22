from tqdm import tqdm
import zipfile
import pathlib
import os

dataset_subfolders = ["Genere", "Robot", "Simulation", "HardNegative", "NewNegative", "TestRobot"]
cameras = ["lower", "upper"]
things_to_zip = ["labels.json", "YCbCr"]

with zipfile.ZipFile('Dataset.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for subfolder in dataset_subfolders:
        for camera in cameras:
            for thing_to_zip in things_to_zip:
                s = f'Dataset/{subfolder}/{camera}/{thing_to_zip}'
                print(s)
                if thing_to_zip == "YCbCr":
                    files = pathlib.Path(f'Dataset/{subfolder}/{camera}/YCbCr/').glob('**/*')
                    files = [f for f in files]
                    for file in tqdm(files):
                        archive.write(str(file), arcname=str(file))
                elif os.path.exists(s):
                    archive.write(s)

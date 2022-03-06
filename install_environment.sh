#/bin/sh

sudo apt update
sudo apt upgrade

#dependances de Anaconda
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

#Telecharge l'installateur d'Anaconda et le lance
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output conda_installer.sh
chmod +x conda_installer.sh
./conda_installer.sh

#Initialise Anaconda (est normalement fait automatiquement dans le .bashrc, mais on evite ici de devoir relancer une session shell)
. ~/anaconda3/etc/profile.d/conda.sh

#cree un environnement conda
conda create --name=vision python=3.7 -y
conda activate vision

#clone de depot
git clone https://github.com/Naova/ApprentissageMachineVision.git
cd ApprentissageMachineVision

#installe les dependances, y compris NNCG
pip install -r requirements.txt

#telecharge le dataset a partir de Google drive, puis le dezip (peut aussi etre fait manuellement)
pip install gdown
gdown 1lySyNXhlMx1p8s47hX0RO_AQCA9fAopt

#dezip le dataset
sudo apt install unzip
unzip Dataset.zip

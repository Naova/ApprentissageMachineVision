import argparse
import os

from pathlib import Path
import paramiko
from paramiko import SSHClient
from scp import SCPClient

from yolo.training.configuration_provider import ConfigurationProvider as cfg_prov

def download_from_robot():
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    nao_ip = input("Nao IP : ")
    ssh.connect(hostname=f'{nao_ip}', username='nao', password='nao')

    scp = SCPClient(ssh.get_transport())
    
    if not os.path.exists('Dataset/Temp/'):
        os.mkdir('Dataset/Temp')
    scp.get('/var/volatile/Dataset/lower', recursive=True, local_path='Dataset/Robot/')
    scp.get('/var/volatile/Dataset/upper', recursive=True, local_path='Dataset/Robot/')
    
    for camera in ['lower', 'upper']:
        for batch in Path(f'Dataset/Robot/{camera}/YCbCr/').glob('batch_*'):
            for file in Path(batch).glob('*_label'):
                os.remove(file)
    
def copy_from_simulation():
    pass


def main():
    cfg_prov.set_config('balles')
    cfg_prov.get_config().camera = "upper"
    
    parser = argparse.ArgumentParser(description='Add images to the dataset')

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('-s', '--simulation', action='store_true',
                        help='Utiliser l\'environnement de la simulation.')
    action.add_argument('-r', '--robot', action='store_true',
                        help='Utiliser l\'environnement des robots.')
    args = parser.parse_args()
    
    if args.simulation:
        copy_from_simulation()
    else:
        download_from_robot()


if __name__ == '__main__':
    main()

# Script to download the Titanic dataset into datos/datos_ini
import os
import requests

def download_titanic(dest_dir='datos/datos_ini', url=None):
    os.makedirs(dest_dir, exist_ok=True)
    if url is None:
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    dest_path = os.path.join(dest_dir, 'titanic.csv')
    r = requests.get(url)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(r.content)
    print(f'Downloaded titanic dataset to {dest_path}')

if __name__ == '__main__':
    download_titanic()

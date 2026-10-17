#!/usr/bin/env python3
"""
Descarga el dataset público del Titanic y guarda en datos/datos_ini/titanic.csv
Genera además un archivo de versionado SHA256 datos/datos_ini/version.txt
"""
import hashlib
import os
from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
ROOT = Path(__file__).resolve().parents[1]
DATOS_INI = ROOT / "datos" / "datos_ini"
DATOS_INI.mkdir(parents=True, exist_ok=True)
TARGET = DATOS_INI / "titanic.csv"


def download():
    print(f"Descargando {URL} -> {TARGET}")
    r = requests.get(URL)
    r.raise_for_status()
    TARGET.write_bytes(r.content)
    print("Descarga completa.")
    # Hash
    h = hashlib.sha256(r.content).hexdigest()
    (DATOS_INI / "version.txt").write_text(h)
    print("Version (SHA256):", h)


if __name__ == "__main__":
    download()

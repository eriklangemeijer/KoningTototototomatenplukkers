# https://api.nevobo.nl/export/poule/{regio}/{poule}/programma.{type}
# https://api.nevobo.nl/export/poule/{regio}/{poule}/resultaten.{type}
# https://api.nevobo.nl/export/poule/{regio}/{poule}/stand.{type}
# https://api.nevobo.nl/export/vereniging/{verenigingscode}/programma.{type}
# https://api.nevobo.nl/export/vereniging/{verenigingscode}/resultaten.{type}
# https://api.nevobo.nl/export/team/{verenigingscode}/{teamtype}/{volgnummer}/programma.{type}
# https://api.nevobo.nl/export/team/{verenigingscode}/{teamtype}/{volgnummer}/resultaten.{type}
# https://api.nevobo.nl/export/sporthal/{sporthal}/programma.{type}
# https://api.nevobo.nl/export/sporthal/{sporthal}/resultaten.{type}
# https://api.nevobo.nl/export/nieuws.{type}
# https://api.nevobo.nl/export/activiteiten.{type}
# https://api.nevobo.nl/export/toernooien.{type}

# {type} = ‘rss’, ‘xlsx’, ‘ics’ (niet alle types zijn altijd beschikbaar)
# {regio} = ‘regio-noord’, ‘regio-oost’, ‘regio-zuid’, ‘regio-west’, ‘nationale-competitie’, ‘kampioenschappen’
# {poule} = pouleafkorting
# {teamtype} = ‘dames-senioren’, ‘jongens-a’, etc.
# {sporthal} = sporthalcode (vijf letters)

# https://api.nevobo.nl/export/poule/regio-zuid/H1E/resultaten.xlsx
# https://api.nevobo.nl/export/vereniging/ckm0d24/programma.xlsx

import requests
import pandas as pd
import os.path
from pathlib import Path

CLUB_CODE = 'ckm0d24'
FORCE_DOWNLOAD = True

def get_club_program(club_code, filename):
    parent_path = Path(filename).parent.absolute()
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    url = f"https://api.nevobo.nl/export/vereniging/{club_code}/programma.xlsx"
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def get_poule_results(poule_code, filename, region='regio-zuid'):
    #https://api.nevobo.nl/export/poule/regio-zuid/H1E/resultaten.xlsx
    url = f"https://api.nevobo.nl/export/poule/{region}/{poule_code}/resultaten.xlsx"
    r = requests.get(url, allow_redirects=True)

    open(filename, 'wb').write(r.content)

def download_poule_results_for_club(club_code):
    club_program_path = f"output_files/club_program.xlsx"
    if(FORCE_DOWNLOAD or not os.path.isfile(club_program_path)):
        get_club_program(club_code, club_program_path)
 
    # read by default 1st sheet of an excel file
    dataframe = pd.read_excel(club_program_path)
    poules = set(dataframe.Poule)
    for poule in poules:
        poule_result_path = f"output_files/result_poule_{poule}.xlsx"
        if(FORCE_DOWNLOAD or not os.path.isfile(poule_result_path)):
            get_poule_results(poule, poule_result_path)
    

if __name__ == "__main__":
    download_poule_results_for_club(CLUB_CODE)

import pandas as pd
import os.path
import logging

from datetime import datetime

class MatchResult:
    def __init__(self, row):
        self.match_state = row['Wedstrijd status']
        if(self.match_state == 'gespeeld'):                
            self.date = row['Datum']
            self.time = row['Tijd']
            self.team_home = row['Team thuis']
            self.team_away = row['Team uit']
            self.outcome = row['Uitslag']
            self.set_outcomes = row['Setstanden']
            self.region = row['Regio']
            self.poule_code = row['Poule']
            self.match_code = row['Code']
            self.hall_code = row['Zaalcode']
            self.hall_name = row['Zaal']
            self.town = row['Plaats']
        else:     
            return None
        

def get_current_program():
    results_dict = {}
    club_program_path = f"output_files/club_program.xlsx" 
    # read by default 1st sheet of an excel file
    df_program = pd.read_excel(club_program_path)
    poules = set(df_program.Poule)
    for poule in poules:
        results_dict[poule] = []

    today = datetime.now()
    for index, row in df_program.iterrows():
        if(row.Datum > today and (row.Datum - today).days < 7 ):
            results_dict[row.Poule].append(row)

    return results_dict
        


def get_historic_results():
    results_dict = {}
    results_array = []
    club_program_path = f"output_files/club_program.xlsx" 
    # read by default 1st sheet of an excel file
    df_program = pd.read_excel(club_program_path)
    poules = set(df_program.Poule)
    for poule in poules:
        results_dict[poule] = []
        poule_result_path = f"output_files/result_poule_{poule}.xlsx"
        try:
            df_results = pd.read_excel(poule_result_path)
        except ValueError:
            logging.warning(f"Cannot parse file for poule {poule}, continueing")
            continue


        for index, row in df_results.iterrows():
            
            result = MatchResult(row)
            if(result.match_state == 'gespeeld'):         
                results_dict[poule].append(result )
                results_array.append(result )
    return results_array


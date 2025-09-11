from dateutil import parser
import re
from datasets import load_dataset
import pandas as pd
import json
import unicodedata
import os


datasets_ = {dataset: load_dataset("sxiong/TGQA", dataset)['test']
            for dataset in ['TGQA_Story_TG_Trans', 'TimeQA_Story_TG_Trans', 'TempReason_Story_TG_Trans']
            }

# print(datasets_['TimeQA_Story_TG_Trans']['TG'])

def clean_term(i: str):
    if i[0].isdigit():
        i = '_' + i
    i = unicodedata.normalize("NFKD", i).encode("ascii", "ignore").decode("ascii")
    for tok in ['(', ')', ',', '.', '&', '-', '/']:
        i = i.replace(tok, '_')
    return i


def tg_to_asp(TG, TG_type: str) -> str:
    """
    transforms a temporal graph to a string which can be written as content into an ASP instance file
    
    Args:
        TG (list) a temporal graph object encoding some story
    
    Return:
        A string containing valid ASP code to be written into an instance file
    """
    if TG_type == 'TGQA':
        # explicitl write out all relations as indicated in the TGQA data
        relation_dict = {
                        'was born in': 'born_in',
                        'was birthed in': 'born_in',
                        'entered the world in': 'born_in',
                        'died in': 'die',
                        'passed away in': 'die',
                        'expired in': 'die',
                        'worked at': 'work_at',
                        'served at': 'work_at',
                        'employed by': 'work_at',
                        'played for': 'play_for',
                        'joined': 'play_for',
                        'won prize': 'win_prize',
                        'received award': 'win_prize',
                        'received prize': 'win_prize',
                        'was married to': 'married_to',
                        'tied the knot with': 'married_to',
                        'united in marriage with': 'married_to',
                        'owned': 'own',
                        'possessed': 'own',
                        'studied in': 'study',
                        'educated in': 'study',
                        'was affiliated to': 'affiliated_to',
                        'was a member of': 'affiliated_to',
                        'was associated with': 'affiliated_to',
                        'created': 'create',
                        'produced': 'create',
                        'crafted': 'create'
                        }
        
        # dictionary to store ongoing events (since start and end have different entries in the graph)
        ongoing_events = {}
        # facts to be written into an asp instance file
        facts = ''
        for temporal_event in TG:
            for relation_type in relation_dict:
                if relation_type in temporal_event:
                    # Split on the relation
                    parts = temporal_event.split(relation_type)
                    if len(parts) == 2:
                        # First part has subject (remove opening parenthesis)
                        subject = parts[0].strip().replace('(', '').strip().lower().replace(' ', '_')
                        
                        # Second part has object and start/end info
                        second_part = parts[1].strip()
                        
                        # Extract object (everything before ") starts/ends")
                        if ') starts' in second_part:
                            obj = second_part.split(') starts')[0].strip().lower().replace(' ', '_')
                            start_or_end = 'starts'
                            year = int(second_part.split('starts at ')[1])
                        elif ') ends' in second_part:
                            obj = second_part.split(') ends')[0].strip().lower().replace(' ', '_')
                            start_or_end = 'ends'
                            year = int(second_part.split('ends at ')[1])
                        else:
                            continue
                        
                        relation_token = relation_dict[relation_type]
                        event_key = (subject, relation_token, obj)
                        
                        if start_or_end == 'starts':
                            ongoing_events[event_key] = year
                        elif start_or_end == 'ends':
                            if event_key in ongoing_events:
                                start_year = ongoing_events[event_key]
                                facts += f'event({clean_term(subject)}, {clean_term(relation_token)}, {clean_term(obj)}, {start_year}, 1, {year}, 12).\n'
                                del ongoing_events[event_key]
                        
                        break  # Found matching relation

        # Handle events that only had starts (no explicit ends)
        for (subject, relation_token, obj), start_year in ongoing_events.items():
            facts += f'event({clean_term(subject)}, {clean_term(relation_token)}, {clean_term(obj)}, {start_year}, 1, {start_year}, 1).\n'

    elif TG_type == 'TimeQA':
        facts = ''
        for temporal_event in TG:
            # extract time interval and event
            interval, event = temporal_event.split(':')
            # split to get start str and end string separately
            if '-' in interval:
                start_date, end_date = interval.split('-')
            else:
                start_date, end_date = interval, interval
            # parse start and end strings into timestamp float
            start_date = parser.parse(start_date.strip())
            end_date = parser.parse(end_date.strip())
            start_year, start_moth = int(f"{start_date.year}"), int(f"{start_date.month:02d}")
            end_year, end_month = int(f"{end_date.year}"), int(f"{end_date.month:02d}")
            
            # extract entities (nodes) in temporal graph
            poss_match = re.match(r"(.+?)'s\s+(.*?)\s+", event.strip())
            
            if poss_match:
                out_node = poss_match.group(1).strip().lower().replace(' ', '_')
                relation = poss_match.group(2).strip().lower().replace(' ', '_')
                
                # Find all objects in parentheses
                in_nodes = re.findall(r'\(\s*([^)]+)\s*\)', event.strip())
                in_nodes = [node.strip().lower().replace(' ', '_') for node in in_nodes]
            
            else:
                # Assumes format like "Galatasaray S.K. (football) is ( Unknown )"
                direct_match = re.match(r"(.+?)\s+(\w+)\s+", event.strip())
                
                if not direct_match:
                    return None  # unrecognized format, skip or log it
                
                out_node = direct_match.group(1).strip().lower().replace(' ', '_')
                relation = direct_match.group(2).strip().lower().replace(' ', '_')
                
                # Find all objects in parentheses
                in_nodes = re.findall(r'\(\s*([^)]+)\s*\)', event.strip())
                in_nodes = [node.strip().lower().replace(' ', '_') for node in in_nodes]
            
            for in_node in in_nodes:
                facts += f'event({clean_term(out_node)}, {clean_term(relation)}, {clean_term(in_node)}, {start_year}, {start_moth}, {end_year}, {end_month}).\n'
    
    # my implementation does not support TempReason data.
    # The questions from the two other datasets alone are already way too many if we were to use all $30 for text complete
    elif TG_type == 'TempReason':
        pass

    else:
        raise ValueError(f'TG_type variable must be "TGQA", "TimeQA" or "TempReason", not {TG_type}.')
    
    return facts
    


def create_asp_instance_files(dataset, TG_type: str) -> None:
    """
    given a TGR dataset generate all samples corresponding instance files containing the corresp
    """


    # Create directory if it doesn't exist
    directory = f"materials/{TG_type}"
    os.makedirs(directory, exist_ok=True)

    for instance in dataset:
        # extract identifier of dataset example and construct corresponding asp instance file
        file_name = instance['id'].replace('/', '_') + '.lp'
        facts = tg_to_asp(instance['TG'], TG_type)
        # Write facts to file
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as f:
            f.write(facts)

def reason(instance_path, encoding_path) -> str:
    pass


if __name__ == '__main__':
    create_asp_instance_files(datasets_['TGQA_Story_TG_Trans'], 'TGQA')
    create_asp_instance_files(datasets_['TimeQA_Story_TG_Trans'], 'TimeQA')
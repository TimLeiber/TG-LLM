from dateutil import parser
import re
from datasets import load_dataset
import pandas as pd
import json


datasets_ = {dataset: load_dataset("sxiong/TGQA", dataset)['test']
            for dataset in ['TGQA_Story_TG_Trans', 'TimeQA_Story_TG_Trans', 'TempReason_Story_TG_Trans']
            }

# print(datasets_['TimeQA_Story_TG_Trans']['TG'])

def tg_to_asp(TG, TG_type: str) -> str:
    """
    transforms a temporal graph to a string which can be written as content into an ASP instance file
    
    Args:
        TG (list) a temporal graph object encoding some story
    
    Return:
        A string containing valid ASP code to be written into an instance file
    """
    if TG_type == 'TGQA':
        pass
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
            start_date = float(f"{start_date.year}.{start_date.month:02d}")
            end_date = float(f"{end_date.year}.{end_date.month:02d}")
            
            # extract entities (nodes) in temporal graph
            poss_match = re.match(r"(.+?)'s\s+(.*?)\s+", event.strip())
            
            if poss_match:
                out_node = poss_match.group(1).strip().lower().replace(' ', '_')
                relation = poss_match.group(2).strip().lower().replace(' ', '_')
                
                # Find all objects in parentheses
                in_nodes = re.findall(r'\(\s*([^)]+)\s*\)', event.strip())
                in_nodes = [node.strip().lower().replace(' ', '_') for node in in_nodes]
            
            else:
                # 3. Fallback: direct pattern: "<subject> <relation> (object)"
                # Assumes format like "Galatasaray S.K. (football) is ( Unknown )"
                direct_match = re.match(r"(.+?)\s+(\w+)\s+", event.strip())
                
                if not direct_match:
                    return None  # unrecognized format, skip or log it
                
                out_node = direct_match.group(1).strip().lower().replace(' ', '_')
                relation = direct_match.group(2).strip().lower().replace(' ', '_')
                
                # Find all objects in parentheses
                in_nodes = re.findall(r'\(\s*([^)]+)\s*\)', event.strip())
                in_nodes = [node.strip().lower().replace(' ', '_') for node in in_nodes]
            
            print(in_nodes)
            for in_node in in_nodes:
                facts += f'event({start_date}, {end_date}, {out_node}, {relation}, {in_node}).\n'
        print(facts)
        print()
        return facts
    elif TG_type == 'TempReason':
        pass
    else:
        raise ValueError(f'TG_type variable must be "TGQA", "TimeQA" or "TempReason", not {TG_type}.')
    


def create_asp_instance_files(dataset) -> None:
    """
    given a TGR dataset generate all samples corresponding instance files containing the corresp
    """
    pass

def reason(instance_path, encoding_path) -> str:
    pass


if __name__ == '__main__':
    test = dict()
    for idx, TG in enumerate(datasets_['TimeQA_Story_TG_Trans']['TG']):
        test[idx] = tg_to_asp(TG, 'TimeQA')
    
    print(datasets_['TimeQA_Story_TG_Trans']['TG'][670])
    print()
    print(test[670])
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import json 
import argparse
from constants import DATA_PATH, EXP_PATH
import seaborn as sns 
import logging 
from datetime import datetime 
from pathlib import Path
from obspy.core import UTCDateTime
import utils 


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


def distplot(events, args):
    
    model = TauPyModel(model=str(EXP_DIR / 'models' / args.model))
    p_ls, s_ls, dist_ls, diff_ls = [], [], [], []
    fig, ax = plt.subplots()
    colors = sns.color_palette("tab10")
    distances_dict = {}

    for dist in np.arange(15, 50, 0.1):
        arrivals = model.get_ray_paths(phase_list=["P", "S"], source_depth_in_km=args.depth, distance_in_degree=dist)
        try:
            p_arrival = arrivals[0].time
            p_ls.append(p_arrival)
            s_arrival = arrivals[1].time
            s_ls.append(s_arrival)
            dist_ls.append(dist)
        except:
            pass
            
        diff = [s - p for s, p in zip(s_ls, p_ls)]
        
    
    ax.scatter(dist_ls, diff, s=0.1, c='k')
    ax.set_title(f'{args.model}')
    ax.set_xlabel('Distance (degrees)')
    ax.set_ylabel('S-P Arrival (sec)')
    
    ymin, ymax = 0, 300 
    counter = 0 
    # print(dist_ls)
    for event, values in events.items():
        time = UTCDateTime(values['end']) - UTCDateTime(values['start'])
        diff_index, actual_diff = min(enumerate(diff), key=lambda x: abs(x[1]-time))
        actual_dist = truncate(dist_ls[diff_index], 1)
        logger.info(f'Diff index: {diff_index} Actual: {actual_diff}')
        logger.info(f'Distance of {event}: {dist_ls[diff_index]}')
        distances_dict[event] = dist_ls[diff_index]
        ax.vlines(dist_ls[diff_index], ymin, ymax, alpha=0.7, color=colors[counter], linestyles='dashed', label=event)
        counter += 1 
        # ax.text(dist_ls[diff_index]-1, 0, (event, actual_dist), size='x-small', rotation=90)

    plt.legend()
    filename = EXP_DIR / 'plots' / f'{args.model}_depth{args.depth}_distances.png'
    fig.savefig(filename)

    filename = EXP_DIR / 'events' / f'{args.model}_depth{args.depth}_distances.pkl'
    utils.save_file(filename, distances_dict)

    return diff, dist_ls


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--exp_dir", required=True, type=str)
    parser.add_argument("--file", required=True, type=str,
                        help="""File containing the p-wave arrival and s-wave arrival times.
                                NOTE: Contents of the file must be in json format and file must be placed under exp_dir/events/ folder. 
                                Example file: 
                                            {
                                            "S0235b":
                                                {	
                                                    "start": "2019-07-26T12:19:18", 
                                                    "end": "2019-07-26T12:22:05" 
                                                },

                                            "S0173a":
                                                {	
                                                    "start": "2019-05-23T02:22:59", 
                                                    "end": "2019-05-23T02:25:53" 
                                                }
                                            }
	
                            """)
    parser.add_argument("--model", required=True, type=str, help="Name of TauP model to use for plotting")
    parser.add_argument("--depth", required=True, type=float, help="Source Depth (km)")
    args = parser.parse_args()

    # Plot settings 
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    EXP_DIR = EXP_PATH / args.exp_dir 
    with open(EXP_DIR / 'events' / args.file) as f: 
        events = json.load(f)

    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running {Path(__file__).name}")
    logger.info(f"{args.__dict__}")

    distplot(events, args)

    
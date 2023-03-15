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
import os 
from itertools import product
import sys 
from collections import defaultdict


def depth_dist_all(args): 
    args.depths = list(map(float, args.depths))
    combs = list(product(args.models, args.depths))

    for event, values in events.items(): 
        print("*"*20)
        print(f'Event: {event}')
        fig, ax = plt.subplots()
        plot_dict = defaultdict(list)  #  Structure: {'model': [(depth1, dist1), (depth2, dist2)...]}

        for model, depth in combs: 
            print(f"Running for: {model}, {depth}")
            model_tm = TauPyModel(model=str(EXP_DIR / 'models' / model))

            p_ls, s_ls, dist_ls, diff_ls = [], [], [], []


            for dist in np.arange(15, 50, 0.1):
                arrivals = model_tm.get_ray_paths(phase_list=["P", "S"], source_depth_in_km=depth, distance_in_degree=dist)
                try:
                    p_arrival = arrivals[0].time
                    p_ls.append(p_arrival)
                    s_arrival = arrivals[1].time
                    s_ls.append(s_arrival)
                    dist_ls.append(dist)
                except:
                    pass
                    
                diff = [s - p for s, p in zip(s_ls, p_ls)]
                

            time = UTCDateTime(values['end']) - UTCDateTime(values['start'])
            diff_index, actual_diff = min(enumerate(diff), key=lambda x: abs(x[1]-time))
            plot_dict[model].append((depth, dist_ls[diff_index]))
            # ax.scatter(depth, dist_ls[diff_index], c=colors[args.models.index(model)], label=f'{model}')

        for m, v in plot_dict.items(): 
            v = np.array(v) 
            depths = v[:, 0] 
            dists = v[:, 1]
            ax.scatter(depths, dists, label=m)

        ax.legend()
        ax.set_title(f'Event: {event}')
        ax.set_xlabel('Depths')
        ax.set_ylabel('Distance')

        filename = EXP_DIR / 'plots' / 'distance_plots' / f'{event}_dist_depth_compare.png'
        fig.savefig(filename)
        plt.close(fig)


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
    parser.add_argument("--models", default=['Combined', 'TAYAK', 'Gudkova', 'NewGudkova'], nargs='+', help="Name of TauP model to use for plotting")
    parser.add_argument("--depths", default=[15, 25, 35, 45, 55],  nargs='+', help="Source Depth (km)")

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
    os.makedirs(EXP_DIR / "plots" / 'distance_plots', exist_ok=True)

    depth_dist_all(args)
    
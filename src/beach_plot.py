import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import beach
from obspy.imaging.source import plot_radiation_pattern

import argparse
from constants import DATA_PATH, EXP_PATH
import logging 
from datetime import datetime 
from pathlib import Path
import utils 
import warnings 
import sys 
import os 


def alphabeach(event, df, n, norm_const, c='b'):
    fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig2, ax2 = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig3, ax3 = plt.subplots(subplot_kw={'aspect': 'equal'})

    # hide axes + ticks
    ax1.axison = False
    ax2.axison = False
    ax3.axison = False 

    max_mis = df['Misfit'].max()
    min_mis = df['Misfit'].min()    

    # Max-min normalization 
    # df['Normalized'] = (max_mis - df['Misfit'])/(max_mis-min_mis)
    
    # Gaussian normalization 
    sigma = max_mis 
    # pref = 1/(sigma*np.sqrt(2*np.pi)) 
    df['Normalized'] = norm_const*np.exp(-(np.square(df['Misfit']))/(2*np.square(sigma)))

    # Plot complete solution set, with n figures in each plot 
    counter = 1
    num_figs = 0

    for index, rows in df.iterrows():
        f = [rows.Strike, rows.Dip, rows.Rake]
        x = 210 * ((counter-1) % 10)
        y = 210 * ((counter-1) // 10)

        # Collection1 is for individual solutions 
        collection1  = beach(f, xy=(x, y), facecolor=c, alpha=rows.Normalized)
        ax1.add_collection(collection1)

        # Collection2 is for superimposed solutions 
        collection2 = beach(f, xy=(0, 0), facecolor=c, alpha=rows.Normalized)
        ax2.add_collection(collection2)

        # Plot set of n solutions per plot 
        if counter % n == 0 and index not in (0, len(df)-1): 
            ax1.autoscale_view(tight=False, scalex=True, scaley=True)
            ax1.set_title(f"{event}: Complete Solution Set [{index-counter+1}-{index}]")
            fig1.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_CS{num_figs}.png', bbox_inches='tight')
            plt.close(fig1) 
            fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
            ax1.axison = False
            num_figs += 1 
            counter = 0 
        # Plot the last remaining set of solutions 
        elif index == len(df)-1: 
            ax1.autoscale_view(tight=False, scalex=True, scaley=True)
            ax1.set_title(f"{event}: Complete Solution Set [{index-counter+1}-{index}]")
            fig1.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_CS{num_figs}.png', bbox_inches='tight')
        # Create a separate plot for the best solution  
        elif index == 0: 
            ax3.set_title(f"{event}: Best fit - Strike {rows.Strike} Dip: {rows.Dip} Rake: {rows.Rake}")
            collection = beach(f, xy=(0, 0), facecolor=c, alpha=rows.Normalized)
            ax3.add_collection(collection)

        counter += 1 

    
    ax2.autoscale_view(tight=False, scalex=True, scaley=True)
    ax2.set_title(f"{event}: Complete Solution Set - Superimposed")

    ax3.autoscale_view(tight=False, scalex=True, scaley=True)

    fig2.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_CS_super.png', bbox_inches='tight')
    fig3.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_best.png', bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    # Model Parameters 
    parser.add_argument("--model", required=True, type=str, help="Model name used for distance calculation")
    parser.add_argument("--depth", required=True, type=float, help="Source Depth (km) used for distance calculation")
    parser.add_argument("--exp_dir", required=True, type=str)

    # Plotting params 
    parser.add_argument("--n", default=50, type=int, 
                        help="Max number of beach balls per plot (param is only used in complete solution set)")
    parser.add_argument("--norm_const", default=0.6, type=float, 
                        help="Normalization constant used to control transparency of the beach balls")
    
    args = parser.parse_args()

    EXP_DIR = EXP_PATH / args.exp_dir 

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    
    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running {Path(__file__).name}")
    logger.info(f"{args.__dict__}")

    SAVE_PATH = EXP_DIR / 'plots' / 'beach_plots'
    os.makedirs(SAVE_PATH, exist_ok=True)

    events_results = utils.load_file(EXP_DIR / 'events' /  f'{args.model}_depth{args.depth}_events_results.pkl')
    for event, values in events_results.items(): 
        print(f"Running for: {event}")
        filename = EXP_DIR / 'events' / f'{event}' /  f'{event}_{args.model}_depth{args.depth}.csv'
        if filename.exists():
            df = pd.read_csv(filename)
            alphabeach(event, df, args.n, args.norm_const)
        else: 
            warnings.warn(f"File not found: {filename}")

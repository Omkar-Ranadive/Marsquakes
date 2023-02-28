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


def alphabeach(event, df, n=50, c='b'):
    fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig2, ax2 = plt.subplots(subplot_kw={'aspect': 'equal'})
    fig3, ax3 = plt.subplots(subplot_kw={'aspect': 'equal'})

    # hide axes + ticks
    ax1.axison = False
    ax2.axison = False
    ax3.axison = False 

    max_mis = df['Misfit'].max()
    min_mis = df['Misfit'].min()    

    df['Normalized'] = (max_mis - df['Misfit'])/(max_mis-min_mis)

    for index, rows in df.iterrows():
        f = [rows.Strike, rows.Dip, rows.Rake]
        x = 210 * (index % 10)
        y = 210 * (index // 10)

        if index == 0: 
            ax3.set_title(f"{event}: Best fit - Strike {rows.Strike} Dip: {rows.Dip} Rake: {rows.Rake}")
            collection = beach(f, xy=(0, 0), facecolor=c, alpha=rows.Normalized)
            ax3.add_collection(collection)

        collection1  = beach(f, xy=(x, y), facecolor=c, alpha=rows.Normalized)
        collection2 = beach(f, xy=(0, 0), facecolor=c, alpha=rows.Normalized)
        # Only plot top n solution in case of solution set plot (for readibility)
        if index < n: 
            ax1.add_collection(collection1)
        ax2.add_collection(collection2)
    
    ax1.autoscale_view(tight=False, scalex=True, scaley=True)
    ax1.set_title(f"{event}: Complete Solution Set")

    ax2.autoscale_view(tight=False, scalex=True, scaley=True)
    ax2.set_title(f"{event}: Complete Solution Set - Superimposed")

    ax3.autoscale_view(tight=False, scalex=True, scaley=True)

    fig1.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_CS.png', bbox_inches='tight')
    fig2.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_CS_super.png', bbox_inches='tight')
    fig3.savefig(SAVE_PATH / f'{args.model}_depth{args.depth}_{event}_best.png', bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Model Parameters -
    parser.add_argument("--model", required=True, type=str, help="Model name used for distance calculation")
    parser.add_argument("--depth", required=True, type=float, help="Source Depth (km) used for distance calculation")
    parser.add_argument("--exp_dir", required=True, type=str)

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
            alphabeach(event, df)
        else: 
            warnings.warn(f"File not found: {filename}")

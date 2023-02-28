import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import argparse
import logging
from constants import DATA_PATH, EXP_PATH
import os 
from itertools import product 
import logging 
from datetime import datetime 
from pathlib import Path
from obspy.imaging.beachball import beach


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, required=True)
    parser.add_argument("--dir2", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="comparision_plots")
    parser.add_argument('--models',  nargs='+', default= ['Combined', 'NewGudkova']) 
    parser.add_argument("--depths", nargs="+", default=[15, 35, 55])
    args = parser.parse_args()

    depths = list(map(float, args.depths)) 
    DIR1 = EXP_PATH / args.dir1 / 'events' 
    DIR2 = EXP_PATH / args.dir2 / 'events'
    SAVE_DIR = EXP_PATH / args.save_dir 
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=str(SAVE_DIR / 'sdr.log'), format='%(message)s', filemode='w')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running {Path(__file__).name}")
    logger.info(f"{args.__dict__}")

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'
    sns.set_theme(style='whitegrid')

    combs = list(product(depths, args.models)) 
    n = 5 
    for event in os.listdir(DIR1): 
        if os.path.isdir(DIR1 / event) and os.path.exists(DIR2 / event):
            for depth, model in combs: 
                fig, ax = plt.subplots()
                fig2, ax2 = plt.subplots(subplot_kw={'aspect': 'equal'})
                ax2.axison = False
                filename = f'{event}_{model}_depth{depth}.csv'
                if os.path.exists(DIR1 / event / filename) and os.path.exists(DIR2 / event / filename): 
                    logging.info("*"*30)
                    logging.info(f"Event: {event}")
                    df1 = pd.read_csv(DIR1 / event / filename)
                    df2 = pd.read_csv(DIR2 / event / filename) 
                    misfit1 = df1['Misfit'].to_numpy()
                    misfit2 = df2['Misfit'].to_numpy()

                    sdr1 = df1[['Strike', 'Dip', 'Rake']].head(n) 
                    sdr2 = df2[['Strike', 'Dip', 'Rake']].head(n) 
                    logging.info(f"{args.dir1}: Top {n} solutions (strike, dip, rake)")
                    logging.info(sdr1) 
                    logging.info(f"{args.dir2}: Top {n} solutions (strike, dip, rake)")
                    logging.info(sdr2) 

                    ax.plot(range(0, len(misfit1)), misfit1, label=f'{args.dir1} dir')
                    ax.plot(range(0, len(misfit2)), misfit2, label=f'{args.dir2} dir') 
                    ax.set_ylabel('Misfit Value')
                    ax.set_title(f'{event}-{model}-{depth}')
                    ax.legend()
                    fig.savefig(SAVE_DIR / f'{args.dir1}_{args.dir2}_{event}_{model}_depth{depth}.png')

                    # Plot beach plot to compare top 5 strike, dip, rake of both csvs 
                    for index in range(n):
                        f1 = sdr1.iloc[[index]].to_numpy()
                        x = 210 * (index % n)
                        y = 210 * (index // n)
                        collection  = beach(f1.squeeze(), xy=(x, y), facecolor='b')
                        ax2.add_collection(collection)

                    for index in range(n):
                        f2 = sdr2.iloc[[index]].to_numpy()
                        x = 210 * ((n + index) % n)
                        y = 210 * ((n + index) // n)
                        collection  = beach(f2.squeeze(), xy=(x, y), facecolor='r')
                        ax2.add_collection(collection)

            
                    ax2.autoscale_view(tight=False, scalex=True, scaley=True)
                    ax2.set_title(f"Blue: {args.dir1}: Red: {args.dir2}")                          
                    fig2.savefig(SAVE_DIR / f'{args.dir1}_{args.dir2}_{event}_{model}_depth{depth}_beach.png')

                else: 
                    print(f"Warning: {filename} not found!")
                plt.close(fig)
                plt.close(fig2)


                
            



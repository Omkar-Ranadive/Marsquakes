from obspy.taup import TauPyModel, plot_travel_times
from obspy.taup.taup_create import build_taup_model
from constants import DATA_PATH, EXP_PATH
import argparse
import seaborn as sns 
import os 
import matplotlib.pyplot as plt 
import logging 
from datetime import datetime
from pathlib import Path


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument("-m", '--mod_list',  nargs='+', default=['Gudkova', 'Combined', 'NewGudkova', 'TAYAK'], 
                        help="""List of models to be converted to taup model format""")
    parser.add_argument("--exp_dir", default='results', type=str)

    # Parameters related to ray plots 
    parser.add_argument("--rayplot", default=False, action=argparse.BooleanOptionalAction, 
                        help="If true, then plot ray paths for each model. Pass --no-rayplot to prevent plotting.")
    parser.add_argument("--phase_list", nargs='+', default=['P', 'pP', 'sP', 'S', 'sS'],
                        help="List of phases for which rayplot will be plotted")
    parser.add_argument("--source_depth", default=35, type=float, help="Source depth in km")
    parser.add_argument("--deg_start", default=10, type=float, 
                        help="Starting distance in degrees. Rayplots will be calculated for all degrees in range (start, end, step) ")
    parser.add_argument("--deg_end", default=60, type=float, help="Ending distance in degrees.")
    parser.add_argument("--step", default=10, type=float, help="Step size to cycle through the degrees")
    parser.add_argument("--plot_type", default='cartesian', type=str, 
                        help="Refer https://docs.obspy.org/packages/obspy.taup.html for different types")

    args = parser.parse_args()

    MODELS_PATH = DATA_PATH / 'models' 
    EXP_DIR = EXP_PATH / args.exp_dir
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(EXP_DIR / 'models', exist_ok=True) 
    os.makedirs(EXP_DIR / 'events', exist_ok=True) 
    os.makedirs(EXP_DIR / "plots", exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running model build ")
    logger.info(f"{args.__dict__}")

    # Plot settings 
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    # Convert the models in .nd format to taup format 
    for model in os.listdir(MODELS_PATH): 
        if os.path.splitext(model)[0] in args.mod_list: 
            build_taup_model(str(MODELS_PATH / model), output_folder=EXP_DIR / 'models')

    # Plot ray paths for the taup models 
    if args.rayplot: 
        for model in os.listdir(EXP_DIR / 'models'): 
            model_name = os.path.splitext(model)[0] 
            if model_name in args.mod_list: 
                taup_model = TauPyModel(EXP_DIR / 'models' / model) 
                for dist_deg in range(args.deg_start, args.deg_end+1, args.step): 
                    arrivals = taup_model.get_ray_paths(source_depth_in_km=args.source_depth, distance_in_degree=dist_deg, 
                                                        phase_list=args.phase_list)
                    logger.info(f"{model_name} arrival times")
                    logger.info(f"{arrivals}")
                    fig = plt.figure()
                    ax = arrivals.plot_rays(plot_type=args.plot_type, fig=fig, show=False, legend=True)
                    ax.set_title(f'{model_name}')
                    filename = f'{model_name}_{args.plot_type}_depth{args.source_depth}_deg{dist_deg}.png'
                    fig.savefig(EXP_DIR / 'plots' / filename) 
                    plt.close(fig)




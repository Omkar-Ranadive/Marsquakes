from obspy.taup import TauPyModel, plot_travel_times
from obspy.taup.taup_create import build_taup_model
from constants import DATA_PATH
import argparse
import seaborn as sns 
import os 
import matplotlib.pyplot as plt 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument("-m", '--mod_list',  nargs='+', default=['Gudkova', 'Combined', 'NewGudkova', 'TAYAK'], 
                        help="""List of models to be converted to taup model format""")
    parser.add_argument("--save_dir", default='models_taup', type=str)

    # Parameters related to ray plots 
    parser.add_argument("--rayplot", default=True, action=argparse.BooleanOptionalAction, 
                        help="If true, then plot ray paths for each model. Pass --no-rayplot to prevent plotting.")
    parser.add_argument("--phase_list", nargs='+', default=['P', 'pP', 'sP', 'S', 'sS'],
                        help="List of phases for which rayplot will be plotted")
    parser.add_argument("--source_depth", default=35, type=float, help="Source depth in km")
    parser.add_argument("--dist_deg", default=38, type=float, help="Distance in degrees")
    parser.add_argument("--plot_type", default='cartesian', type=str, 
                        help="Refer https://docs.obspy.org/packages/obspy.taup.html for different types")

    args = parser.parse_args()

    MODELS_PATH = DATA_PATH / 'models' 
    SAVE_PATH = DATA_PATH / args.save_dir

    # Plot settings 
    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    # Convert the models in .nd format to taup format 
    for model in os.listdir(MODELS_PATH): 
        if os.path.splitext(model)[0] in args.mod_list: 
            build_taup_model(str(MODELS_PATH / model), output_folder=SAVE_PATH)

    # Plot ray paths for the taup models 
    if args.rayplot: 
        os.makedirs(SAVE_PATH / "plots", exist_ok=True)

        for model in os.listdir(SAVE_PATH): 
            model_name = os.path.splitext(model)[0] 
            if model_name in args.mod_list: 
                taup_model = TauPyModel(SAVE_PATH / model) 
                arrivals = taup_model.get_ray_paths(source_depth_in_km=args.source_depth, distance_in_degree=args.dist_deg, 
                                                    phase_list=args.phase_list)
                
                fig = plt.figure()
                ax = arrivals.plot_rays(plot_type=args.plot_type, fig=fig, show=False, legend=True)
                ax.set_title(f'{model_name}')
                filename = f'{model_name}_{args.plot_type}_depth{args.source_depth}_deg{args.dist_deg}.png'
                fig.savefig(SAVE_PATH / 'plots' / filename) 
                plt.close(fig)




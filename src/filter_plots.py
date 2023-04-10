import matplotlib.pyplot as plt
import numpy as np
from obspy.clients.fdsn import Client
import argparse
from constants import EXP_PATH, DATA_PATH 
import json
import sys 
from obspy.core import UTCDateTime
import logging 
from datetime import datetime
from pathlib import Path
import utils 
import warnings 
import os 
import seaborn as sns 


def wfplot(st, axs, iax, scale=1, sh=0, Ptime=0, title=""):
    if Ptime == 0: Ptime = st[0].stats.starttime
    time_axis = st[0].times(reftime=Ptime)
    n = min([len(time_axis)]+[len(st[i].data) for i in np.arange(len(st))])

    a = 1/scale
    ylim = [-2,10]
    shift = sh
    dsh = 3
    
    for trace in st:
        axs[iax][0].plot(time_axis[:n], a*trace.data[:n] + shift, label = trace.stats.channel, alpha=0.9)
        shift += dsh
    axs[iax][0].set_ylim(ylim)
    sENZ = utils.uvw2enz(st)
    shift = sh
    for trace in sENZ:
        axs[iax][1].plot(time_axis[:n], a*trace.data[:n] + shift, label = trace.stats.channel, alpha=0.9)
        shift += dsh
    axs[iax][1].set_ylim(ylim)

    axs[iax][0].set_title(title)
    axs[iax][1].set_title(title)

    if iax == len(axs)-1: 
       axs[len(axs)-1][0].legend(loc='lower right')  
       axs[len(axs)-1][1].legend(loc='lower right') 

    return axs 


def filter_data(events, args): 
    bpfilters = [
            [2,8],
            [1.0,4],
            [0.125,1.0],
            [0.03,0.125]
        ]
    nrows = len(bpfilters)+3

    scales = list(map(int, args.scales))
    
    if len(scales) < nrows: 
        last_scale = scales[-1] 
        diff = nrows - len(scales)
        scales.extend([last_scale]*diff) 
    
    print(f"Scales used: {scales}") 

    for event, values in events.items(): 
        print(event)
        logger.info("*"*30)
        logger.info(f"Calculating for event {event}")
        pstart = UTCDateTime(values['start']) 
        sstart = UTCDateTime(values['end'])
        start = pstart - 60  # 1 minute earlier 
        end = sstart + 5*(sstart - pstart) 

        fig, axs = plt.subplots(nrows, 2, figsize=(16, 10))
        fig.suptitle(f'Event: {event}')

        fig.tight_layout(pad=1)
        iax = 0
        
        st, inv = utils.get_waveforms(start, end, event, args)
        st.detrend(type='simple')

        sps = st[0].stats.sampling_rate
        dt = 1/sps

        if 2*bpfilters[0][1] > sps: 
            warnings.warn(f'High bandpass corner too high. f_nyq = {0.5*sps}Hz')
        shift = 0

        # plot raw data
        stmp = st.slice(starttime=start,endtime=end)
        #time_axis = np.arange(start173a-P173a,end173a-P173a+dt,dt)
        axs = wfplot(stmp, axs, iax, scales[iax], shift, pstart, title='Raw Data')
        iax += 1

        # plot high-passed data
        stmp = st.copy()
        stmp.taper(0.01,max_length=1)
        stmp.filter('highpass',freq=bpfilters[0][1], corners=4, zerophase=True)
        stm = stmp.slice(starttime=start,endtime=end)
        #time_axis = np.arange(start173a-P173a,end173a-P173a+dt,dt)
        axs = wfplot(stm, axs, iax, scales[iax], shift, pstart, title=f'High-Passed Data {bpfilters[0][1]}Hz')
        iax += 1

        # plot band-passed data
        for f in bpfilters:
            stmp = st.copy()
            stmp.taper(0.01,max_length=1)
            stmp.filter('bandpass',freqmin=f[0], freqmax=f[1],corners=4, zerophase=True)
            stm = stmp.slice(starttime=start,endtime=end)
            #time_axis = np.arange(start173a-P173a,end173a-P173a+dt,dt)
            scale = 2000 
            axs = wfplot(stm, axs, iax, scales[iax], shift, pstart, title=f'Band-Passed Data {f[0]}-{f[1]}Hz')
            iax += 1

        # plot low-passed data
        stmp = st.copy()
        stmp.taper(0.01, max_length=1)
        stmp.filter('lowpass',freq=bpfilters[-1][0], corners=4, zerophase=True)
        stm = stmp.slice(starttime=start, endtime=end)
        #time_axis = np.arange(start173a-P173a,end173a-P173a+dt,dt)
        axs = wfplot(stm, axs, iax, scales[iax], shift, pstart, title=f'Low-Passed Data {bpfilters[-1][0]}Hz')
        fig.savefig(EXP_DIR / 'plots' / 'filter_plots' / f'{event}_filtered_data.png')
        plt.close(fig)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    # Obspy Parameters 
    parser.add_argument("--net", default='XB', type=str, help='network code')
    parser.add_argument("--sta", default='ELYSE', type=str, help='station code') 
    parser.add_argument("--loc", default='02', type=str, help='location code') 
    parser.add_argument('--cl', nargs='+', default=['B*'], help='list of channel codes')
    parser.add_argument("--client", default='IRIS', type=str, help="Server to use to retrieve data")

    # Event parameters 
    parser.add_argument("--start", default="2019-07-26T12:19:18", type=str, help="Start time of the event")
    parser.add_argument("--end", default="2019-07-26T12:22:05", type=str, help="End time of the event")
    parser.add_argument("--ename", default="S0235b", type=str, help="Event name") 
    parser.add_argument("--file", default=None, 
                        help="""Multiple events can be provided in the form of a json file and file must be placed under exp_dir/events/ folder.
                        NOTE: Previous event parameters will be ignored if a file name is provided.
                        Example file: 
                                    {
                                    "S0235b":
                                        {	
                                            "start": "2019-07-26T12:19:18", 
                                            "end": "2019-07-26T12:19:18" 
                                        },
                                        
                                    "S0183a":
                                        {	
                                            "start": "2019-06-03T02:27:49", 
                                            "end": "2019-06-03T02:32:15" 
                                        }
                                    }
                        """)
    parser.add_argument("--adjtime", default=600, type=float, help="Retrieved waveform is of the duration start-adjtime, end+adjtime")
    parser.add_argument("--scales", default=[200], nargs="+", 
                        help="""Data gets multipled by 1/scale before back azimuth calculation. 
                        A different scale can be passed for each filter plot. Example -  [80000, 5000, 80000, 80000, 5000, 2000, 1000]
                        If a single scale is passed, it will be used across all plots""")
    parser.add_argument("--save", default=True, action=argparse.BooleanOptionalAction, 
                        help="If true, it would save the waveforms / load from saved data if it already exists")
    
    # Exp Parameters 
    parser.add_argument("--exp_dir", required=True, type=str)

    # Plot params 
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    sns.set_theme(style='whitegrid')

    args = parser.parse_args()

    client = Client(args.client)
    EXP_DIR = EXP_PATH / args.exp_dir 


    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running {Path(__file__).name}")
    logger.info(f"{args.__dict__}")

    if args.file: 
        with open(EXP_DIR / 'events' / args.file) as f: 
            events = json.load(f)
    else: 
        events = {args.ename: {'start': args.start, 'end': args.end}}
    
    
    os.makedirs(EXP_DIR / "plots" / 'filter_plots', exist_ok=True)
    filter_data(events, args)
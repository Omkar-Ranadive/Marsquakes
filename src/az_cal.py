import matplotlib.pyplot as plt
import numpy as np
from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import argparse
from constants import EXP_PATH
import json
import sys 
from obspy.core import UTCDateTime
import logging 
from datetime import datetime
from pathlib import Path
import utils 
import warnings 
import os 


def waveforms(start, end, args):
    st = client.get_waveforms(args.net, args.sta, args.loc, ",".join(args.cl), start-args.adjtime, end+args.adjtime, attach_response=True)
    st.detrend(type='simple')
    st_disp = st.copy()
    st_disp.remove_response(pre_filt=[0.006, 0.01, 8, 9.9], water_level=None, zero_mean=False, 
                            taper=True, taper_fraction=0.01, output='DISP')
    return st_disp


def uvw2enz(st):
    if len(st) != 3:
       print('Stream does not contain 3 Traces')
       return st
    for trace in st:
        head = trace.stats
        channel = head.channel
        if channel == 'BHU': U = trace.data
        elif channel == 'BHV': V = trace.data
        elif channel == 'BHW': W = trace.data
        else:
            print('Trace.channel is not BHU, BHV, or BHW')
            return st

    d = np.radians(-30)
    aU = np.radians(135)
    aV = np.radians(15)
    aW = np.radians(255)

    A = np.array([[np.cos(d)*np.sin(aU), np.cos(d)*np.cos(aU),-np.sin(d)],
                  [np.cos(d)*np.sin(aV), np.cos(d)*np.cos(aV), -np.sin(d)],
                  [np.cos(d)*np.sin(aW), np.cos(d)*np.cos(aW), -np.sin(d)]])

    B = np.linalg.inv(A)
    E,N,Z = np.dot(B,(U,V,W))

    head.channel = 'BHE'; trE = Trace(data=E, header=head)
    head.channel = 'BHN'; trN = Trace(data=N, header=head)
    head.channel = 'BHZ'; trZ = Trace(data=Z, header=head)
    stENZ = Stream(traces=[trE,trN,trZ])

    return stENZ


def rotate(c1,c2,a):
    """
    IN: c1,c2 (arrays) and a (angle)
    c1 c2 are the X and Y axes, respectively of a Cartesian coordinate system
    a is an angle in degrees, positive angle means a clockwise rotation of the coordinate system.
    OUT: o1, o2 (arrays)
    o1 o2 are the X and Y axes, respectively of a rotated Cartesian coordinate system
    """
    o1 = np.cos(np.radians(a))*c1 - np.sin(np.radians(a))*c2
    o2 = np.sin(np.radians(a))*c1 + np.cos(np.radians(a))*c2
    return o1, o2


def calculate_baz(events, args): 
    baz_angles = {}
    
    for event, values in events.items(): 
        print(event)
        logger.info("*"*30)
        logger.info(f"Calculating for event {event}")
        start = UTCDateTime(values['start'])
        end = UTCDateTime(values['end'])

        st_uvw = waveforms(start, end, args)
        print(st_uvw)
        st_z12 = uvw2enz(st_uvw)

        stf = st_z12.copy()
        stf.filter('bandpass', freqmin=0.125, freqmax=1.0, corners=4, zerophase=True)
        
        stP = stf.slice(starttime=start-1,endtime=start+2)
        stS = stf.slice(starttime=end-2, endtime=end+10)


        # Calculate noise tolerance level 
        st_tols = []
        scale = 1/args.scale
        noise_start, noise_end = start-1, start+2 
        for i in range(args.tol_it):
            noise_start = noise_start - args.tol_dur 
            noise_end = noise_end - args.tol_dur
            stPi = stf.slice(starttime=noise_start, endtime=noise_end) 
            hhe = scale * stPi[0].data
            hhn = scale * stPi[1].data
            st_tols.append(np.dot(hhe, hhe))
            st_tols.append(np.dot(hhn, hhn))

        tol_level = np.mean(st_tols)
        logger.info(f"Noise tolerance level: {tol_level}")

        # stP2 and stS2 are used for visualization purposes only 
        stP2 = stf.slice(starttime=start-10,endtime=start+10)
        stS2 = stf.slice(starttime=end-10, endtime=end+10)
        filename1 = EXP_DIR / 'plots' / 'wave_plots' / f'{args.model}_depth{args.depth}_{event}_pwave.png'
        filename2 = EXP_DIR / 'plots' / 'wave_plots' / f'{args.model}_depth{args.depth}_{event}_swave.png'

        stP2.plot(outfile=filename1)
        stS2.plot(outfile=filename2)

        hhe = scale * stP[0].data
        hhn = scale * stP[1].data

        energies = []
        angles = np.arange(0, 360, 1)

        # calculate Energy on channels oriented in the a direction, for all a in alpha:
        # angle a is relative to orientation of channels 1 and 2.
        # c1 is the x-axis, c2 is the y-axis
        for a in angles:
            hhT, hhR = rotate(hhe,hhn,a)
            Tenergy = np.dot(hhT,hhT)
            energies.append(Tenergy)

        energies = np.array(energies) 
        sol_indices = np.where(energies < tol_level)[0]
        logger.info(f"Solution angles < noise tolerance level: {sol_indices}")
        min_angle = np.argmin(energies)
        min_energy = energies[min_angle]
        baz_angles[event] = min_angle 
        energy_reported = None 
        if 'baz' in values: 
            logger.info(f"Calculated Baz: {min_angle} or {min_angle-180}. Reported Baz: {values['baz']}")
            angle_reported = int(values['baz'])
            energy_reported = energies[angle_reported]
            # Angle at which energy is minimized
            logger.info(f"Energy at cal angle: {min_energy}, Energy at reported angle {energy_reported}")
        else: 
            logger.info(f"Calculated Baz: {min_angle} or {min_angle-180}")
            logger.info(f"Energy at cal angle: {min_energy}")

        # Plot the energies 
        fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

        alphas = np.where(np.in1d(angles, sol_indices), 0.6, 0.5)
        sizes = np.where(np.in1d(angles, sol_indices), 8, 5)
        colors = np.where(np.in1d(angles, sol_indices), 'orange', 'b')

        ax1[0].scatter(angles, energies, alpha=alphas, s=sizes, c=colors) 
        ax1[0].hlines(tol_level, np.min(angles), np.max(angles), alpha=0.5, color='k', linestyles='dashed', label='Tolerance Level')
        ax1[0].vlines(min_angle, np.min(energies), np.max(energies), alpha=0.7, color='r', linestyles='dashed', label='Cal min energy')

        ax1[1].scatter(sol_indices, energies[sol_indices], c='orange', s=10, alpha=0.8, label='Angles < tol')
        ax1[1].scatter(min_angle, min_energy, c='r', marker='^', s=80, alpha=0.8, label=f'Cal angle: {min_angle}')
        if energy_reported: 
            ax1[0].vlines(angle_reported, np.min(energies), np.max(energies), alpha=0.7, color='g', linestyles='dashed', label='Reported min energy')
            ax1[1].scatter(angle_reported, energy_reported, marker='^', c='g', alpha=0.8, s=80, label=f'Reported angle: {angle_reported}')

        ax1[1].set_xlabel('Angles')
        ax1[0].set_ylabel('Energy')
        ax1[1].set_ylabel('Energy')
        ax1[0].legend()
        ax1[1].legend()
        filename3 = EXP_DIR / 'plots' / 'wave_plots' / f'{args.model}_depth{args.depth}_{event}_energies.png'
        fig1.savefig(filename3, bbox_inches='tight')
        # Rotate the s-wave and plot 
        hhe = stS2[0].data
        hhn = stS2[1].data
        rotation = 180 + min_angle if min_angle > 0 else 360 - min_angle 
        hhT, hhR = rotate(hhe, hhn, rotation)

        # Plot the rotated s-wave on the larger slice but mark the smaller slice with vertical lines 
        fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
        times = [(stS2[0].stats.starttime + t).datetime for t in stS2[0].times()]
        labels = ['T Component', 'R Component', 'Z Component']
        for i in range(3): 
            ymin, ymax = np.min(stS2[i].data), np.max(stS2[i].data) 
            ax2[i].plot(times, stS2[i].data, label=labels[i], c='k')
            ax2[i].vlines(stS[i].stats.starttime, ymin, ymax, alpha=0.7, color='r', linestyles='dashed')
            ax2[i].vlines(stS[i].stats.endtime, ymin, ymax, alpha=0.7, color='r', linestyles='dashed')

        # ax.legend()
        filename4 = EXP_DIR / 'plots' / 'wave_plots' / f'{args.model}_depth{args.depth}_{event}_swave_rotated_{rotation}.png'
        fig2.savefig(filename4, bbox_inches='tight')

        plt.close('all')
        # streamRT = stS.copy()
        # streamRT[0].data = hhT
        # streamRT[1].data = hhR
        # streamRT[0].stats.component = 'T'
        # streamRT[1].stats.component = 'R'
        # streamRT.plot(outfile=filename3)

    return baz_angles


def azdelt(deld, az):
    """
    Compute destination lat,lon, given a starting lat,lon, bearing (azimuth), and a distance to travel
    Travel is along a great circle.
    IN:  lat1, lon1: lattitude na dlongitude of starting point
     deld and az: distance and azimuth (both in degrees) (azimuth is measured clockwise from N)
     OUT: lat2, lon2 : latitude and longitude of destination point (in degrees)
    """

    teta1 = np.radians(4.5024)
    fi1 = np.radians(135.6234)
    delta = np.radians(deld)
    azimuth = np.radians(az)

    if teta1 > 0.5*np.pi or teta1 < -0.5*np.pi:
       print('error, non-existent latitude')
       return 0,0

    if delta < 0.:
       print('error, non-existent delta')
       return 0,0

    term1 = np.cos(delta)*np.sin(fi1)*np.cos(teta1)
    factor2 = np.sin(delta)*np.cos(azimuth)*np.sin(teta1)

    term2 = factor2*np.sin(fi1)
    factor3 = np.sin(delta)*np.sin(azimuth)

    term3 = factor3*np.cos(fi1)
    teller = term1 - term2 + term3

    term1 = np.cos(delta)*np.cos(fi1)*np.cos(teta1)
    term2 = factor2*np.cos(fi1)
    term3 = factor3*np.sin(fi1)
    rnoemer = term1 - term2 - term3

    fi2 = np.arctan2(teller, rnoemer)

    term1 = np.cos(delta)*np.sin(teta1)
    term2 = np.sin(delta)*np.cos(azimuth)*np.cos(teta1)
    som = term1 + term2
    teta2 = np.arcsin(som)

    return np.degrees(teta2), np.degrees(fi2)


def deltaz(deta2, di2):
    '''
    IN: long1, lat1, long2, lat2
    OUT: distance in degrees, azimuth to go from point 1 to point 2 and azimuth to go from point 2 to point 1
    '''

    teta1 = np.radians(4.5024)
    fi1 = np.radians(135.6234)
    teta2 = np.radians(deta2)
    fi2 = np.radians(di2)

    c1 = np.cos(teta1); c2 = np.cos(teta2)
    s1 = np.sin(teta1); s2 = np.sin(teta2)
    c21 = np.cos(fi2-fi1); s21 = np.sin(fi2-fi1)
    som  = s1*s2 + c21*c1*c2

    delta = np.arccos(som)

    teller = c2*s21
    rnoemer =  s2*c1 - c2*s1*c21
    azimuth = np.arctan2(teller,rnoemer)
    numerator = -s21*c1
    denominator =  s1*c2 - c1*s2*c21
    baz = np.arctan2(numerator, denominator)

    return np.degrees(delta), np.degrees(azimuth), np.degrees(baz)


def calculate_az(baz_angles): 
    event_distances = utils.load_file(EXP_DIR / 'events' / f'{args.model}_depth{args.depth}_distances.pkl')
    event_results = {}
    for event, mina in baz_angles.items(): 
        logger.info("*"*30)

        if event in event_distances:
            distance = event_distances[event] 
            lat, lon = azdelt(distance, mina)
            # deltaz is returing dist, az, baZ but it's from lander to event, so we reverse it after calling the function (to get it from event to lander)
            dist, bAz, az = deltaz(lat, lon)  
            logger.info(f"Event: {event}")
            logger.info(f"Latitude {lat} : Longitude: {lon}")
            logger.info(f"Distance in degrees: {dist}")
            logger.info(f"bAz in degrees: {bAz}")
            logger.info(f"az in degrees: {az}")
            event_results[event] = {'latitude': lat, 'longitude': lon, 'distance': dist, 'baz': bAz, 'az': az}
        else: 
            warnings.warn(f"Event {event} not found in distances dict. Skipping this event") 
            continue 
    
    utils.save_file(EXP_DIR / 'events' / f'{args.model}_depth{args.depth}_events_results.pkl', event_results)
              
              
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
    parser.add_argument("--scale", default=200, type=float, help="Data gets multipled by 1/scale before back azimuth calculation")
    
    # Model Parameters
    parser.add_argument("--model", required=True, type=str, help="Model name used for distance calculation")
    parser.add_argument("--depth", required=True, type=float, help="Source Depth (km) used for distance calculation")

    # Noise Tolerance calculation params 
    # These two quantities will be used to calculate the noise-tolerance level 
    parser.add_argument("--tol_dur", default=3, type=float, 
                        help="Length of slices calculated before the start of p-wave")
    parser.add_argument("--tol_it", default=3, type=int, help="Number of slices calculated before start of p-wave")

    # Exp Parameters 
    parser.add_argument("--exp_dir", required=True, type=str)

    # Plot params 
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'


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
    
    
    os.makedirs(EXP_DIR / "plots" / 'wave_plots', exist_ok=True)

    baz_angles = calculate_baz(events, args)
    calculate_az(baz_angles)    

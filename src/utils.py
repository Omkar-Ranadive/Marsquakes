import pickle 
import numpy as np 
from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
import re 
from constants import DATA_PATH 
import os 


def save_file(filename, file):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
    """

    with open(filename, 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filename):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)

    return file


def uvw2enz(st):
    """
    Rotate the waveforms 
    """
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


def get_waveforms(start, end, event, args):
    client = Client(args.client)
    channels = ",".join(args.cl) 
    # To save file with the * and ',' in channel codes, use re to replace them 
    channels = re.sub(r'\*', 'all', channels)
    channels = re.sub(r',', '_', channels)

    filename = f'{event}_{args.net}_{args.sta}_{args.loc}_{channels}_{args.adjtime}.mseed'
    os.makedirs(DATA_PATH / 'events', exist_ok=True)

    SAVE_PATH = DATA_PATH / 'events' / filename
    if os.path.exists(SAVE_PATH) and args.save:
        print("Loading from saved data")
        st = read(str(SAVE_PATH))
        inv = read_inventory(str(SAVE_PATH.with_suffix('.xml')))
    else: 
        st = client.get_waveforms(args.net, args.sta, args.loc, ",".join(args.cl), start-args.adjtime, end+args.adjtime, attach_response=True)
        inv = client.get_stations(network=args.net, station=args.sta, location=args.loc, channel=",".join(args.cl), 
                                  starttime=start-args.adjtime, endtime=end+args.adjtime, level="response")
        if args.save: 
            print(f"Saving data to: {SAVE_PATH}")
            st.write(SAVE_PATH, format='MSEED') 
            inv.write(SAVE_PATH.with_suffix('.xml'), format='STATIONXML')

    return st, inv 

import matplotlib.pyplot as plot
import numpy as np
from obspy.taup import TauPyModel
import pandas as pd
from pathlib import Path
import argparse
import logging
from constants import DATA_PATH, EXP_PATH
from datetime import datetime 
import utils 
import json 
import os 


def azdp(v):
       # (c) 1994 Suzan van der Lee
       vr=v[0]; vt=v[1]; vp=v[2]
       dparg = np.sqrt(vt*vt + vp*vp)
       if dparg>1.:
          dparg = 1.
          print('argument error for np.arccos: ',dparg,'  Set to 1.')
       if dparg<-1.:
          dparg = -1.
          print('argument error for np.arccos: ',dparg,'  Set to -1.')
       vdp = np.arccos(dparg)
       dp = halfpi - vdp
       vaz = halfpi + np.arctan2(vt,vp)
       if vr>0.: vaz = vaz + np.pi
       st = vaz + halfpi
       if st>=twopi: st = st - twopi
       if vaz>=twopi: vaz = vaz - twopi
       vaz = vaz*rad; st = st*rad; vdp = vdp*rad; dp = dp*rad
       return vaz,vdp,st,dp


def getmt(fault):
    # (c) 1994 Suzan van der Lee
    rad = 180./np.pi; m0 = 1
    st,dp,rk = fault

    st = st/rad; dp = dp/rad; rk = rk/rad
    sd = np.sin(dp); cd = np.cos(dp)
    sd2 = np.sin(2*dp); cd2 = np.cos(2*dp)
    ss = np.sin(st); cs = np.cos(st)
    ss2 = np.sin(2*st); cs2 = np.cos(2*st)
    sr = np.sin(rk); cr = np.cos(rk)

    # formulas from Aki & Richards Box 4.4
    # mt(1-6): Mrr, Mtt, Mff, Mrt, Mrf, Mtf
    # mt(1-6): Mzz, Mxx, Myy, Mzx, -Mzy, -Mxy
    mt = [ sr*sd2, -1.*sd*cr*ss2 - sd2*sr*ss*ss, sd*cr*ss2 - sd2*sr*cs*cs, -1.*cd*cr*cs - cd2*sr*ss, cd*cr*ss - cd2*sr*cs, -1.*sd*cr*cs2 - 0.5*sd2*sr*ss2]
    return mt


def getplanes(xm):
    """
    needs function azdp. - converts MT to DC
    IN: xm = list of moment tensor elements in Harvard order (GCMT)
    OUT: strike dip rake (twice). Also: P and T vectors
    (c) 1994 Suzan van der Lee
    """
    xmatrix = np.array([[xm[0],xm[3],xm[4]],[xm[3],xm[1],xm[5]],[xm[4],xm[5],xm[2]]])
    tr=(xm[0]+xm[1]+xm[2])/3.
    if np.abs(tr)>eps:
       xmatrix[0,0] = xmatrix[0,0] - tr
       xmatrix[1,1] = xmatrix[1,1] - tr
       xmatrix[2,2] = xmatrix[2,2] - tr
    #print('removed isotropic component from Moment Tensor:')
    d,pt = np.linalg.eigh(xmatrix)
    jt = np.argmax(d) ; dmax = d[jt]
    jp = np.argmin(d) ; dmin = d[jp]
    for j in [0,1,2]:
        if j!=jp and j!=jt: jn=j
    if (jn+jp+jt)!=3:
        print('ERROR in axis determination')
        return 0

    p = pt[:,jp]
    t = pt[:,jt]
    n = pt[:,jn]
    if p[0] < 0.: p = -1.*p
    if t[0] < 0.: t = -1.*t
    pole1 = (t+p)/np.sqrt(2.)
    pole2 = (t-p)/np.sqrt(2.)
    if p[0] > t[0]: pole1 = -1.*pole1
    # planes' poles not part of function output, but they could be in future
    azt,dpt,st,dp = azdp(t)
    azn,dpn,st,dp = azdp(n)
    azp,dpp,st,dp = azdp(p)

    az1,dp1,st1,dip1 = azdp(pole1)
    az2,dp2,st2,dip2 = azdp(pole2)

    if -1.*d[jp]>d[jt]:
       djpt = d[jp]
    else:
       djpt = d[jt]
    clvd = d[jn]/djpt

    m0 = 0.5*(np.abs(d[jp])+np.abs(d[jt]))

    x = np.array([0.,-1*np.cos(st1/rad),np.sin(st1/rad)])
    vfin = np.dot(pole2,x)
    if vfin>1.:
       vfin = 1.
    if vfin<-1.:
       vfin = -1.
    rake1 = rad*np.arccos(vfin)
    if pole2[0]<0.: rake1 = -1.*rake1


    x = np.array([0.,-1*np.cos(st2/rad),np.sin(st2/rad)])
    vfin = np.dot(pole1,x)
    if vfin>1.:
       vfin = 1.
    if vfin<-1.:
       vfin = -1.
    rake2 = rad*np.arccos(vfin)
    if pole1[0]<0.: rake2 = -1.*rake2
    return 3*tr,clvd, m0,(azt,dpt),(azn,dpn),(azp, dpp), (st1,dip1,rake1), (st2,dip2,rake2)


def Rpattern(fault,azimuth,incidence_angles):
    """
    Calculate predicted amplitudes of P, SV, and SH waves.
    IN: fault = [strike, dip, rake]
             = faulting mechanism, described by a list of strike, dip, and rake
             (note, strike is measured clockwise from N, dip is measured positive downwards
             (between 0 and 90) w.r.t. a horizontal that is 90 degrees clockwise from strike,
             and rake is measured positive upwards (counterclockwise)
        azimuth: azimuth with which ray path leaves source (clockwise from N)
        incidence_angles = [i, j]
              i = angle between P ray path & vertical in the source model layer
              j = angle between S ray path & vertical in the source model layer
    OUT: Amplitudes for P, SV, and SH waves
    P as measured on L (~Z) component, SV measured on Q (~R) component, and SH measured on T component.
    All input is in degrees. 
    (c) 2020 Suzan van der Lee
    """

    strike,dip,rake = fault
    a = azimuth; rela = strike - azimuth
    sinlam = np.sin(np.radians(rake))
    coslam = np.cos(np.radians(rake))
    sind = np.sin(np.radians(dip))
    cosd = np.cos(np.radians(dip))
    cos2d = np.cos(np.radians(2*dip))
    sinrela = np.sin(np.radians(rela))
    cosrela = np.cos(np.radians(rela))
    sin2rela = np.sin(np.radians(2*rela))
    cos2rela = np.cos(np.radians(2*rela))

    sR = sinlam*sind*cosd
    qR = sinlam*cos2d*sinrela + coslam*cosd*cosrela
    pR = coslam*sind*sin2rela - sinlam*sind*cosd*cos2rela
    pL = sinlam*sind*cosd*sin2rela + coslam*sind*cos2rela
    qL = -coslam*cosd*sinrela + sinlam*cos2d*cosrela

    iP = np.radians(incidence_angles[0])
    jS = np.radians(incidence_angles[1])

    AP = sR*(3*np.cos(iP)**2 - 1) - qR*np.sin(2*iP) - pR*np.sin(iP)**2
    ASV = 1.5*sR*np.sin(2*jS) + qR*np.cos(2*jS) + 0.5*pR*np.sin(2*jS)
    ASH = qL*np.cos(jS) + pL*np.sin(jS)

    return AP,ASV,ASH


def getamp(azimuth, strike, dip, rake, rayp, mod, depth, Pvelz, Svelz):
    """
    INPUT:  az = azimuth in degrees from the event to the lander
            st, dp, rk = strike dip rake from 3 separate lists
            Pp, Sp = ray paramters calculated from the model in obspy
            Pvelz, Svelz = velocity @ depth from model
            radius = predefined radius of planet
    OUTPUT: df = dataframe containing synthetic amplitudes
            ip, ij = exit angles (???)
    # (c) 2021 Madelyn Sita
    """
    # define empty lists
    strike_ls = []; dip_ls = []; rake_ls = []
    P_ls = []; SH_ls = []; SV_ls = []

    # loop over fault plane combinations
    for st in strike:
        for dp in dip:
            for rk in rake:
                strike_ls.append(st); dip_ls.append(dp); rake_ls.append(rk)

                # define single fault for calculations
                fault = [st, dp, rk]

                # calculating exit angles using the models velocity @ depth & ray parameters
                # radius should be the radius @ depth
                iP = np.degrees(np.arcsin(Pvelz*rayp[0]/(radius-depth)))
                jS = np.degrees(np.arcsin(Svelz*rayp[1]/(radius-depth)))

                # calculating amplitudes
                P,iSV,iSH = Rpattern(fault, azimuth, [iP, jS])
                scalefactor = (Pvelz/Svelz)**3
                SV,SH = iSV*scalefactor, iSH*scalefactor
                P_ls.append(P); SH_ls.append(SH); SV_ls.append(SV)

    # creating dataframe
    data = {
            'Model': mod,
            'Depth': depth,
            'Strike': strike_ls,
            'Dip': dip_ls,
            'Rake': rake_ls,
            'P': P_ls,
            'SV': SV_ls,
            'SH': SH_ls
            }

    df = pd.DataFrame(data, columns = ['Model','Depth','Strike', 'Dip', 'Rake', 'P', 'SV', 'SH'])
    return df, iP, jS


def eventbuild(taup_model, event, dist, depth):
    # (c) 2021 Madelyn Sita
    # determine travel times using obspy
    mtimes = taup_model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist, phase_list=['P','S'])

    # ray parameters & incidence angles at the station
    Pp = mtimes[0].ray_param ; Pa = mtimes[0].incident_angle

    try:
        Sp = mtimes[1].ray_param ; Sa = mtimes[1].incident_angle
    except:
        Sp = 0 ; Sa = 0
        print('Within S-wave shadow zone')

    return Pp, Sp, Pa, Sa


def autofault(df, depth, mod, obsP, obsSV, obsSH, errP, errSV, errSH):
    # (c) 2021 Madelyn Sita
    vobserved = np.array([5*obsP, obsSV, obsSH])
    vobslength = np.linalg.norm(vobserved)
    norm = vobserved/vobslength

    eobs = np.array([5*errP,errSV,errSH])
    eca_ls = np.sqrt((eobs[0]**2*(1-norm[0]**2) + \
                     eobs[1]**2*(1-norm[1]**2) + \
                     eobs[2]**2*(1-norm[2]**2))/3)/vobslength
    eca = np.arctan(eca_ls)
    print('tolerance (cut-off value): ', eca, 'radians')

    xd = 5*df['P']
    yd = df['SV']
    zd = df['SH']
    ncalc = len(zd)

    len_xd = len(xd); len_yd = len(yd)
    if ncalc != len_xd:
        print('ERROR xd LENGTH')
    if ncalc != len_yd:
        print('ERROR yd LENGHT')

    vcall = np.array([xd,yd,zd])
    vca = vcall.T

    # ------misfit-------
    # empty array
    mf3d = np.zeros(ncalc)

    # list of index values for fault planes below misfit val
    select = []

    for i in np.arange(ncalc):
        # angle in 3 dimensions: (in radians)
        mf3d[i] = np.arccos(np.dot(norm, vca[i])/np.linalg.norm(vca[i]))
        if mf3d[i] < eca:
            select.append(i)

    # pulling fault plane data associated w/ index value
    st_ls = []; dp_ls =[]; rk_ls=[]; mf_ls = []; dep_ls = []; mod_ls = []
    azt_ls=[]; dpt_ls=[]; azn_ls = []; dpn_ls = []; azp_ls = []; dpp_ls = []
    for i in select:
        st = df.at[i,'Strike']
        dp = df.at[i,'Dip']
        rk = df.at[i,'Rake']
        st_ls.append(st); dp_ls.append(dp); rk_ls.append(rk)

        mt = getmt([st,dp,rk])
        dum1,dum2,dum3,(azt,dpt),(azn,dpn),(azp, dpp), (st1,dip1,rake1), (st2,dip2,rake2) = getplanes(mt)
        azt_ls.append(int(azt)); dpt_ls.append(int(dpt))
        azn_ls.append(int(azn)); dpn_ls.append(int(dpn))
        azp_ls.append(int(azp)); dpp_ls.append(int(dpp))

        mf_ls.append(mf3d[i])
        dep_ls.append(depth); mod_ls.append(mod)

    faults = {'AZT': azt_ls,
        'DPT': dpt_ls,
        'AZN': azn_ls,
        'DPN': dpn_ls,
        'AZP': azp_ls,
        'DPP': dpp_ls,
        'Strike': st_ls,
        'Dip': dp_ls,
        'Rake': rk_ls,
        'Misfit': mf_ls,
        'Depth': dep_ls,
        'Mod': mod_ls}
    posfaults = pd.DataFrame.from_dict(faults)

    faults_sorted = posfaults.sort_values(by=['Misfit'])
    faults_sorted = faults_sorted.drop_duplicates(subset = ['Strike', 'Dip', 'Rake'])
    faults_sorted = faults_sorted.drop_duplicates(subset = ['AZT','DPT','AZN','DPN','AZP','DPP'])
    return faults_sorted


def predictamp(data,az,rayp,print_state=False):
    # (c) 2021 Madelyn Sita
    for index, rows in data.iterrows():
        ampsdf, iP, jS = getamp(az, [rows.Strike], [rows.Dip], [rows.Rake], rayp)

        if print_state ==True:
            print(ampsdf)
        else:
            pass


def eventoutput(events_results, events_amps, args):
    n = 0
    for mod in args.mod_list:
        taup_model = TauPyModel(model=str(EXP_DIR / 'models' / mod))
        for depth in [15, 35, 55]:
            if mod == 'NewGudkova':
                if depth <= 203 and depth > 50:
                    Pvelz = 7.45400; Svelz = 4.21600
                elif depth <= 50 and depth > 42:
                    Pvelz = 7.12500; Svelz = 4.00300    #rounded
                elif depth <= 42 and depth > 21:
                    Pvelz = 7.13900; Svelz = 4.01900
                elif depth <=21 and depth > 16:
                    Pvelz = 7.14300; Svelz = 4.02300
                elif depth <= 16 and depth > 10:
                    Pvelz = 7.15000; Svelz = 4.03000    #rounded

            elif mod == 'TAYAK':
                if depth <= 77 and depth > 10:
                    Pvelz = 5.84666; Svelz = 3.28116
                elif depth <= 10 and depth >1:
                    Pvelz = 4.95225; Svelz = 2.78097

            elif mod == 'Combined':
                if depth <= 203 and depth > 50:
                    Pvelz = 7.45400; Svelz = 4.21600
                elif depth <= 50 and depth > 22:
                    Pvelz = 7.12700; Svelz = 4.00200
                elif depth <= 22 and depth > 8.6:
                    Pvelz = 5.14700; Svelz = 2.73900
                elif depth <= 8.6 and depth > 0:
                    Pvelz = 3.50400; Svelz = 1.77100


            for event, event_vals in events_results.items(): 
                if event in events_amps: 
                    print("*"*30)
                    print(f"Model: {mod},  Depth: {depth}, Pvelz: {Pvelz}, Svelz: {Svelz}")
                    print(f"Fault guessing for {event}")
                    Pp, Sp, Pa, Sa = eventbuild(taup_model=taup_model, event=event, dist=event_vals['distance'], depth=depth)

                    df, Pe, Se = getamp(azimuth=event_vals['az'], strike=strike_rang, dip=dip_rang, rake=rake_rang, rayp=[Pp, Sp], 
                                        mod=mod, depth=depth, Pvelz=Pvelz, Svelz=Svelz)

                    print(f"Exit Angles: {Pe}, {Se}")
                    df = autofault(df, depth, mod, **events_amps[event])
                    if n == 0:
                        os.makedirs(EXP_DIR / 'events' / f'{event}', exist_ok=True)
                        df.to_csv(EXP_DIR / 'events' / f'{event}' /  f'{event}.csv', header=True)
                    else:
                        df.to_csv(EXP_DIR / 'events' / f'{event}' / f'{event}.csv', mode='a', header=False)
                else: 
                    print(f"Skipping {event}: Not found in events amps dict")

            n+=1



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Model Parameters -
    parser.add_argument("--model", required=True, type=str, help="Model name used for distance calculation")
    parser.add_argument("--depth", required=True, type=float, help="Source Depth (km) used for distance calculation")
    parser.add_argument("--radius", default=3389.5, type=float, help="Radius of Mars")
    parser.add_argument("--exp_dir", required=True, type=str)
    parser.add_argument("--file", required=True, default=None, 
                        help="""File in json format containing the amp information (obsP, obsSV, obsSH, errP, errSV, errSH) for each event. 
                        Example file: 
                        {
                            "S0235b":
                                {	
                                    "obsP": 3.62e-10
                                    "obsSV": 3.47e-9, 
                                    "obsSH": -1.611e-9, 
                                    "errP": 3.73e-11, 
                                    "errSV": 7.03e-11, 
                                    "errSH": 7.42e-11 
                                },

                            "S0173a":
                                {	
                                    "obsP": -1.25e-9,
                                    "obsSV": 0.955e-9, 
                                    "obsSH": -0.371e-9, 
                                    "errP": 1.13e-10, 
                                    "errSV": 1.46e-10, 
                                    "errSH": 2.01e-10 
                                }
                        }
                        """)
    parser.add_argument("-m", '--mod_list',  nargs='+', default=['NewGudkova', 'Combined', 'TAYAK'], 
                        help="List of mars models to use for fault guessing")

    args = parser.parse_args()

    EXP_DIR = EXP_PATH / args.exp_dir 

    
    logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
    logger=logging.getLogger() 
    time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    logger.info("*"*50)
    logger.info(f"{time_stamp}: Running {Path(__file__).name}")
    logger.info(f"{args.__dict__}")


    eps = 1e-6
    rad = 180/np.pi
    halfpi = 0.5*np.pi
    twopi = 2.0*np.pi

    # ---------- MARS -----------------
    radius = args.radius 


    # ------ FAULT PLANE LISTS -------    
    strike_rang = list(range(0, 360, 2))
    dip_rang = list(range(0, 90, 2))
    rake_rang = list(range(-180, 180, 2)) 

    # Load event-related information 

    with open(EXP_DIR / 'events' / args.file) as f: 
        events_amps = json.load(f)

    events_results = utils.load_file(EXP_DIR / 'events' /  f'{args.model}_depth{args.depth}_events_results.pkl')

    eventoutput(events_results, events_amps, args)

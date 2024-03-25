import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget
import numpy as np
from tqdm import tqdm
import scipy.stats as st 
from scipy import signal
import scipy
from scipy.signal import detrend
import glob
from numpy.linalg import lstsq
import math
import os

from dtscalibration import read_silixa_files

import warnings

from dtscalibration import read_silixa_files

from dtscalibration.datastore_utils import suggest_cable_shift_double_ended, shift_double_ended
# If the icetemperature library is not in your PYTHONPATH, you will not be able to load those functions
# Check and update here if necessary
import sys
cs_dir = '/home/jmanos/notebooks/iceotherm/'
if cs_dir not in sys.path:
    sys.path.append(cs_dir)

# Import the ice temperature model and relevant constants
from iceotherm.lib.numerical_model import ice_temperature
from iceotherm.lib.ice_properties import conductivity, heat_capacity
from iceotherm.lib.constants import constants
const = constants()

# Interpolator for density profile
from scipy.interpolate import interp1d

def data_processing(filepath):
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')

    suggested_shift = suggest_cable_shift_double_ended(
        ds,
        np.arange(-10, 10),
        plot_result=True,
        figsize=(12,8))

    ds_restored = shift_double_ended(ds, suggested_shift[0])
    ds = ds_restored

    sections = {
    "referenceTemperature": [slice(-23, -3)], 
    }
    ds.sections = sections
    
#     matching_sections = [
#     (slice(74.8, 104.62), slice(518.95, 548.77), True)
#     ]
    
    st_var, resid = ds.variance_stokes(st_label='st')
    ast_var, _ = ds.variance_stokes(st_label='ast')
    rst_var, _ = ds.variance_stokes(st_label='rst')
    rast_var, _ = ds.variance_stokes(st_label='rast')

    ds.calibration_double_ended(
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        #matching_sections=matching_sections,
        store_tmpw='tmpw',
        method='wls',
        solver='sparse')
    
    return ds

def data_processing_matching_sections(filepath, matching_sections_dict):
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')

    suggested_shift = suggest_cable_shift_double_ended(
        ds,
        np.arange(-10, 10),
        plot_result=True,
        figsize=(12,8))

    ds_restored = shift_double_ended(ds, suggested_shift[0])
    ds = ds_restored

    sections = {
    "referenceTemperature": [slice(-23, -3)], 
    }
    ds.sections = sections
    matching_sections = matching_sections_dict[filepath[13:]][0]
#     matching_sections = [
#     (slice(74.8, 104.62), slice(518.95, 548.77), True)
#     ]
    
    st_var, resid = ds.variance_stokes(st_label='st')
    ast_var, _ = ds.variance_stokes(st_label='ast')
    rst_var, _ = ds.variance_stokes(st_label='rst')
    rast_var, _ = ds.variance_stokes(st_label='rast')

    ds.calibration_double_ended(
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        trans_att=matching_sections_dict[filepath[13:]][1],
        matching_sections=[matching_sections],
        store_tmpw='tmpw',
        method='wls',
        solver='sparse'),
    
    resid.plot(figsize=(12, 4))
    return ds

### This function will read each borehole and process it how we want ###
def borehole_data_reader(data_dict,
                         borehole_name,
                         geometry
                        ):
    x = data_dict[borehole_name].sel(x=slice(geometry[borehole_name][0],
                                              geometry[borehole_name][2])).x - geometry[borehole_name][0]
    y = data_dict[borehole_name].sel(x=slice(geometry[borehole_name][0],
                                              geometry[borehole_name][2])).tmpw
        
    if y.shape[1] == 1:
        mean = y
        ci95 = None
        ci05 = None
    else:
        mean = np.mean(y, axis=1)
        ci95, ci05 = st.t.interval(alpha=0.95,
                                   df=len(data_dict[borehole_name])-1,
                                   loc=mean,
                                   scale=st.sem(y, axis=1))
        
            #pick the surface temperature by the linear trend
    slope,b = np.polyfit(y.sel(x=slice(geometry[borehole_name][1],
                                       geometry[borehole_name][2])).x, 
                         y.sel(x=slice(geometry[borehole_name][1],
                                       geometry[borehole_name][2])), 
                         1)

    Ts_all  = np.mean(slope)*y.sel(x=slice(geometry[borehole_name][0],geometry[borehole_name][2])).x + np.mean(b)
    
    qgeo = np.mean(slope)*2 #TODO: Why do we need to multiply by 2?
    
    H=x.values[-1] - x.values[0]#ice thickness
    Ts = Ts_all[0].values
    anomaly = np.mean(y - Ts_all, axis=1)
    return x, y, mean, ci95, ci05, qgeo, H, Ts, anomaly


### Updated Greens Function generator ###
def greens_generator_v4(borehole_proc,
                        timeseries_toModel = None,
                        tmin=1923,
                        tmax=2023,
                        t_steps=100,
                        num_steps=5000,
                        nz=11,
                        tol=1e-2,
                        include_mechanics=False,
                        step_function_input=False,
                        start_at_zero=False):
    adot = -0.01 #accumulation rate
    year = np.linspace(tmin,tmax, num_steps)
    H = borehole_proc[6]
    qgeo = borehole_proc[5]
    step_func_len = num_steps / t_steps #please make this a whole number
    z = np.linspace(0,H, nz)

    if step_function_input == True:
        temp_input = np.zeros((t_steps,num_steps))
        if step_func_len != int(step_func_len):
            raise Exception('num_steps divided by t_steps needs to be an integer!')
    
        for i in range(t_steps):
            start = i*step_func_len
            end = (i*step_func_len) + step_func_len
            temp_input[i,start:end] = 1
            
    if step_function_input == False:
        temp_input = np.eye(num_steps)

    
    if start_at_zero == False:
        Ts = borehole_proc[7]

        # Instantiate the model class
        m = ice_temperature(
            Ts=Ts,
            adot=adot,
            H=H,
            qgeo=qgeo,
            p=1000.,
            dS=0,
            nz=nz,
            tol=tol)
    if start_at_zero == True:
        Ts = 0.
        # Instantiate the model class
        m = ice_temperature(
            Ts=Ts,
            adot = 0.,
            qgeo=0.,
            H=H,
            p=1000.,
            nz=nz,
            tol=tol)

    m.ts = year*const.spy
    m.adot_s = np.asarray([adot]*len(year))/const.spy

    if include_mechanics == True:
        # Set velocity terms
        m.Udef=0.
        m.Uslide=0.2/const.spy

        # Longitudinal advection forcing

        m.dTs = 0
        m.dH = 0.3
        m.da = 0.
        m.flags.append('long_advection')

        # Thermal conductivity should be temperature and density dependent; set here
        m.flags.append('temp-dependent')
    m.initial_conditions()


    # Initialize the model to steady state

    m.source_terms()
    m.stencil()
        
    m.numerical_to_steady_state()

    if np.isnan(m.T).any():
        raise Exception('NaN introduced during steady state. Check nz and timesteps.')

    steady_state = m.T
    arr = np.zeros((nz, len(temp_input)))
    m.flags.append('save_all')
    m.flags.pop(0) #pop verbose so it doesnt print every iteration
    for k, surface_temp in tqdm(enumerate(temp_input)):

        # Run the model

        m.Ts_s = surface_temp+Ts
        m.numerical_transient()
        arr[:,k] = m.T
        if np.isnan(m.T).any():
            raise Exception('NaN introduced during forward model. Check nz and timesteps.')
    arr = arr[::-1]        #Flip array because they come upside down
    return steady_state, arr, z, year


def foreword_modeler(borehole_proc,
                     timeseries_toModel = None,
                     new_qgeo = None,
                     tmin=1923,
                     tmax=2023,
                     t_steps=100,
                     num_steps=5000,
                     nz=11,
                     tol=1e-2,
                     include_mechanics=False,
                     start_at_zero=False,
                     model_timeseries = False,
                     qgeo_update = False
                    ):
    adot = -0.01 #accumulation rate
    year = np.linspace(tmin,tmax, num_steps)
    H = borehole_proc[6]
    qgeo = borehole_proc[5]
    step_func_len = num_steps / t_steps #please make this a whole number
    z = np.linspace(0,H, nz)
    
    if qgeo_update == True:
        if new_qgeo == None:
            raise Exception('Please give a value for qgeo_update!')
        else:
            qgeo = new_qgeo
    
    if start_at_zero == False:
        Ts = borehole_proc[7]

        # Instantiate the model class
        m = ice_temperature(
            Ts=Ts,
            adot=adot,
            H=H,
            qgeo=qgeo,
            p=1000.,
            dS=0.,
            nz=nz,
            tol=tol)
    if start_at_zero == True:
        Ts = 0.
        # Instantiate the model class
        m = ice_temperature(
            Ts=Ts,
            adot = 0.,
            qgeo=0.,
            H=H,
            p=1000.,
            nz=nz,
            tol=tol)
    # Set the time step to 5 years and subsample the paleoclimate data to match
    m.ts = year*const.spy
    m.adot_s = np.asarray([adot]*len(year))/const.spy

    if include_mechanics == True:
        # Set velocity terms
        m.Udef=0.
        m.Uslide=0.2/const.spy

        # Longitudinal advection forcing

        m.dTs = 0
        m.dH = 0.3
        m.da = 0.
        m.flags.append('long_advection')

        # Thermal conductivity should be temperature and density dependent; set here
        m.flags.append('temp-dependent')
    m.initial_conditions()


    # Initialize the model to steady state

    m.source_terms()
    m.stencil()
    m.flags.pop(0) #pop verbose so it doesnt print every iteration
    m.numerical_to_steady_state()

    if np.isnan(m.T).any():
        raise Exception('NaN introduced during steady state. Check nz and timesteps.')

    steady_state = m.T
#     m.flags.pop(0) #pop verbose so it doesnt print every iteration
    if model_timeseries == True:
        if any(timeseries_toModel):
            
            #adjust the time series to the mean temp at the surface
            adjusted_timeseries = timeseries_toModel + Ts
            # Run the model
            m.Ts_s = adjusted_timeseries
            m.flags.append('save_all')
            m.numerical_transient()

            modeled_profile = m.T
        else:
            raise Exception('Please include a time series to model!')
    elif model_timeseries == False:
        modeled_profile = None

    return steady_state, modeled_profile, z

def brown_noise(length, alpha=1):
    # Generate white noise
    noise = np.random.randn(length)
    
    # Integrate the noise
    integrated = np.cumsum(noise)
    
    # Scale by alpha
    scaled = integrated * (1/np.arange(1, length+1)**(alpha/2))
    
    # Normalize to have unit standard deviation
    return scaled / np.std(scaled)

def inverser_SVD(greensFunction,
                 boreholeTemp,
                 p=4, 
                 nz=100,
                 detrend_arr = False):
    

    if detrend_arr == True:
        u, s, v_T = np.linalg.svd(detrend(greensFunction, axis=1), full_matrices=True)
    if detrend_arr == False:
        u, s, v_T = np.linalg.svd(greensFunction, full_matrices=True)
        
    u_p = u[:,:p]
    s_p = np.diag(s[:p])
    vh_p = v_T.T[:,:p]

    #Now finding the generalized G matrix

    g_gen = u_p @ s_p @ vh_p.T
    # print('G = \n', g_gen)

    #And the inverse Gen G matrix

    u_p_transpose = u_p.T
    s_p_inv = np.linalg.inv(s_p)

    g_inv = vh_p @ s_p_inv @ u_p_transpose
                #model estimates
    
    if boreholeTemp.shape[0] == g_inv.shape[1]:
        if detrend_arr == False:
            m_est = g_inv @ boreholeTemp
        if detrend_arr == True:
            m_est = g_inv @ detrend(boreholeTemp)
    elif boreholeTemp.shape[0] != g_inv.shape[1]:
        #Resample by 
        #mean_resamp = scipy.signal.resample(mean,greens.shape[0],window=10)
        x = np.linspace(0, len(boreholeTemp)*.256,len(boreholeTemp))
        boreholeTemp = boreholeTemp.reshape((len(boreholeTemp),))
        f = scipy.interpolate.interp1d(x, boreholeTemp)
        resamp_x = np.linspace(0, x[-1], nz)
        y_resamp = f(resamp_x)
        if detrend_arr == False:
            m_est = g_inv @ y_resamp
        if detrend_arr == True:
            m_est = g_inv @ detrend(y_resamp)

    return m_est, u_p, s_p, vh_p


def inverser_LS(greensFunction, data, nz = 10, detrend_arr = False):
    # greensFunction: Our Greens function
    # data: borehole temperature. Make sure this has same dimension as greensFunction
    if data.shape[0] == greensFunction.shape[1]:
        if detrend_arr == False:
            m_est = lstsq(greensFunction, data)[0]
        if detrend_arr == True:
            m_est = lstsq(detrend(greensFunction, axis=1), data)[0]
    if data.shape[0] != greensFunction.shape[1]:
        #Resample by 
        #mean_resamp = scipy.signal.resample(mean,greens.shape[0],window=10)
        x = np.linspace(0, len(data)*.256,len(data))
        data = data.reshape((len(data),))
        f = scipy.interpolate.interp1d(x, data)
        resamp_x = np.linspace(0, x[-1], nz)
        y_resamp = f(resamp_x)
        if detrend_arr == True:
            m_est = lstsq(detrend(greensFunction, axis=1), y_resamp)[0]
        else:
            m_est = lstsq(greensFunction, y_resamp)[0]
    return m_est

def temp_timeSeries_creator(freq =1, 
                            phase = 0,
                            amp = -30,
                            long_term = -45, #We dont really need this since, we use surface temp in foreword model.
                            trend_amp = 0.02, 
                            tmin = 1923, 
                            tmax = 2023, 
                            num_steps= 100):
    t = np.linspace(tmin, tmax,num_steps)
    trend = (t-tmin) * trend_amp
    temp = long_term + amp*np.sin(2*np.pi*t * freq) + trend
    
    return temp


def likelihood(real_borehole,modeled_borehole,sigma):

    mae = np.sum(np.absolute(real_borehole - modeled_borehole))
    return np.exp(-mae**2/sigma**2 / 2) / np.sqrt(2*np.pi) / sigma     
    
def greens_generator_efc(borehole_proc,
                         tmin = 1923, 
                         tmax = 2023, 
                         nz = 11, 
                         num_steps = 100):
    k_diff = 1.09e-6 * 3.154e+7 #* ((tmax - tmin) / (num_steps)) #seconds
    H = borehole_proc[6]
    z = np.linspace(0.254,H, nz)
    t = np.linspace(tmin, tmax,num_steps) - tmin
    A_jk = np.zeros(shape = (len(z), len(t)-1))
    A_jk_beltrami = np.zeros(shape = (len(z), len(t)+1))
    for k in range(len(t) - 1):
        for j in range(len(z)):
            #print(t[k])
            #A_jk[j,k] = (math.erfc(z[j] / (2*np.sqrt(k_diff * t[k])))) - (math.erfc(z[j] / (2*np.sqrt(k_diff * t[k+1]))))
            A_jk[j,k] = math.erfc(z[j] / (2*np.sqrt(k_diff * t[k]))) - math.erfc(z[j] / (2*np.sqrt(k_diff * t[k+1])))

            # This A_jk is evaluated following Beltrami 
            A_jk_beltrami[j,k+2] = math.erfc(z[j] / (2*np.sqrt(k_diff * t[k]))) - math.erfc(z[j] / (2*np.sqrt(k_diff * t[k+1])))
            
    return A_jk, A_jk_beltrami   
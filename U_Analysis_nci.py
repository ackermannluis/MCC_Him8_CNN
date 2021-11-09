#!/usr/bin/env python
# Copyright 2021
# author: Luis Ackermann <ackermann.luis@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from scipy.stats import ttest_ind
import netCDF4 as nc

import pickle
import os
from PIL import Image as PIL_Image
import sys
import shutil
import glob
import datetime
import time
import calendar
from numpy import genfromtxt
from scipy.optimize import curve_fit
from scipy.cluster.vq import kmeans,vq
from scipy.interpolate import interpn, interp1d
from math import e as e_constant
import math
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import (MultipleLocator, NullFormatter, ScalarFormatter)
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import warnings
warnings.filterwarnings("ignore")
plt.style.use('classic')

# font size
# font_size = 14
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Arial'], 'size': font_size})
# matplotlib.rc('font', weight='bold')
p_progress_writing = False

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




time_format = '%d-%m-%Y_%H:%M'
time_format_khan = '%Y%m%d.0%H'
time_format_mod = '%Y-%m-%d_%H:%M:%S'
time_format_twolines = '%H:%M\n%d-%m-%Y'
time_format_twolines_noYear_noMin_intMonth = '%H\n%d-%m'
time_format_twolines_noYear = '%H:%M\n%d-%b'
time_format_twolines_noYear_noMin = '%H\n%d-%b'
time_format_date = '%Y-%m-%d'
time_format_time = '%H:%M:%S'
time_format_parsivel = '%Y%m%d%H%M'
time_format_parsivel_seconds = '%Y%m%d%H%M%S'
time_str_formats = [
    time_format,
    time_format_mod,
    time_format_twolines,
    time_format_twolines_noYear,
    time_format_date,
    time_format_time,
    time_format_parsivel
]


default_cm = cm.jet
cm_vir = cm.viridis
listed_cm_colors_list = ['silver', 'red', 'green', 'yellow', 'blue', 'black']
listed_cm = ListedColormap(listed_cm_colors_list, 'indexed')

colorbar_tick_labels_list_cloud_phase = ['Clear', 'Water', 'SLW', 'Mixed', 'Ice', 'Unknown']
listed_cm_colors_list_cloud_phase = ['white', 'red', 'green', 'yellow', 'blue', 'purple']
listed_cm_cloud_phase = ListedColormap(listed_cm_colors_list_cloud_phase, 'indexed')


avogadros_ = 6.022140857E+23 # molecules/mol
gas_const = 83144.598 # cm3  mbar  k-1   mol-1
gas_const_2 = 8.3144621 # J mol-1 K-1
gas_const_water = 461 # J kg-1 K-1
gas_const_dry = 287 # J kg-1 K-1

boltzmann_ = gas_const / avogadros_ #  cm3  mbar / k   molecules
gravity_ = 9.80665  # m/s
poisson_ = 2/7 # for dry air (k)
latent_heat_v = 2.501E+6 # J/kg
latent_heat_f = 3.337E+5 # J/kg
latent_heat_s = 2.834E+6 # J/kg

heat_capacity__Cp = 1005.7 # J kg-1 K-1    dry air
heat_capacity__Cv = 719 # J kg-1 K-1      water vapor

Rs_da = 287.05          # Specific gas const for dry air, J kg^{-1} K^{-1}
Rs_v = 461.51           # Specific gas const for water vapour, J kg^{-1} K^{-1}
Cp_da = 1004.6          # Specific heat at constant pressure for dry air
Cv_da = 719.            # Specific heat at constant volume for dry air
Cp_v = 1870.            # Specific heat at constant pressure for water vapour
Cv_v = 1410.            # Specific heat at constant volume for water vapour
Cp_lw = 4218	          # Specific heat at constant pressure for liquid water
Epsilon = 0.622         # Epsilon=Rs_da/Rs_v; The ratio of the gas constants
degCtoK = 273.15        # Temperature offset between K and C (deg C)
rho_w = 1000.           # Liquid Water density kg m^{-3}
grav = 9.80665          # Gravity, m s^{-2}
Lv = 2.5e6              # Latent Heat of vaporisation
boltzmann = 5.67e-8     # Stefan-Boltzmann constant
mv = 18.0153e-3         # Mean molar mass of water vapor(kg/mol)
m_a = 28.9644e-3        # Mean molar mass of air(kg/mol)
Rstar_a = 8.31432       # Universal gas constant for air (N m /(mol K))


path_output = '/g/data/k10/la6753/'


# Misc
class Object_create(object):
    pass
def list_files_recursive(path_, filter_str=None):
    # create list of raw spectra files
    file_list = []
    # r=root, d=directories, f = files
    if filter_str is None:
        for r, d, f in os.walk(path_):
            for file in f:
                file_list.append(os.path.join(r, file))
    else:
        for r, d, f in os.walk(path_):
            for file in f:
                if filter_str in file:
                    file_list.append(os.path.join(r, file))
    return file_list
def list_files(path_, filter_str='*'):
    file_list = sorted(glob.glob(str(path_ + filter_str)))
    return file_list
def coincidence(arr_1,arr_2):
    # only coincidences
    check_ = arr_1 * arr_2
    check_[check_ == check_] = 1

    arr_1_checked = arr_1 * check_
    arr_2_checked = arr_2 * check_

    return arr_1_checked[~np.isnan(arr_1_checked)], arr_2_checked[~np.isnan(arr_2_checked)]
def array_2d_fill_gaps_by_interpolation_linear(array_):

    rows_ = array_.shape[0]
    cols_ = array_.shape[1]
    output_array_X = np.zeros((rows_, cols_), dtype=float)
    output_array_Y = np.zeros((rows_, cols_), dtype=float)

    row_sum = np.sum(array_, axis=1)
    col_index = np.arange(array_.shape[1])

    col_sum = np.sum(array_, axis=0)
    row_index = np.arange(array_.shape[0])

    for r_ in range(array_.shape[0]):
        if row_sum[r_] != row_sum[r_]:
            # get X direction interpolation
            coin_out = coincidence(col_index, array_[r_, :])
            output_array_X[r_, :][np.isnan(array_[r_, :])] = np.interp(
                col_index[np.isnan(array_[r_, :])], coin_out[0], coin_out[1])

    for c_ in range(array_.shape[1]):
        if col_sum[c_] != col_sum[c_]:
            # get Y direction interpolation
            coin_out = coincidence(row_index, array_[:, c_])
            output_array_Y[:, c_][np.isnan(array_[:, c_])] = np.interp(
                row_index[np.isnan(array_[:, c_])], coin_out[0], coin_out[1])

    output_array = np.array(array_)
    output_array[np.isnan(array_)] = 0

    return output_array + ((output_array_X + output_array_Y)/2)
def array_2d_fill_gaps_by_interpolation_cubic(array_):

    rows_ = array_.shape[0]
    cols_ = array_.shape[1]
    output_array_X = np.zeros((rows_, cols_), dtype=float)
    output_array_Y = np.zeros((rows_, cols_), dtype=float)

    row_sum = np.sum(array_, axis=1)
    col_index = np.arange(array_.shape[1])

    col_sum = np.sum(array_, axis=0)
    row_index = np.arange(array_.shape[0])

    for r_ in range(array_.shape[0]):
        if row_sum[r_] != row_sum[r_]:
            # get X direction interpolation
            coin_out = coincidence(col_index, array_[r_, :])
            interp_function = interp1d(coin_out[0], coin_out[1], kind='cubic')
            output_array_X[r_, :][np.isnan(array_[r_, :])] = interp_function(col_index[np.isnan(array_[r_, :])])

    for c_ in range(array_.shape[1]):
        if col_sum[c_] != col_sum[c_]:
            # get Y direction interpolation
            coin_out = coincidence(row_index, array_[:, c_])
            interp_function = interp1d(coin_out[0], coin_out[1], kind='cubic')
            output_array_Y[:, c_][np.isnan(array_[:, c_])] = interp_function(row_index[np.isnan(array_[:, c_])])

    output_array = np.array(array_)
    output_array[np.isnan(array_)] = 0

    return output_array + ((output_array_X + output_array_Y)/2)
def combine_2_time_series(time_1_reference, data_1, time_2, data_2,
                          forced_time_step=None, forced_start_time=None, forced_stop_time=None,
                          cumulative_var_1=False, cumulative_var_2=False):
    """
    takes two data sets with respective time series, and outputs the coincident stamps from both data sets
    It does this by using mean_discrete() for both sets with the same start stamp and averaging time, the averaging time
    is the forced_time_step
    :param time_1_reference: 1D array, same units as time_2, this series will define the returned time step reference
    :param data_1: can be 1D or 2D array, first dimention most be same as time_1
    :param time_2: 1D array, same units as time_1
    :param data_2: can be 1D or 2D array, first dimention most be same as time_2
    :param window_: optional, if 0 (default) the values at time_1 and time_2 most match exactly, else, the match can
                    be +- window_
    :param forced_time_step: if not none, the median of the differential of the time_1_reference will be used
    :param forced_start_time: if not none, the returned series will start at this time stamp
    :param forced_stop_time: if not none, the returned series will stop at this time stamp
    :param cumulative_var_1: True is you want the variable to be accumulated instead of means, only of 1D data
    :param cumulative_var_2: True is you want the variable to be accumulated instead of means, only of 1D data
    :return: Index_averaged_1: 1D array, smallest coincident time, without time stamp gaps
    :return: Values_mean_1: same shape as data_1 both according to Index_averaged_1 times
    :return: Values_mean_2: same shape as data_2 both according to Index_averaged_1 times
    """

    # define forced_time_step
    if forced_time_step is None:
        forced_time_step = np.median(np.diff(time_1_reference))

    # find time period
    if forced_start_time is None:
        first_time_stamp = max(np.nanmin(time_1_reference), np.nanmin(time_2))
    else:
        first_time_stamp = forced_start_time
    if forced_stop_time is None:
        last_time_stamp = min(np.nanmax(time_1_reference), np.nanmax(time_2))
    else:
        last_time_stamp = forced_stop_time

    # do the averaging
    print('starting averaging of data 1')
    if cumulative_var_1:
        Index_averaged_1, Values_mean_1 = mean_discrete(time_1_reference, data_1, forced_time_step,
                                                        first_time_stamp, last_index=last_time_stamp,
                                                        cumulative_parameter_indx=0)
    else:
        Index_averaged_1, Values_mean_1 = mean_discrete(time_1_reference, data_1, forced_time_step,
                                                        first_time_stamp, last_index=last_time_stamp)
    print('starting averaging of data 2')
    if cumulative_var_2:
        Index_averaged_2, Values_mean_2 = mean_discrete(time_2, data_2, forced_time_step,
                                                        first_time_stamp, last_index=last_time_stamp,
                                                        cumulative_parameter_indx=0)
    else:
        Index_averaged_2, Values_mean_2 = mean_discrete(time_2, data_2, forced_time_step,
                                                        first_time_stamp, last_index=last_time_stamp)


    # check that averaged indexes are the same
    if np.nansum(np.abs(Index_averaged_1 - Index_averaged_2)) != 0:
        print('error during averaging of series, times do no match ????')
        return None, None, None

    # return the combined, trimmed data
    return Index_averaged_1, Values_mean_1, Values_mean_2
def split_str_chunks(s, n):
    """Produce `n`-character chunks from `s`."""
    out_list = []
    for start in range(0, len(s), n):
        out_list.append(s[start:start+n])
    return out_list
def coincidence_multi(array_list):
    # only coincidences
    parameters_list = array_list

    check_ = parameters_list[0]
    for param_ in parameters_list[1:]:
        check_ = check_ * param_
    check_[check_ == check_] = 1
    new_arr_list = []
    for param_ in parameters_list:
        new_arr_list.append(param_ * check_)
        check_ = check_ * param_
    # delete empty rows_
    list_list = []
    for param_ in parameters_list:
        t_list = []
        for i in range(check_.shape[0]):
            if check_[i] == check_[i]:
                t_list.append(param_[i])
        list_list.append(t_list)
    # concatenate
    ar_list = []
    for ii in range(len(parameters_list)):
        ar_list.append(np.array(list_list[ii]))
    return ar_list
def coincidence_zero(arr_1,arr_2):
    # only coincidences
    check_ = arr_1 * arr_2
    # delete empty rows_
    list_1 = []
    list_2 = []
    for i in range(check_.shape[0]):
        if check_[i] != 0:
            list_1.append(arr_1[i])
            list_2.append(arr_2[i])
    return np.array(list_1),np.array(list_2)
def discriminate(X_, Y_, Z_, value_disc_list, discrmnt_invert_bin = False):
    if discrmnt_invert_bin:
        Z_mask = np.ones(Z_.shape[0])
        Z_mask[Z_ > value_disc_list[0]] = np.nan
        Z_mask[Z_ >= value_disc_list[1]] = 1

        Y_new = Y_ * Z_mask
        X_new = X_ * Z_mask

    else:
        Z_mask = np.ones(Z_.shape[0])
        Z_mask[Z_ < value_disc_list[0]] = np.nan
        Z_mask[Z_ > value_disc_list[1]] = np.nan

        Y_new = Y_ * Z_mask
        X_new = X_ * Z_mask

    return X_new, Y_new
def add_ratio_to_values(header_, values_, nominator_index, denominator_index, ratio_name, normalization_value=1.):
    nominator_data = values_[:,nominator_index]
    denominator_data = values_[:,denominator_index]

    ratio_ = normalization_value * nominator_data / denominator_data

    values_new = np.column_stack((values_,ratio_))
    header_new = np.append(header_,ratio_name)

    return header_new, values_new
def bin_data(x_val_org,y_val_org, start_bin_edge=0, bin_size=1, min_bin_population=1):
    # get coincidences only
    x_val,y_val = coincidence(x_val_org,y_val_org)
    # combine x and y in matrix
    M_ = np.column_stack((x_val,y_val))
    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(x_val.shape[0]-1):
        if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
            if x_val[x+1] < x_val[x]:
                always_ascending = 0
    if always_ascending == 0:
        M_sorted = M_[M_[:,0].argsort()] # sort by first column
        M_ = M_sorted
    # convert data to list of bins
    y_binned = []
    x_binned = []
    last_row = 0
    last_row_temp = last_row
    while start_bin_edge <= np.nanmax(x_val):
        y_val_list = []
        for row_ in range(last_row, M_.shape[0]):
            if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                if M_[row_, 1] == M_[row_, 1]:
                    y_val_list.append(M_[row_, 1])
                    last_row_temp = row_
            if M_[row_, 0] >= start_bin_edge + bin_size:
                last_row_temp = row_
                break
        x_binned.append(start_bin_edge)
        if len(y_val_list) >= min_bin_population:
            y_binned.append(y_val_list)
        else:
            y_binned.append([])
        start_bin_edge += bin_size
        last_row = last_row_temp
    # add series
    if bin_size >= 1:
        x_binned_int = np.array(x_binned, dtype=int)
    else:
        x_binned_int = x_binned
    return x_binned_int, y_binned
def shiftedColorMap(cmap, midpoint=0.5, name='shiftedcmap'):

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(0, 1, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
def student_t_test(arr_1, arr_2):
    return ttest_ind(arr_1, arr_2, nan_policy='omit')
def k_means_clusters(array_, cluster_number, forced_centers=None):
    if forced_centers is None:
        centers_, x = kmeans(array_,cluster_number)
        data_id, x = vq(array_, centers_)
        return centers_, data_id
    else:
        data_id, x = vq(array_, forced_centers)
        return forced_centers, data_id
def grid_(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = matplotlib.mlab.griddata(x, y, z, xi, yi)
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z
def find_max_index_2d_array(array_):
    return np.unravel_index(np.argmax(array_, axis=None), array_.shape)
def find_min_index_2d_array(array_):
    return np.unravel_index(np.argmin(array_, axis=None), array_.shape)
def find_max_index_1d_array(array_):
    return np.argmax(array_, axis=None)
def find_min_index_1d_array(array_):
    return np.argmin(array_, axis=None)
def time_series_interpolate_discrete(Index_, Values_, index_step, first_index,
                                     position_=0., last_index=None):
    """
    this will average values from Values_ that are between Index_[n:n+avr_size)
    :param Index_: n by 1 numpy array to look for position,
    :param Values_: n by m numpy array, values to be averaged
    :param index_step: in same units as Index_
    :param first_index: is the first discrete index on new arrays.
    :param position_: will determine where is the stamp located; 0 = beginning, .5 = mid, 1 = top (optional, default = 0)
    :param last_index: in case you want to force the returned series to some fixed period/length
    :return: Index_averaged, Values_averaged
    """

    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(Index_.shape[0]-1):
        if Index_[x]==Index_[x] and Index_[x+1]==Index_[x+1]:
            if Index_[x+1] < Index_[x]:
                always_ascending = 0
    if always_ascending == 0:
        MM_ = np.column_stack((Index_,Values_))
        MM_sorted = MM_[MM_[:,0].argsort()] # sort by first column
        Index_ = MM_sorted[:,0]
        Values_ = MM_sorted[:,1:]

    # error checking!
    if Index_.shape[0] != Values_.shape[0]:
        print('error during shape check! Index_.shape[0] != Values_.shape[0]')
        return None, None
    if Index_[-1] < first_index:
        print('error during shape check! Index_[-1] < first_index')
        return None, None


    # initialize output matrices
    if last_index is None:
        final_index = np.nanmax(Index_)
    else:
        final_index = last_index

    total_averaged_rows = int((final_index-first_index) / index_step) + 1
    if len(Values_.shape) == 1:
        Values_mean = np.zeros(total_averaged_rows)
        Values_mean[:] = np.nan
    else:
        Values_mean = np.zeros((total_averaged_rows,Values_.shape[1]))
        Values_mean[:,:] = np.nan
    Index_interp = np.zeros(total_averaged_rows)
    for r_ in range(total_averaged_rows):
        Index_interp[r_] = first_index + (r_ * index_step)

    Index_interp -= (position_ * index_step)

    Values_interp = np.interp(Index_interp, Index_, Values_)

    Index_interp = Index_interp + (position_ * index_step)

    return Index_interp, Values_interp
def array_2D_sort_ascending_by_column(array_, column_=0):
    array_sorted = array_[array_[:, column_].argsort()]
    return array_sorted
def get_ax_range(ax):
    x_1 = ax.axis()[0]
    x_2 = ax.axis()[1]
    y_1 = ax.axis()[2]
    y_2 = ax.axis()[3]
    return x_1, x_2, y_1, y_2
def get_array_perimeter_only(array_):
    return np.concatenate([array_[0, :-1], array_[:-1, -1], array_[-1, ::-1], array_[-2:0:-1, 0]])


# WRF
def wrf_var_search(wrf_nc_file, description_str):
    description_str_lower = description_str.lower()
    for var_ in sorted(wrf_nc_file.variables):
        try:
            if description_str_lower in wrf_nc_file.variables[var_].description.lower():
                print(var_, '|', wrf_nc_file.variables[var_].description)
        except:
            pass
def create_virtual_sonde_from_wrf(sonde_dict, filelist_wrf_output,
                                  wrf_filename_time_format = 'wrfout_d03_%Y-%m-%d_%H_%M_%S'):
    # create time array
    filelist_wrf_output_noPath = []
    for filename_ in filelist_wrf_output:
        filelist_wrf_output_noPath.append(filename_.split('/')[-1])
    wrf_time_file_list = np.array(time_str_to_seconds(filelist_wrf_output_noPath, wrf_filename_time_format))

    # create lat and lon arrays
    wrf_domain_file = nc.Dataset(filelist_wrf_output[0])
    # p(sorted(wrf_domain_file.variables))
    # wrf_vars = sorted(wrf_domain_file.variables)
    # for i_ in range(len(wrf_vars)):
    #     try:
    #         print(wrf_vars[i_], '\t\t', wrf_domain_file.variables[wrf_vars[i_]].description)
    #     except:
    #         print(wrf_vars[i_])

    wrf_lat = wrf_domain_file.variables['XLAT'][0, :, :].filled(np.nan)
    wrf_lon = wrf_domain_file.variables['XLONG'][0, :, :].filled(np.nan)
    wrf_lat_U = wrf_domain_file.variables['XLAT_U'][0, :, :].filled(np.nan)
    wrf_lon_U = wrf_domain_file.variables['XLONG_U'][0, :, :].filled(np.nan)
    wrf_lat_V = wrf_domain_file.variables['XLAT_V'][0, :, :].filled(np.nan)
    wrf_lon_V = wrf_domain_file.variables['XLONG_V'][0, :, :].filled(np.nan)
    wrf_domain_file.close()


    # load sonde's profile
    sonde_hght = sonde_dict['hght']  # m ASL
    sonde_pres = sonde_dict['pres']  # hPa
    sonde_time = sonde_dict['time']  # seconds since epoc
    sonde_lati = sonde_dict['lati']  # degrees
    sonde_long = sonde_dict['long']  # degrees


    # create output lists of virtual sonde
    list_p__ = []
    list_hgh = []
    list_th_ = []
    list_th0 = []
    list_qv_ = []
    list_U__ = []
    list_V__ = []
    list_tim = []
    list_lat = []
    list_lon = []


    wrf_point_abs_address_old = 0

    # loop thru real sonde's points
    for t_ in range(sonde_hght.shape[0]):
        p_progress_bar(t_, sonde_hght.shape[0])
        point_hght = sonde_hght[t_]
        point_pres = sonde_pres[t_]
        point_time = sonde_time[t_]
        point_lati = sonde_lati[t_]
        point_long = sonde_long[t_]

        # find closest cell via lat, lon
        index_tuple = find_index_from_lat_lon_2D_arrays(wrf_lat,wrf_lon, point_lati,point_long)
        index_tuple_U = find_index_from_lat_lon_2D_arrays(wrf_lat_U,wrf_lon_U, point_lati,point_long)
        index_tuple_V = find_index_from_lat_lon_2D_arrays(wrf_lat_V,wrf_lon_V, point_lati,point_long)

        # find closest file via time
        file_index = time_to_row_sec(wrf_time_file_list, point_time)

        # open wrf file
        wrf_domain_file = nc.Dataset(filelist_wrf_output[file_index])
        # get pressure array from wrf
        wrf_press = (wrf_domain_file.variables['PB'][0, :, index_tuple[0], index_tuple[1]].data +
                     wrf_domain_file.variables['P'][0, :, index_tuple[0], index_tuple[1]].data) / 100  # hPa

        # find closest model layer via pressure
        layer_index = find_min_index_1d_array(np.abs(wrf_press - point_pres))

        # define point absolute address and check if it is a new point
        wrf_point_abs_address_new = (index_tuple[0], index_tuple[1], file_index, layer_index)
        if wrf_point_abs_address_new != wrf_point_abs_address_old:
            wrf_point_abs_address_old = wrf_point_abs_address_new

            # get wrf data
            index_tuple_full   = (0, layer_index, index_tuple[0], index_tuple[1])
            index_tuple_full_U = (0, layer_index, index_tuple_U[0], index_tuple_U[1])
            index_tuple_full_V = (0, layer_index, index_tuple_V[0], index_tuple_V[1])


            # save to arrays
            list_p__.append(float(wrf_press[layer_index]))
            list_hgh.append(float(point_hght))
            list_th_.append(float(wrf_domain_file.variables['T'][index_tuple_full]))
            list_th0.append(float(wrf_domain_file.variables['T00'][0]))
            list_qv_.append(float(wrf_domain_file.variables['QVAPOR'][index_tuple_full]))
            list_U__.append(float(wrf_domain_file.variables['U'][index_tuple_full_U]))
            list_V__.append(float(wrf_domain_file.variables['V'][index_tuple_full_V]))
            list_tim.append(float(wrf_time_file_list[file_index]))
            list_lat.append(float(wrf_lat[index_tuple[0], index_tuple[1]]))
            list_lon.append(float(wrf_lon[index_tuple[0], index_tuple[1]]))

            wrf_domain_file.close()


    # convert lists to arrays
    array_p__ = np.array(list_p__)
    array_hgh = np.array(list_hgh)
    array_th_ = np.array(list_th_)
    array_th0 = np.array(list_th0)
    array_qv_ = np.array(list_qv_)
    array_U__ = np.array(list_U__)
    array_V__ = np.array(list_V__)
    array_tim = np.array(list_tim)
    array_lat = np.array(list_lat)
    array_lon = np.array(list_lon)

    # calculate derivative variables
    wrf_temp_K = calculate_temperature_from_potential_temperature(array_th_ + array_th0, array_p__)
    wrf_temp_C = kelvin_to_celsius(wrf_temp_K)
    wrf_e = MixR2VaporPress(array_qv_, array_p__*100)
    wrf_td_C = DewPoint(wrf_e)
    wrf_td_C[wrf_td_C > wrf_temp_C] = wrf_temp_C[wrf_td_C > wrf_temp_C]
    wrf_RH = calculate_RH_from_QV_T_P(array_qv_, wrf_temp_K,  array_p__*100)
    wrf_WD, wrf_WS = cart_to_polar(array_V__, array_U__)
    wrf_WD_met = wrf_WD + 180
    wrf_WD_met[wrf_WD_met >= 360] = wrf_WD_met[wrf_WD_met >= 360] - 360
    wrf_WS_knots = ws_ms_to_knots(wrf_WS)

    # create virtual sonde dict
    wrf_sonde_dict = {}
    wrf_sonde_dict['hght'] = array_hgh
    wrf_sonde_dict['pres'] = array_p__
    wrf_sonde_dict['temp'] = wrf_temp_C
    wrf_sonde_dict['dwpt'] = wrf_td_C
    wrf_sonde_dict['sknt'] = wrf_WS_knots
    wrf_sonde_dict['drct'] = wrf_WD_met
    wrf_sonde_dict['relh'] = wrf_RH
    wrf_sonde_dict['time'] = array_tim
    wrf_sonde_dict['lati'] = array_lat
    wrf_sonde_dict['long'] = array_lon



    return wrf_sonde_dict
def wrf_get_temp_K(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_press = (wrf_domain_file.variables['PB'][0, :, :, :].data +
                 wrf_domain_file.variables['P'][0, :, :, :].data) / 100  # hPa

    wrf_theta = (wrf_domain_file.variables['T'][0, :, :, :].data +
                 wrf_domain_file.variables['T00'][0].data) # K

    wrf_temp_K = calculate_temperature_from_potential_temperature(wrf_theta, wrf_press)

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_temp_K
def wrf_get_press_hPa(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_press = (wrf_domain_file.variables['PB'][0, :, :, :].data +
                 wrf_domain_file.variables['P'][0, :, :, :].data) / 100  # hPa

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_press
def wrf_get_height_m(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_height = (wrf_domain_file.variables['PH'][0,:-1,:,:].data +
                  wrf_domain_file.variables['PHB'][0,:-1,:,:].data) / gravity_

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_height
def wrf_get_terrain_height_m(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_height = (wrf_domain_file.variables['PH'][0,0,:,:].data +
                  wrf_domain_file.variables['PHB'][0,0,:,:].data) / gravity_

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_height
def wrf_get_water_vapor_mixing_ratio(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_QVAPOR = wrf_domain_file.variables['QVAPOR'][0,:,:,:].data

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_QVAPOR
def wrf_get_cloud_water_mixing_ratio(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_QCLOUD = wrf_domain_file.variables['QCLOUD'][0,:,:,:].data

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_QCLOUD
def wrf_get_ice_mixing_ratio(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_QICE = wrf_domain_file.variables['QICE'][0,:,:,:].data

    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_QICE
def wrf_get_lat_lon(wrf_nc):
    original_arg_type_str = False
    if type(wrf_nc) == str:
        original_arg_type_str = True
        wrf_domain_file = nc.Dataset(wrf_nc)
    else:
        wrf_domain_file = wrf_nc

    # get pressure array from wrf
    wrf_lat = wrf_domain_file.variables['XLAT'][0, :, :].filled(np.nan)
    wrf_lon = wrf_domain_file.variables['XLONG'][0, :, :].filled(np.nan)


    if original_arg_type_str:
        wrf_domain_file.close()

    return wrf_lat, wrf_lon
def wrf_rename_files_fix_time_format(filename_original_list, original_character=':', replacement_character='_'):
    for i_, filename_ in enumerate(filename_original_list):
        p_progress_bar(i_, len(filename_original_list))
        new_filename = filename_.replace(original_character,replacement_character)
        os.rename(filename_, new_filename)



# meteorology
def calculate_saturation_vapor_pressure_wexler(T_array_K):
    # result in mb (hPa)
    G0 = -0.29912729E+4
    G1 = -0.60170128E+4
    G2 =  0.1887643854E+2
    G3 = -0.28354721E-1
    G4 =  0.17838301E-4
    G5 = -0.84150417E-9
    G6 =  0.44412543E-12
    G7 =  0.2858487E+1

    e_s = np.exp((G0 * (T_array_K ** -2)) +
                 (G1 * (T_array_K ** -1)) +
                 G2 +
                 (G3 * T_array_K) +
                 (G4 * (T_array_K ** 2)) +
                 (G5 * (T_array_K ** 3)) +
                 (G6 * (T_array_K ** 4)) +
                 (G7 * np.log(T_array_K)))
    return e_s * 0.01
def calculate_saturation_mixing_ratio(P_array_mb, T_array_K):
    e_s = calculate_saturation_vapor_pressure_wexler(T_array_K)
    q_s = 621.97 * (e_s / (P_array_mb - e_s))
    return q_s
def calculate_potential_temperature(T_array_K, P_array_hPa):
    potential_temp = T_array_K * ((1000 / P_array_hPa) ** poisson_)
    return potential_temp
def calculate_equivalent_potential_temperature(T_array_K, P_array_hPa, R_array_kg_over_kg):
    P_o = 1000
    T_e = T_array_K + (latent_heat_v * R_array_kg_over_kg / heat_capacity__Cp)
    theta_e = T_e * ((P_o/P_array_hPa)**poisson_)
    return theta_e
def calculate_temperature_from_potential_temperature(theta_array_K, P_array_hPa):
    temperature_ = theta_array_K * ( (P_array_hPa/1000) ** poisson_ )
    return temperature_
def calculate_mountain_height_from_sonde(sonde_dict):
    """
    calculates H_hat from given values of u_array, v_array, T_array, effective_height, rh_array, q_array, p_array
    """
    # Set initial conditions
    height = 1000  # metres

    # define arrays
    WS_array = ws_knots_to_ms(sonde_dict['SKNT'])
    U_array, V_array = polar_to_cart(sonde_dict['DRCT'], WS_array)
    T_array = celsius_to_kelvin(sonde_dict['TEMP'])
    RH_array = sonde_dict['RELH']
    P_array = sonde_dict['PRES']
    Z_array = sonde_dict['HGHT']
    Q_array = sonde_dict['MIXR']/1000
    TH_array = sonde_dict['THTA']

    # calculated arrays
    q_s = calculate_saturation_mixing_ratio(P_array, T_array)
    e_ = gas_const_dry / gas_const_water

    # gradients
    d_ln_TH = np.gradient(np.log(TH_array))
    d_z = np.gradient(Z_array)
    d_q_s = np.gradient(q_s)

    # Dry Brunt - Vaisala
    N_dry = gravity_ * d_ln_TH / d_z
    N_dry[RH_array >= 90] = 0


    # Moist Brunt - Vaisala
    term_1_1 = 1 + (  latent_heat_v * q_s / (gas_const_dry * T_array)  )
    term_1_2 = 1 + (  e_ * (latent_heat_v**2) * q_s / (heat_capacity__Cp * gas_const_dry * (T_array**2) )  )

    term_2_1 = d_ln_TH / d_z
    term_2_2 = latent_heat_v / (heat_capacity__Cp * T_array)
    term_2_3 = d_q_s / d_z

    term_3 = d_q_s / d_z # should be d_q_w but sonde data has no cloud water data

    N_moist = gravity_ * (  (term_1_1 / term_1_2) * (term_2_1 + ( term_2_2 * term_2_3) )  -  term_3  )
    N_moist[RH_array < 90] = 0

    # define output array
    N_2 = (N_dry + N_moist)**2


    H_hat_2 = N_2 * (height**2) / (U_array**2)

    return H_hat_2
def calculate_mountain_height_from_era5(era5_pressures_filename, era5_surface_filename, point_lat, point_lon,
                                        return_arrays=False, u_wind_mode='u', range_line_degrees=None,
                                        time_start_str_YYYYmmDDHHMM='',time_stop_str_YYYYmmDDHHMM='',
                                        reference_height=1000, return_debug_arrays=False):
    """
    calculates H_hat from given values of u_array, v_array, T_array, effective_height, rh_array, q_array, p_array
    u_wind_mode: can be u, wind_speed, normal_to_range. If normal_to_range, then range_line most not be none
    if range_line_degrees is not None, u_wind_mode will automatically be set to normal_to_range

    range_line_degrees: degress (decimals) from north, clockwise, of the mountain range line.
    """

    # load files
    era5_sur = nc.Dataset(era5_surface_filename, 'r')
    era5_pre = nc.Dataset(era5_pressures_filename, 'r')


    # check if times are the same for both files
    dif_sum = np.sum(np.abs(era5_pre.variables['time'][:] - era5_sur.variables['time'][:]))
    if dif_sum > 0:
        print('Error, times in selected files are not the same')
        return

    # check if lat lon are the same for both files
    dif_sum = np.sum(np.abs(era5_pre.variables['latitude'][:] - era5_sur.variables['latitude'][:]))
    dif_sum = dif_sum + np.sum(np.abs(era5_pre.variables['longitude'][:] - era5_sur.variables['longitude'][:]))
    if dif_sum > 0:
        print('Error, latitude or longitude in selected files are not the same')
        return

    # find lat lon index
    lat_index, lon_index = find_index_from_lat_lon(era5_sur.variables['latitude'][:],
                                                   era5_sur.variables['longitude'][:], [point_lat], [point_lon])
    lat_index = lat_index[0]
    lon_index = lon_index[0]


    # copy arrays
    time_array = time_era5_to_seconds(np.array(era5_sur.variables['time'][:]))
    r_1 = 0
    r_2 = -1
    if time_start_str_YYYYmmDDHHMM != '':
        r_1 = time_to_row_str(time_array, time_start_str_YYYYmmDDHHMM)
    if time_stop_str_YYYYmmDDHHMM != '':
        r_2 = time_to_row_str(time_array, time_stop_str_YYYYmmDDHHMM)
    time_array = time_array[r_1:r_2]

    sp_array = np.array(era5_sur.variables['sp'][r_1:r_2, lat_index, lon_index]) / 100 # hPa
    P_array =  np.array(era5_pre.variables['level'][:]) # hPa
    if range_line_degrees is not None:
        WD_, WS_ = cart_to_polar(np.array(era5_pre.variables['v'][r_1:r_2,:,lat_index,lon_index]).flatten(),
                                 np.array(era5_pre.variables['u'][r_1:r_2,:,lat_index,lon_index]).flatten())
        WD_delta = WD_ - range_line_degrees

        range_normal_component = WS_ * np.sin(np.deg2rad(WD_delta))
        U_array = range_normal_component.reshape((sp_array.shape[0], P_array.shape[0]))
    else:
        if u_wind_mode == 'u':
            U_array =  np.array(era5_pre.variables['u'][r_1:r_2,:,lat_index,lon_index])
        else:
            U_array = np.sqrt(np.array(era5_pre.variables['v'][r_1:r_2,:,lat_index,lon_index]) ** 2 +
                              np.array(era5_pre.variables['u'][r_1:r_2,:,lat_index,lon_index]) ** 2)
    T_array = np.array(era5_pre.variables['t'][r_1:r_2, :, lat_index, lon_index])
    Q_L_array = np.array(era5_pre.variables['crwc'][r_1:r_2, :, lat_index, lon_index])
    RH_array = np.array(era5_pre.variables['r'][r_1:r_2, :, lat_index, lon_index])
    Z_array = np.array(era5_pre.variables['z'][r_1:r_2, :, lat_index, lon_index]) / gravity_


    # calculate arrays
    TH_array = np.zeros((time_array.shape[0], P_array.shape[0]), dtype=float)
    for t_ in range(time_array.shape[0]):
        TH_array[t_,:] = calculate_potential_temperature(T_array[t_,:], P_array[:])

    # calculated arrays
    q_s = calculate_saturation_mixing_ratio(P_array, T_array)
    e_ = gas_const_dry / gas_const_water

    # create output dict
    H_hat_2 = {}

    # loop tru time stamps
    for t_ in range(time_array.shape[0]):
        p_progress_bar(t_,time_array.shape[0])

        # find surface pressure at this time stamp
        surface_p = sp_array[t_]

         # find pressure at 1000 meters
        pressure_1000m = np.interp(reference_height, Z_array[t_, :], P_array)
        pressure_1000m_index = np.argmin(np.abs(P_array - pressure_1000m))

        # find extrapolations
        ql_0  = np.interp(np.log(surface_p), np.log(P_array), Q_L_array[t_, :])
        z__0  = np.interp(np.log(surface_p), np.log(P_array), Z_array[t_, :])
        th_0  = np.interp(np.log(surface_p), np.log(P_array), TH_array[t_, :])
        qs_0  = np.interp(np.log(surface_p), np.log(P_array), q_s[t_, :])

        t__1000 = np.interp(reference_height, Z_array[t_, :], T_array[t_, :])
        u__1000 = np.interp(reference_height, Z_array[t_, :], U_array[t_, :])
        ql_1000 = np.interp(reference_height, Z_array[t_, :], Q_L_array[t_, :])
        z__1000 = reference_height
        th_1000 = np.interp(reference_height, Z_array[t_, :], TH_array[t_, :])
        qs_1000 = np.interp(reference_height, Z_array[t_, :], q_s[t_, :])


        # gradients
        d_ln_TH = np.log(th_1000) - np.log(th_0)
        d_z     = z__1000 - z__0
        d_q_s   = qs_1000 - qs_0
        d_q_w   = (d_q_s) + (ql_1000 - ql_0)


        # Brunt - Vaisala
        if np.max(RH_array[t_, pressure_1000m_index:])>= 90:
            # Moist
            term_1_1 = 1 + (  latent_heat_v * qs_1000 / (gas_const_dry * t__1000)  )
            term_1_2 = 1 + (  e_ * (latent_heat_v**2) * qs_1000 /
                              (heat_capacity__Cp * gas_const_dry * (t__1000**2) )  )

            term_2_1 = d_ln_TH / d_z
            term_2_2 = latent_heat_v / (heat_capacity__Cp * t__1000)
            term_2_3 = d_q_s / d_z

            term_3 = d_q_w / d_z

            N_2 = gravity_ * (  (term_1_1 / term_1_2) * (term_2_1 + ( term_2_2 * term_2_3) )  -  term_3  )

        else:
            # Dry
            N_2 = gravity_ * d_ln_TH / d_z

        # populate each time stamp
        H_hat_2[time_array[t_]] = N_2 * (reference_height ** 2) / (u__1000 ** 2)

    era5_sur.close()
    era5_pre.close()

    if return_arrays:
        H_hat_2_time = sorted(H_hat_2.keys())
        H_hat_2_time = np.array(H_hat_2_time)
        H_hat_2_vals = np.zeros(H_hat_2_time.shape[0], dtype=float)
        for r_ in range(H_hat_2_time.shape[0]):
            H_hat_2_vals[r_] = H_hat_2[H_hat_2_time[r_]]
        if return_debug_arrays:
            return H_hat_2_time, H_hat_2_vals, N_2, u__1000 ** 2
        else:
            return H_hat_2_time, H_hat_2_vals
    else:
        return H_hat_2
def calculate_mountain_height_from_WRF(filename_SP, filename_PR,
                                       filename_UU, filename_VV,
                                       filename_TH, filename_QR,
                                       filename_QV, filename_PH,
                                       return_arrays=False, u_wind_mode='u', range_line_degrees=None,
                                       reference_height=1000):
    """
    calculates H_hat from WRF point output text files
    u_wind_mode: can be u, wind_speed, normal_to_range. If normal_to_range, then range_line most not be none
    if range_line_degrees is not None, u_wind_mode will automatically be set to normal_to_range

    range_line_degrees: degress (decimals) from north, clockwise, of the mountain range line.
    :param filename_SP: fullpath filename of surface pressure file
    :param filename_PR: fullpath filename of pressure file
    :param filename_UU: fullpath filename of u wind file
    :param filename_VV: fullpath filename of v wind file
    :param filename_TH: fullpath filename of potential temperature file
    :param filename_QR: fullpath filename of rain water mixing ratio file
    :param filename_QV: fullpath filename of Water vapor mixing ratio file
    :param filename_PH: fullpath filename of geopotential height file
    :param return_arrays: if true, will return also brunt vaisalla and wind component squared
    :param u_wind_mode: can be u, wind_speed, normal_to_range. If normal_to_range, then range_line most not be none
    :param range_line_degrees: if not None, u_wind_mode will automatically be set to normal_to_range
    :param reference_height: mean height of mountain range
    :return:
    H_hat_2
    """


    # load arrays from text
    SP_array = genfromtxt(filename_SP, dtype=float, skip_header=1)[:,9] / 100 # hPa
    PR_array =  genfromtxt(filename_PR, dtype=float, skip_header=1)[:,1:] / 100 # hPa
    UU_array =  genfromtxt(filename_UU, dtype=float, skip_header=1)[:,1:]
    VV_array =  genfromtxt(filename_VV, dtype=float, skip_header=1)[:,1:]
    TH_array =  genfromtxt(filename_TH, dtype=float, skip_header=1)[:,1:]
    QR_array =  genfromtxt(filename_QR, dtype=float, skip_header=1)[:,1:]
    QV_array =  genfromtxt(filename_QV, dtype=float, skip_header=1)[:,1:]
    Z_array =  genfromtxt(filename_PH, dtype=float, skip_header=1)[:,1:] # already in meters

    # calculate arrays
    if range_line_degrees is not None:
        WD_, WS_ = cart_to_polar(UU_array.flatten(), VV_array.flatten())
        WD_delta = WD_ - range_line_degrees

        range_normal_component = WS_ * np.sin(np.deg2rad(WD_delta))
        U_array = range_normal_component.reshape((UU_array.shape[0], UU_array.shape[1]))
    else:
        if u_wind_mode == 'u':
            U_array =  UU_array
        else:
            U_array = np.sqrt(UU_array ** 2 + VV_array ** 2)

    T_array = calculate_temperature_from_potential_temperature(TH_array, PR_array)
    RH_array = calculate_RH_from_QV_T_P(QV_array, T_array, PR_array*100)

    q_s = calculate_saturation_mixing_ratio(PR_array, T_array)
    e_ = gas_const_dry / gas_const_water

    # create output array
    H_hat_2 = np.zeros(PR_array.shape[0], dtype=float)

    # loop tru time stamps
    for r_ in range(PR_array.shape[0]):
        p_progress_bar(r_, PR_array.shape[0])

        # find surface pressure at this time stamp
        surface_p = SP_array[r_]

         # find pressure at 1000 meters
        pressure_1000m = np.interp(reference_height, Z_array[r_, :], PR_array[r_, :])
        pressure_1000m_index = np.argmin(np.abs(PR_array[r_, :] - pressure_1000m))

        # find extrapolations
        ql_0  = np.interp(np.log(surface_p), np.log(PR_array[r_, :]), QR_array[r_, :])
        z__0  = np.interp(np.log(surface_p), np.log(PR_array[r_, :]), Z_array[r_, :])
        th_0  = np.interp(np.log(surface_p), np.log(PR_array[r_, :]), TH_array[r_, :])
        qs_0  = np.interp(np.log(surface_p), np.log(PR_array[r_, :]), q_s[r_, :])

        t__1000 = np.interp(reference_height, Z_array[r_, :], T_array[r_, :])
        u__1000 = np.interp(reference_height, Z_array[r_, :], U_array[r_, :])
        ql_1000 = np.interp(reference_height, Z_array[r_, :], QR_array[r_, :])
        z__1000 = reference_height
        th_1000 = np.interp(reference_height, Z_array[r_, :], TH_array[r_, :])
        qs_1000 = np.interp(reference_height, Z_array[r_, :], q_s[r_, :])


        # gradients
        d_ln_TH = np.log(th_1000) - np.log(th_0)
        d_z     = z__1000 - z__0
        d_q_s   = qs_1000 - qs_0
        d_q_w   = (d_q_s) + (ql_1000 - ql_0)


        # Brunt - Vaisala
        if np.max(RH_array[r_, pressure_1000m_index:])>= 90:
            # Moist
            term_1_1 = 1 + (  latent_heat_v * qs_1000 / (gas_const_dry * t__1000)  )
            term_1_2 = 1 + (  e_ * (latent_heat_v**2) * qs_1000 /
                              (heat_capacity__Cp * gas_const_dry * (t__1000**2) )  )

            term_2_1 = d_ln_TH / d_z
            term_2_2 = latent_heat_v / (heat_capacity__Cp * t__1000)
            term_2_3 = d_q_s / d_z

            term_3 = d_q_w / d_z

            N_2 = gravity_ * (  (term_1_1 / term_1_2) * (term_2_1 + ( term_2_2 * term_2_3) )  -  term_3  )

        else:
            # Dry
            N_2 = gravity_ * d_ln_TH / d_z

        # populate each time stamp
        H_hat_2[r_] = N_2 * (reference_height ** 2) / (u__1000 ** 2)


    if return_arrays:
        return H_hat_2, N_2, u__1000 ** 2
    else:
        return H_hat_2
def calculate_dewpoint_from_T_RH(T_, RH_):
    """
    from Magnus formula, using Bolton's constants
    :param T_: ambient temperature [Celsius]
    :param RH_: relative humidity
    :return: Td_ dew point temperature [celsius]
    """
    a = 6.112
    b = 17.67
    c = 243.5

    y_ = np.log(RH_/100) + ((b*T_)/(c+T_))

    Td_ = (c * y_) / (b - y_)

    return Td_
def calculate_RH_from_QV_T_P(arr_qvapor, arr_temp_K, arr_press_Pa):
    tv_ = 6.11 * e_constant**((2500000/461) * ((1/273) - (1/arr_temp_K)))
    pv_ = arr_qvapor * (arr_press_Pa/100) / (arr_qvapor + 0.622)
    return np.array(100 * pv_ / tv_)
def calculate_profile_input_for_cluster_analysis_from_ERA5(p_profile, t_profile, td_profile, q_profile,
                                                           u_profile, v_profile, h_profile, surface_p):
    """
    takes data from ERA5 for only one time stamp for all pressure levels from 250 to 1000 hPa
    :param p_profile: in hPa
    :param t_profile: in Celsius
    :param td_profile: in Celsius
    :param q_profile: in kg/kg
    :param u_profile: in m/s
    :param v_profile: in m/s
    :param h_profile: in m
    :param surface_p: in hPa
    :return: surface_p, qv_, qu_, tw_, sh_, tt_
    """

    # trim profiles from surface to top
    # find which levels should be included
    levels_total = 0
    for i_ in range(p_profile.shape[0]):
        if p_profile[i_] > surface_p:
            break
        levels_total += 1

    ####################################### find extrapolations
    surface_t = np.interp(np.log(surface_p), np.log(p_profile), t_profile)
    surface_td = np.interp(np.log(surface_p), np.log(p_profile), td_profile)
    surface_q = np.interp(np.log(surface_p), np.log(p_profile), q_profile)
    surface_u = np.interp(np.log(surface_p), np.log(p_profile), u_profile)
    surface_v = np.interp(np.log(surface_p), np.log(p_profile), v_profile)
    surface_h = np.interp(np.log(surface_p), np.log(p_profile), h_profile)

    # create temp arrays
    T_array = np.zeros(levels_total + 1, dtype=float)
    Td_array = np.zeros(levels_total + 1, dtype=float)
    Q_array = np.zeros(levels_total + 1, dtype=float)
    U_array = np.zeros(levels_total + 1, dtype=float)
    V_array = np.zeros(levels_total + 1, dtype=float)
    H_array = np.zeros(levels_total + 1, dtype=float)
    P_array = np.zeros(levels_total + 1, dtype=float)

    T_array[:levels_total] = t_profile[:levels_total]
    Td_array[:levels_total] = td_profile[:levels_total]
    Q_array[:levels_total] = q_profile[:levels_total]
    U_array[:levels_total] = u_profile[:levels_total]
    V_array[:levels_total] = v_profile[:levels_total]
    H_array[:levels_total] = h_profile[:levels_total]
    P_array[:levels_total] = p_profile[:levels_total]

    T_array[-1] = surface_t
    Td_array[-1] = surface_td
    Q_array[-1] = surface_q
    U_array[-1] = surface_u
    V_array[-1] = surface_v
    H_array[-1] = surface_h
    P_array[-1] = surface_p
    ######################################

    r_850 = np.argmin(np.abs(P_array - 850))
    r_500 = np.argmin(np.abs(P_array - 500))

    dp_ = np.abs(np.gradient(P_array))
    tt_ = (T_array[r_850] - (2 * T_array[r_500]) + Td_array[r_850])

    qu_ = np.sum(Q_array * U_array * dp_) / gravity_
    qv_ = np.sum(Q_array * V_array * dp_) / gravity_
    tw_ = np.sum(Q_array * dp_) / gravity_

    del_u = U_array[r_850] - U_array[r_500]
    del_v = V_array[r_850] - V_array[r_500]
    del_z = H_array[r_850] - H_array[r_500]

    sh_ = ((del_u / del_z) ** 2 + (del_v / del_z) ** 2) ** 0.5

    return surface_p, qv_, qu_, tw_, sh_, tt_
def barometric_equation(presb_pa, tempb_k, deltah_m, Gamma=-0.0065):
    """The barometric equation models the change in pressure with
    height in the atmosphere.

    INPUTS:
    presb_k (pa):     The base pressure
    tempb_k (K):      The base temperature
    deltah_m (m):     The height differential between the base height and the
                      desired height
    Gamma [=-0.0065]: The atmospheric lapse rate

    OUTPUTS
    pres (pa):        Pressure at the requested level

    REFERENCE:
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    return presb_pa * \
        (tempb_k/(tempb_k+Gamma*deltah_m))**(grav*m_a/(Rstar_a*Gamma))
def barometric_equation_inv(heightb_m, tempb_k, presb_pa,
                            prest_pa, Gamma=-0.0065):
    """The barometric equation models the change in pressure with height in
    the atmosphere. This function returns altitude given
    initial pressure and base altitude, and pressure change.

    INPUTS:
    heightb_m (m):
    presb_pa (pa):    The base pressure
    tempb_k (K)  :    The base temperature
    deltap_pa (m):    The pressure differential between the base height and the
                      desired height

    Gamma [=-0.0065]: The atmospheric lapse rate

    OUTPUTS
    heightt_m

    REFERENCE:
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    return heightb_m + \
        tempb_k * ((presb_pa/prest_pa)**(Rstar_a*Gamma/(grav*m_a))-1) / Gamma
def Theta(tempk, pres, pref=100000.):
    """Potential Temperature

    INPUTS:
    tempk (K)
    pres (Pa)
    pref: Reference pressure (default 100000 Pa)

    OUTPUTS: Theta (K)

    Source: Wikipedia
    Prints a warning if a pressure value below 2000 Pa input, to ensure
    that the units were input correctly.
    """

    try:
        minpres = min(pres)
    except TypeError:
        minpres = pres

    if minpres < 2000:
        print("WARNING: P<2000 Pa; did you input a value in hPa?")

    return tempk * (pref/pres)**(Rs_da/Cp_da)
def TempK(theta, pres, pref=100000.):
    """Inverts Theta function."""

    try:
        minpres = min(pres)
    except TypeError:
        minpres = pres

    if minpres < 2000:
        print("WARNING: P<2000 Pa; did you input a value in hPa?")

    return theta * (pres/pref)**(Rs_da/Cp_da)
def ThetaE(tempk, pres, e):
    """Calculate Equivalent Potential Temperature
        for lowest model level (or surface)

    INPUTS:
    tempk:      Temperature [K]
    pres:       Pressure [Pa]
    e:          Water vapour partial pressure [Pa]

    OUTPUTS:
    theta_e:    equivalent potential temperature

    References:
    Eq. (9.40) from Holton (2004)
    Eq. (22) from Bolton (1980)
    Michael P. Byrne and Paul A. O'Gorman (2013), 'Land-Ocean Warming
    Contrast over a Wide Range of Climates: Convective Quasi-Equilibrium
    Theory and Idealized Simulations', J. Climate """

    # tempc
    tempc = tempk - degCtoK
    # Calculate theta
    theta = Theta(tempk, pres)

    # T_lcl formula needs RH
    es = VaporPressure(tempc)
    RH = 100. * e / es

    # theta_e needs q (water vapour mixing ratio)
    qv = MixRatio(e, pres)

    # Calculate the temp at the Lifting Condensation Level
    T_lcl = ((tempk-55)*2840 / (2840-(np.log(RH/100)*(tempk-55)))) + 55

    # print "T_lcl :%.3f"%T_lcl

    # DEBUG STUFF ####
    theta_l = tempk * \
        (100000./(pres-e))**(Rs_da/Cp_da)*(tempk/T_lcl)**(0.28*qv)
    # print "theta_L: %.3f"%theta_l

    # Calculate ThetaE
    theta_e = theta_l * np.exp((Lv * qv) / (Cp_da * T_lcl))

    return theta_e
def ThetaE_Bolton(tempk, pres, e, pref=100000.):
    """Theta_E following Bolton (1980)
    INPUTS:
    tempk:      Temperature [K]
    pres:       Pressure [Pa]
    e:          Water vapour partial pressure [Pa]

    See http://en.wikipedia.org/wiki/Equivalent_potential_temperature
    """

    # Preliminary:
    T = tempk
    qv = MixRatio(e, pres)
    Td = DewPoint(e) + degCtoK
    kappa_d = Rs_da / Cp_da

    # Calculate TL (temp [K] at LCL):
    TL = 56 + ((Td-56.)**-1+(np.log(T/Td)/800.))**(-1)

    # print "TL: %.3f"%TL

    # Calculate Theta_L:
    thetaL = T * (pref/(pres-e))**kappa_d*(T/TL)**(0.28*qv)

    # print "theta_L: %.3f"%thetaL

    # put it all together to get ThetaE
    thetaE = thetaL * np.exp((3036./TL-0.78)*qv*(1+0.448*qv))

    return thetaE
def ThetaV(tempk, pres, e):
    """Virtual Potential Temperature

    INPUTS
    tempk (K)
    pres (Pa)
    e: Water vapour pressure (Pa) (Optional)

    OUTPUTS
    theta_v    : Virtual potential temperature
    """

    mixr = MixRatio(e, pres)
    theta = Theta(tempk, pres)

    return theta * (1+mixr/Epsilon) / (1+mixr)
def GammaW(tempk, pres):
    """Function to calculate the moist adiabatic lapse rate (deg C/Pa) based
    on the environmental temperature and pressure.

    INPUTS:
    tempk (K)
    pres (Pa)
    RH (%)

    RETURNS:
    GammaW: The moist adiabatic lapse rate (Deg C/Pa)
    REFERENCE:
    http://glossary.ametsoc.org/wiki/Moist-adiabatic_lapse_rate
    (Note that I multiply by 1/(grav*rho) to give MALR in deg/Pa)

    """

    tempc = tempk-degCtoK
    es = VaporPressure(tempc)
    ws = MixRatio(es, pres)

    # tempv=VirtualTempFromMixR(tempk,ws)
    tempv = VirtualTemp(tempk, pres, es)
    latent = Latentc(tempc)

    Rho = pres / (Rs_da*tempv)

    # This is the previous implementation:
    # A=1.0+latent*ws/(Rs_da*tempk)
    # B=1.0+Epsilon*latent*latent*ws/(Cp_da*Rs_da*tempk*tempk)
    # Gamma=(A/B)/(Cp_da*Rho)

    # This is algebraically identical but a little clearer:
    A = -1. * (1.0+latent*ws/(Rs_da*tempk))
    B = Rho * (Cp_da+Epsilon*latent*latent*ws/(Rs_da*tempk*tempk))
    Gamma = A / B

    return Gamma
def DensHumid(tempk, pres, e):
    """Density of moist air.
    This is a bit more explicit and less confusing than the method below.

    INPUTS:
    tempk: Temperature (K)
    pres: static pressure (Pa)
    mixr: mixing ratio (kg/kg)

    OUTPUTS:
    rho_air (kg/m^3)

    SOURCE: http://en.wikipedia.org/wiki/Density_of_air
    """

    pres_da = pres - e
    rho_da = pres_da / (Rs_da * tempk)
    rho_wv = e/(Rs_v * tempk)

    return rho_da + rho_wv
def Density(tempk, pres, mixr):
    """Density of moist air

    INPUTS:
    tempk: Temperature (K)
    pres: static pressure (Pa)
    mixr: mixing ratio (kg/kg)

    OUTPUTS:
    rho_air (kg/m^3)
    """

    virtualT = VirtualTempFromMixR(tempk, mixr)
    return pres / (Rs_da * virtualT)
def VirtualTemp(tempk, pres, e):
    """Virtual Temperature

    INPUTS:
    tempk: Temperature (K)
    e: vapour pressure (Pa)
    p: static pressure (Pa)

    OUTPUTS:
    tempv: Virtual temperature (K)

    SOURCE: hmmmm (Wikipedia)."""

    tempvk = tempk / (1-(e/pres)*(1-Epsilon))
    return tempvk
def VirtualTempFromMixR(tempk, mixr):
    """Virtual Temperature

    INPUTS:
    tempk: Temperature (K)
    mixr: Mixing Ratio (kg/kg)

    OUTPUTS:
    tempv: Virtual temperature (K)

    SOURCE: hmmmm (Wikipedia). This is an approximation
    based on a m
    """

    return tempk * (1.0+0.6*mixr)
def Latentc(tempc):
    """Latent heat of condensation (vapourisation)

    INPUTS:
    tempc (C)

    OUTPUTS:
    L_w (J/kg)

    SOURCE:
    http://en.wikipedia.org/wiki/Latent_heat#Latent_heat_for_condensation_of_water
    """

    return 1000 * (2500.8 - 2.36*tempc + 0.0016*tempc**2 - 0.00006*tempc**3)
def VaporPressure(tempc, phase="liquid"):
    """Water vapor pressure over liquid water or ice.

    INPUTS:
    tempc: (C) OR dwpt (C), if SATURATION vapour pressure is desired.
    phase: ['liquid'],'ice'. If 'liquid', do simple dew point. If 'ice',
    return saturation vapour pressure as follows:

    Tc>=0: es = es_liquid
    Tc <0: es = es_ice


    RETURNS: e_sat  (Pa)

    SOURCE: http://cires.colorado.edu/~voemel/vp.html (#2:
    CIMO guide (WMO 2008), modified to return values in Pa)

    This formulation is chosen because of its appealing simplicity,
    but it performs very well with respect to the reference forms
    at temperatures above -40 C. At some point I'll implement Goff-Gratch
    (from the same resource).
    """

    over_liquid = 6.112 * np.exp(17.67*tempc/(tempc+243.12))*100.
    over_ice = 6.112 * np.exp(22.46*tempc/(tempc+272.62))*100.
    # return where(tempc<0,over_ice,over_liquid)

    if phase == "liquid":
        # return 6.112*exp(17.67*tempc/(tempc+243.12))*100.
        return over_liquid
    elif phase == "ice":
        # return 6.112*exp(22.46*tempc/(tempc+272.62))*100.
        return np.where(tempc < 0, over_ice, over_liquid)
    else:
        raise NotImplementedError
def SatVap(dwpt, phase="liquid"):
    """This function is deprecated, return ouput from VaporPres"""

    print("WARNING: This function is deprecated, please use VaporPressure()" +
          " instead, with dwpt as argument")
    return VaporPressure(dwpt, phase)
def MixRatio(e, p):
    """Mixing ratio of water vapour
    INPUTS
    e (Pa) Water vapor pressure
    p (Pa) Ambient pressure

    RETURNS
    qv (kg kg^-1) Water vapor mixing ratio`
    """

    return Epsilon * e / (p - e)
def MixR2VaporPress(qv, p):
    """Return Vapor Pressure given Mixing Ratio and Pressure
    INPUTS
    qv (kg kg^-1) Water vapor mixing ratio`
    p (Pa) Ambient pressure

    RETURNS
    e (Pa) Water vapor pressure
    """

    return qv * p / (Epsilon + qv)
def DewPoint(e):
    """ Use Bolton's (1980, MWR, p1047) formulae to find tdew.
    INPUTS:
    e (Pa) Water Vapor Pressure
    OUTPUTS:
    Td (C)
      """

    ln_ratio = np.log(e/611.2)
    Td = ((17.67-ln_ratio)*degCtoK+243.5*ln_ratio)/(17.67-ln_ratio)
    return Td - degCtoK
def WetBulb(tempc, RH):
    """Stull (2011): Wet-Bulb Temperature from Relative Humidity and Air
    Temperature.
    INPUTS:
    tempc (C)
    RH (%)
    OUTPUTS:
    tempwb (C)
    """

    Tw = tempc * np.arctan(0.151977*(RH+8.313659)**0.5) + \
        np.arctan(tempc+RH) - np.arctan(RH-1.676331) + \
        0.00391838*RH**1.5*np.arctan(0.023101*RH) - \
        4.686035

    return Tw


# unit conversions
def convert_unit_and_save_data_ppb_ugm3(filename_, station_name):
    # https://uk-air.defra.gov.uk/assets/documents/reports/cat06/0502160851_Conversion_Factors_Between_ppb_and.pdf
    # http://www2.dmu.dk/AtmosphericEnvironment/Expost/database/docs/PPM_conversion.pdf

    parameters_unit_scaling = {'11' : 1.96, # O3
                               '10' : 1.25, # NO
                               '9' : 1.88, # NO2
                               '16' : 2.62, # SO2
                               '8' : 1.15} # CO

    new_unit_name = '[$\mu$g/m$^3$]'

    parameter_name_mod = {'9' : 'NO$_2$',
                          '11' : 'O$_3$',
                          '12' : 'PM$_1$$_0$',
                          '13' : 'PM$_2$$_.$$_5$',
                          '7' : 'CO$_2$',
                          '16' : 'SO$_2$',
                          }

    # station_name = 'QF_01'


    data_array = open_csv_file(filename_)
    current_header = data_array[0,:]
    new_header = np.array(current_header)
    v_current = np.array(data_array[1:,:],dtype=float)
    v_new = np.array(v_current)

    for keys_ in parameters_unit_scaling.keys():
        v_new[:, int(keys_)] = v_current[:, int(keys_)] * parameters_unit_scaling[str(keys_)]

    # add station name suffix
    for i_ in range(5,22):
        if str(i_) in parameter_name_mod.keys():
            parameter_name = parameter_name_mod[str(i_)]
        else:
            parameter_name = current_header[i_].split('_')[0]

        if str(i_) in parameters_unit_scaling.keys():
            parameter_unit = new_unit_name
        else:
            parameter_unit = current_header[i_].split('_')[1]

        new_header[i_] = station_name + '_' + parameter_name + '_' + parameter_unit


    data_array[1:,:] = v_new
    data_array[0,:] = new_header

    filename_new = filename_.split('\\')[-1].split('.')[0] + '_unit_converted.csv'
    current_filename_without_path = filename_.split('\\')[-1]
    current_filename_path = filename_[:-len(current_filename_without_path)]

    numpy_save_txt(current_filename_path + filename_new, data_array)

    print('done!')
def save_data_with_unit_conversion_ppb_ugm3(file_list_path):
    file_list = sorted(glob.glob(str(file_list_path + '\\' + '*.csv')))

    # https://uk-air.defra.gov.uk/assets/documents/reports/cat06/0502160851_Conversion_Factors_Between_ppb_and.pdf
    # http://www2.dmu.dk/AtmosphericEnvironment/Expost/database/docs/PPM_conversion.pdf

    parameters_unit_scaling = {'12' : 1.96, # O3
                               '13' : 1.25, # NO
                               '14' : 1.88, # NO2
                               '15' : 2.62, # SO2
                               '16' : 1.15} # CO


    parameters_new_names = ['YYYY', # 0
                            'MM', # 1
                            'DD', # 2
                            'HH', # 3
                            'mm', # 4
                            'Day of the week', # 5
                            'WD degrees', # 6
                            'WS m/s', # 7
                            'Temp Celsius', # 8
                            'RH %', # 9
                            'SR W/m2', # 10
                            'ATP mbar', # 11
                            'O3 ug/m3', # 12
                            'NO ug/m3', # 13
                            'NO2 ug/m3', # 14
                            'SO2 ug/m3', # 15
                            'CO mg/m3', # 16
                            'CO2 ppm', # 17
                            'PM10 ug/m3', # 18
                            'PM2.5 ug/m3', # 19
                            'THC ppm', # 20
                            'Rain mm', # 21
                            'Ox ppb', # 22
                            'NOx ppb'] # 23



    for month_ in range(1,13):
        print(month_)

        filename_old = file_list[month_ -1]
        data_array = open_csv_file(file_list[month_ -1])
        v_ppb = np.array(data_array[1:,:],dtype=float)
        v_ug_m3 = np.array(v_ppb)

        for keys_ in parameters_unit_scaling.keys():
            v_ug_m3[:, int(keys_)] = v_ppb[:, int(keys_)] * parameters_unit_scaling[str(keys_)]

        data_array[0, :] = parameters_new_names
        data_array[1:,:] = v_ug_m3

        filename_new = filename_old.split('\\')[-1].split('.')[0] + '_ugm3.csv'

        numpy_save_txt(file_list_path + '\\' + filename_new, data_array)

    print('done!')
def RH_to_abs_conc(arr_RH,arr_T):
    a_ = 1-(373.15/arr_T)
    c_1 = 13.3185
    c_2 = -1.97
    c_3 = -.6445
    c_4 = -.1299
    Po_H2O = 1013.25 * e_constant ** ((c_1 * (a_**1)) +
                                      (c_2 * (a_**2)) +
                                      (c_3 * (a_**3)) +
                                      (c_4 * (a_**4)) )   # mbar

    return (arr_RH * Po_H2O) / (100 * boltzmann_ * arr_T)
def Mixing_Ratio_to_molecules_per_cm3(arr_MR, ATP_mbar, Temp_C):
    arr_temp = Temp_C + 273.15 # kelvin
    arr_Molec_per_cm3 = arr_MR * ( ATP_mbar / ( boltzmann_ * arr_temp ) ) # molecules / cm3
    return arr_Molec_per_cm3
def molecules_per_cm3_to_Mixing_Ratio(arr_Molec_per_cm3, ATP_mbar, Temp_C):
    arr_temp = Temp_C + 273.15 # kelvin
    arr_MR = (arr_Molec_per_cm3 * boltzmann_ * arr_temp) / ATP_mbar
    return arr_MR
def ws_knots_to_ms(arr_):
    return arr_ * .514444
def ws_ms_to_knots(arr_):
    return arr_ / .514444
def kelvin_to_celsius(arr_temp_k):
    return arr_temp_k - 273.15
def celsius_to_kelvin(arr_temp_c):
    return arr_temp_c + 273.15

# geo reference
def find_index_from_lat_lon(series_lat, series_lon, point_lat_list, point_lon_list):
    lat_index_list = []
    lon_index_list = []

    # mask arrays
    lat_m = series_lat
    lon_m = series_lon
    if np.sum(lat_m) != np.sum(lat_m) or np.sum(lon_m) != np.sum(lon_m):
        lat_m = np.ma.masked_where(np.isnan(lat_m), lat_m)
        lat_m = np.ma.masked_where(np.isinf(lat_m), lat_m)
        lon_m = np.ma.masked_where(np.isnan(lon_m), lon_m)
        lon_m = np.ma.masked_where(np.isinf(lon_m), lon_m)


    if type(point_lat_list) == tuple or type(point_lat_list) == list:
        for lat_ in point_lat_list:
            lat_index_list.append(np.argmin(np.abs(lat_m - lat_)))

        for lon_ in point_lon_list:
            lon_index_list.append(np.argmin(np.abs(lon_m - lon_)))
    else:
        lat_index_list = np.argmin(np.abs(lat_m - point_lat_list))
        lon_index_list = np.argmin(np.abs(lon_m - point_lon_list))

    return lat_index_list, lon_index_list
def find_index_from_lat_lon_2D_arrays(lat_arr, lon_arr, point_lat, point_lon):

    lat_del_arr = lat_arr - point_lat
    lon_del_arr = lon_arr - point_lon

    dist_arr = ( lat_del_arr**2  +  lon_del_arr**2 )**0.5

    return find_min_index_2d_array(dist_arr)
def find_index_from_lat_lon_1D_arrays(lat_arr, lon_arr, point_lat, point_lon):

    lat_del_arr = lat_arr - point_lat
    lon_del_arr = lon_arr - point_lon

    dist_arr = ( lat_del_arr**2  +  lon_del_arr**2 )**0.5

    return find_min_index_1d_array(dist_arr)
def distance_array_lat_lon_2D_arrays_degrees(lat_arr, lon_arr, point_lat, point_lon):
    lat_del_arr = lat_arr - point_lat
    lon_del_arr = lon_arr - point_lon

    return ( lat_del_arr**2  +  lon_del_arr**2 )**0.5
def meter_per_degrees(lat_point):
    lat_mean_rad = np.deg2rad(np.abs(lat_point))

    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_mean_rad) + 1.175 * np.cos(4 * lat_mean_rad)
    m_per_deg_lon = 111132.954 * np.cos(lat_mean_rad)

    return np.abs(m_per_deg_lat), np.abs(m_per_deg_lon)
def degrees_per_meter(lat_point):
    m_per_deg_lat, m_per_deg_lon = meter_per_degrees(lat_point)

    return 1/m_per_deg_lat, 1/m_per_deg_lon
def distance_array_lat_lon_2D_arrays_degress_to_meters(lat_arr, lon_arr, point_lat, point_lon):
    m_per_deg_lat, m_per_deg_lon = meter_per_degrees(np.nanmean(lat_arr))

    lat_del_arr_m = (lat_arr - point_lat) * m_per_deg_lat
    lon_del_arr_m = (lon_arr - point_lon) * m_per_deg_lon

    return ( lat_del_arr_m**2  +  lon_del_arr_m**2 )**0.5
def distance_between_to_points_in_meters(point_1_latlon, point_2_latlon):
    latMid = (point_1_latlon[0] + point_2_latlon[0]) / 2

    m_per_deg_lat, m_per_deg_lon = meter_per_degrees(latMid)

    del_lat = (point_1_latlon[0] - point_2_latlon[0]) * m_per_deg_lat
    del_lon = (point_1_latlon[1] - point_2_latlon[1]) * m_per_deg_lon

    return ((del_lat**2) + (del_lon**2))**0.5


# Data Loading
def numpy_load_txt(filename_, delimiter_=",", format_=float, skip_head=0):
    return genfromtxt(filename_, delimiter=delimiter_, dtype=format_, skip_header=skip_head)
def open_csv_file(filename_, delimiter=',', skip_head=0, dtype='<U32'):
    # load data
    return np.array(genfromtxt(filename_, delimiter=delimiter, dtype=dtype, skip_header=skip_head))
def load_time_columns(filename_):
    ## user defined variables
    day_column_number = 2
    month_column_number = 1
    year_column_number = 0
    hour_column_number = 3
    minute_column_number = 4
    time_header = 'Time' #defining time header

    data_array = open_csv_file(filename_)
    # define arrays
    values_str = data_array[1:,5:]
    values_ = np.zeros((values_str.shape[0],values_str.shape[1]),dtype=float)
    for r_ in range(values_.shape[0]):
        for c_ in range(values_.shape[1]):
            try:
                values_[r_,c_] = float(values_str[r_,c_])
            except:
                values_[r_,c_] = np.nan
    header_ = data_array[0 ,1:]
    # defining time arrays
    time_days = np.zeros(data_array.shape[0] - 1, dtype=float)
    time_month = np.zeros(data_array.shape[0] - 1, dtype=int)
    time_weekday = np.zeros(data_array.shape[0] - 1, dtype=int)
    time_hour = np.zeros(data_array.shape[0] - 1)
    for r_ in range(data_array.shape[0] - 1):
        time_days[r_] = mdates.date2num(datetime.datetime(
            int(float(data_array[r_+1,year_column_number])),
            int(float(data_array[r_+1,month_column_number])),
            int(float(data_array[r_+1,day_column_number])),
            int(float(data_array[r_+1,hour_column_number])),
            int(float(data_array[r_+1,minute_column_number]))))
        time_month[r_] = int(float(data_array[r_+1,month_column_number]))
        time_weekday[r_] = datetime.datetime.weekday(mdates.num2date(time_days[r_]))
        time_hour[r_] = float(data_array[r_+1,hour_column_number]) + (float(data_array[r_+1,minute_column_number]) / 60)
    # compile names
    header_[0] = time_header
    header_[1] = 'Month'
    header_[2] = 'Day of week'
    header_[3] = 'Hour of day'
    # compile values
    values_ = np.column_stack((time_days, time_month, time_weekday, time_hour, values_))

    return header_, values_
def load_object(filename):
    with open(filename, 'rb') as input_object:
        object_ = pickle.load(input_object)
    return object_
def read_one_line_from_text_file(filename_, line_number):
    file_ = open(filename_)
    for i, line in enumerate(file_):
        if i == line_number :
            line_str = line
        elif i > line_number:
            break
    file_.close()
    return line_str

# data saving/output
def save_time_variable_as_csv(output_filename, var_name, time_in_secs, var_values, time_format_output='%Y%m%d%H%M%S'):
    out_file = open(output_filename, 'w')

    # write header
    out_file.write(time_format_output)
    out_file.write(',')
    out_file.write(var_name)
    out_file.write('\n')

    for r_ in range(time_in_secs.shape[0]):
        p_progress_bar(r_, time_in_secs.shape[0])
        out_file.write(time_seconds_to_str(time_in_secs[r_], time_format_output))
        out_file.write(',' + str(var_values[r_]))
        out_file.write('\n')

    out_file.close()
def numpy_save_txt(filename_, array_, delimiter_=",", format_='%s'):
    np.savetxt(filename_, array_, delimiter=delimiter_, fmt=format_)
def save_array_to_disk(header_with_units, time_in_seconds, values_in_floats, filename):
    #
    if len(values_in_floats.shape) == 1:
        header_to_print = ['YYYY', 'MM', 'DD', 'HH', 'mm', header_with_units]
    else:
        header_to_print = ['YYYY', 'MM', 'DD', 'HH', 'mm']
        for parameter_ in header_with_units:
            header_to_print.append(parameter_)
    # create values block
    T_ = time_seconds_to_5C_array(time_in_seconds)
    P_ = np.column_stack((T_, values_in_floats))
    # change type to str
    P_str = np.array(P_, dtype='<U32')
    # join header with values
    P_final = np.row_stack((header_to_print, P_str))
    # save to hard drive
    numpy_save_txt(filename, P_final)
    print('final data saved to: ' + filename)
def save_HVF(header_, values_, filename):
    # check if all shapes match
    if len(header_) != values_.shape[1]:
        print('shape of header is not compatible with shape of values')
        return

    time_in_seconds = mdates.num2epoch(values_[:, 0])

    header_with_units = header_[2:]
    values_in_floats = values_[:, 2:]
    header_to_print = ['YYYY', 'MM', 'DD', 'HH', 'mm']
    for parameter_ in header_with_units:
        header_to_print.append(parameter_)
    # create values block
    T_ = np.zeros((time_in_seconds.shape[0], 5), dtype='<U32')
    for r_ in range(time_in_seconds.shape[0]):
        if time_in_seconds[r_] == time_in_seconds[r_]:
            T_[r_] = time.strftime("%Y,%m,%d,%H,%M", time.gmtime(time_in_seconds[r_])).split(',')
    P_ = np.column_stack((T_, values_in_floats))
    # change type to str
    P_str = np.array(P_, dtype='<U32')
    # join header with values
    P_final = np.row_stack((header_to_print, P_str))
    # save to hard drive
    numpy_save_txt(filename, P_final)
    print('final data saved to: ' + filename)
def save_simple_array_to_disk(header_list, values_array, filename_):
    # change type to str
    values_str = np.array(values_array, dtype='<U32')
    # join header with values
    array_final = np.row_stack((header_list, values_str))
    # save to hard drive
    numpy_save_txt(filename_, array_final)
    print('final data saved to: ' + filename_)
def save_array_as_is(array_, filename_):
    np.savetxt(filename_, array_, delimiter=",", fmt='%s')
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# png data handeling
def store_array_to_png(array_, filename_out):
    """
    This function saves an array to a png file while keeping as much accuracy as possible with the lowest memory ussage
    :param array_: numpy array
    :param filename_out: string with full path
    :return: none
    """

    # shape
    rows_ = array_.shape[0]
    columns_ = array_.shape[1]

    # nan layer
    array_nan =  np.zeros((rows_, columns_), dtype='uint8')
    array_nan[array_ != array_] = 100

    # replace nans
    array_[array_ != array_] = 0

    # convert to all positive
    array_positive = np.abs(array_)

    # sign layer
    array_sign = np.zeros((rows_, columns_), dtype='uint8')
    array_sign[array_ >= 0] = 100

    # zeros array
    array_zeros = np.zeros((rows_, columns_), dtype='uint8')
    array_zeros[array_positive != 0] = 1

    # sub 1 array
    array_sub1 = np.zeros((rows_, columns_), dtype='uint8')
    array_sub1[array_positive<1] = 1
    array_sub1 = array_sub1 * array_zeros

    # power array
    exp_ = np.array(np.log10(array_positive), dtype=int)
    exp_[array_zeros==0] = 0

    # integral array
    array_integral = array_positive / 10 ** np.array(exp_, dtype=float)

    # array_layer_1
    array_layer_1 = np.array(((array_sub1 * 9) + 1) * array_integral * 10, dtype='uint8') + array_sign

    # array_layer_2
    array_layer_2 = np.array(((array_integral * ((array_sub1 * 9) + 1) * 10)
                              - np.array(array_integral * ((array_sub1 * 9) + 1) * 10, dtype='uint8')) * 100,
                             dtype='uint8')
    array_layer_2 = array_layer_2 + array_nan

    # power sign layer
    exp_ = exp_ - array_sub1
    array_power_sign = np.zeros((rows_, columns_), dtype='uint8')
    array_power_sign[exp_ >= 0] = 100

    # array_layer_3
    array_layer_3 = np.abs(exp_) + array_power_sign

    # initialize out array
    out_array = np.zeros((rows_, columns_, 3), dtype='uint8')

    # dump into out array
    out_array[:, :, 0] = array_layer_1
    out_array[:, :, 1] = array_layer_2
    out_array[:, :, 2] = array_layer_3

    img_arr = PIL_Image.fromarray(out_array)
    img_arr.save(filename_out)
def read_png_to_array(filename_):
    """
    This functions converts pngs files created by "store_array_to_png" back to numpy arrays
    :param filename_: string with full path name to png file created by store_array_to_png
    :return: numpy array
    """
    # read image into array
    img_arr = np.array(PIL_Image.open(filename_))

    # shape
    rows_ = img_arr.shape[0]
    columns_ = img_arr.shape[1]

    # nan array
    nan_array = np.zeros((rows_, columns_), dtype='uint8')
    nan_array[img_arr[:,:,1] >= 100] = 1

    # power array
    power_array_magnitude = ((img_arr[:,:,2]/100) - np.array(img_arr[:,:,2]/100, dtype='uint8') ) * 100
    sign_array = np.zeros((rows_, columns_)) - 1
    sign_array[img_arr[:,:,2] >= 100] = 1
    power_array = power_array_magnitude * sign_array

    # sign array
    sign_array = np.array(img_arr[:,:,0]/100, dtype=int)
    sign_array[sign_array == 0] = -1

    # unit array
    unit_array = np.array(img_arr[:,:,0]/10, dtype='uint8') - (np.array(img_arr[:,:,0]/100, dtype='uint8') * 10)

    # decimal array
    decimal_array_1 = (img_arr[:,:,0]/10) - np.array(img_arr[:,:,0]/10, dtype='uint8')
    decimal_array_2 = ((img_arr[:,:,1]/100) - np.array(img_arr[:,:,1]/100, dtype='uint8') ) / 10

    # compute out array
    out_array = (sign_array * (unit_array + decimal_array_1 + decimal_array_2)) * 10 ** power_array

    # flag nans
    out_array[nan_array==1]=np.nan

    return out_array


# sattelite data load
def load_OMI_NO2_monthly_data(filename_):
    # # [molec./cm-2]
    # filename_ = 'C:\\_input\\no2_201601.grd'
    # arr_NO2, lat_arr_NO2, lon_arr_NO2 = load_OMI_NO2_monthly_data(filename_)
    # [440: -820, 1650: 1960]
    data_array = genfromtxt(filename_, dtype=float, skip_header=7)
    file_object = open(filename_,mode='r')
    ncols = int(file_object.readline().split()[-1])
    nrows = int(file_object.readline().split()[-1])
    xllcorner = float(file_object.readline().split()[-1])
    yllcorner = float(file_object.readline().split()[-1])
    cellsize = float(file_object.readline().split()[-1])
    nodata_value = float(file_object.readline().split()[-1])
    # version = file_object.readline().split()[-1]
    file_object.close()

    lat_arr = np.zeros((nrows, ncols), dtype=float)
    lon_arr = np.zeros((nrows, ncols), dtype=float)

    lat_series = np.linspace(yllcorner + (cellsize * nrows), yllcorner, nrows)
    lon_series = np.linspace(xllcorner, xllcorner + (cellsize * ncols), ncols)

    for r_ in range(nrows):
        lon_arr[r_, :] = lon_series

    for c_ in range(ncols):
        lat_arr[:, c_] = lat_series

    data_array[data_array==nodata_value] = np.nan

    data_array = data_array * 1e13

    return data_array[1:-1,:], lat_arr[1:-1,:], lon_arr[1:-1,:]
def load_OMI_HCHO_monthly_data(filename_):
    # # [molec./cm-2]
    # filename_ = 'C:\\_input\\OMIH2CO_Grid_720x1440_201601.dat'
    # arr_HCHO, lat_arr_HCHO, lon_arr_HCHO = load_OMI_HCHO_monthly_data(filename_)
    # [220: -410, 825: 980]
    data_array = genfromtxt(filename_, dtype=float, skip_header=7)
    ncols = 1440
    nrows = 720
    xllcorner = -180
    yllcorner = -90
    cellsize = 0.25

    lat_arr = np.zeros((nrows, ncols), dtype=float)
    lon_arr = np.zeros((nrows, ncols), dtype=float)

    lat_series = np.linspace(yllcorner + (cellsize * nrows), yllcorner, nrows)
    lon_series = np.linspace(xllcorner, xllcorner + (cellsize * ncols), ncols)

    for r_ in range(nrows):
        lon_arr[r_, :] = lon_series

    for c_ in range(ncols):
        lat_arr[:, c_] = lat_series

    data_array = data_array * 1e15

    return data_array[1:-1,:], lat_arr[1:-1,:], lon_arr[1:-1,:]
def download_HIM8_AUS_ch3_500m(YYYYmmddHHMM_str):
    url_ = 'http://dapds00.nci.org.au/thredds/dodsC/rr5/satellite/obs/himawari8/FLDK/' + \
           YYYYmmddHHMM_str[:4] + \
             '/' + \
           YYYYmmddHHMM_str[4:6] + \
             '/' + \
           YYYYmmddHHMM_str[6:8] + \
             '/' + \
           YYYYmmddHHMM_str[8:12] + \
             '/' + \
           YYYYmmddHHMM_str + '00' \
             '-P1S-ABOM_BRF_B03-PRJ_GEOS141_500-HIMAWARI8-AHI.nc'
    f_ = nc.Dataset(url_)

    r_1 = 13194
    r_2 = 19491
    c_1 = 4442
    c_2 = 14076

    return f_.variables['channel_0003_brf'][0, r_1:r_2, c_1:c_2]
def download_HIM8_AUS_2000m(YYYYmmddHHMM_str, channel_number_str, print_=True):
    url_ = 'http://dapds00.nci.org.au/thredds/dodsC/rr5/satellite/obs/himawari8/FLDK/' + \
           YYYYmmddHHMM_str[:4] + '/' + YYYYmmddHHMM_str[4:6] + '/' + YYYYmmddHHMM_str[6:8] + \
           '/' + YYYYmmddHHMM_str[8:12] + \
           '/' + YYYYmmddHHMM_str + '00' + \
           '-P1S-ABOM_OBS_' \
           'B' + channel_number_str + \
           '-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

    if print_: print('downloading HIM_8', YYYYmmddHHMM_str, channel_number_str)

    f_ = nc.Dataset(url_)

    r_1 = 3298
    r_2 = 4873
    c_1 = 1110
    c_2 = 3519

    variable_name = ''
    for var_key in f_.variables.keys():
        if len(var_key.split('channel')) > 1:
            variable_name = var_key
            break


    return f_.variables[variable_name][0, r_1:r_2, c_1:c_2]
def download_HIM8_2000m(YYYYmmddHHMM_str, channel_number_str):
    url_ = 'http://dapds00.nci.org.au/thredds/dodsC/rr5/satellite/obs/himawari8/FLDK/' + \
           YYYYmmddHHMM_str[:4] + '/' + YYYYmmddHHMM_str[4:6] + '/' + YYYYmmddHHMM_str[6:8] + \
           '/' + YYYYmmddHHMM_str[8:12] + \
           '/' + YYYYmmddHHMM_str + '00' + \
           '-P1S-ABOM_OBS_' \
           'B' + channel_number_str + \
           '-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

    f_ = nc.Dataset(url_)

    variable_name = ''
    for var_key in f_.variables.keys():
        if len(var_key.split('channel')) > 1:
            variable_name = var_key
            break

    print('downloading variable:', variable_name)
    return f_.variables[variable_name][0, :,:]
def download_HIM8_AUS_truecolor_2000m(YYYYmmddHHMM_str):
    H8_b = download_HIM8_AUS_2000m(YYYYmmddHHMM_str, '01')
    H8_g = download_HIM8_AUS_2000m(YYYYmmddHHMM_str, '02')
    H8_r = download_HIM8_AUS_2000m(YYYYmmddHHMM_str, '03')
    img_ = np.zeros((H8_b.shape[0], H8_b.shape[1], 3), dtype='uint8')
    img_[:, :, 0] = H8_r * 170
    img_[:, :, 1] = H8_g * 170
    img_[:, :, 2] = H8_b * 170
    return img_
def download_HIM8_truecolor_2000m(YYYYmmddHHMM_str):
    H8_b = download_HIM8_2000m(YYYYmmddHHMM_str, '01')
    H8_g = download_HIM8_2000m(YYYYmmddHHMM_str, '02')
    H8_r = download_HIM8_2000m(YYYYmmddHHMM_str, '03')
    img_ = np.zeros((H8_b.shape[0], H8_b.shape[1], 3), dtype='uint8')
    img_[:, :, 0] = H8_r * 170
    img_[:, :, 1] = H8_g * 170
    img_[:, :, 2] = H8_b * 170
    return img_
def download_lat_lon_arrays_HIM8_500():
    url_ = 'http://dapds00.nci.org.au/thredds/dodsC/rr5/satellite/obs/himawari8/FLDK/ancillary/' \
                     '20150127000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_500-HIMAWARI8-AHI.nc'

    lat_ = download_big_nc_array_in_parts(url_, 'lat')
    lon_ = download_big_nc_array_in_parts(url_, 'lon')

    lat_[lat_ > 360] = np.nan
    lon_[lon_ > 360] = np.nan

    return lat_, lon_
def download_lat_lon_arrays_HIM8_2000():
    url_ = 'http://dapds00.nci.org.au/thredds/dodsC/rr5/satellite/obs/himawari8/FLDK/ancillary/' \
           '20150127000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

    lat_ = download_big_nc_array_in_parts(url_, 'lat')
    lon_ = download_big_nc_array_in_parts(url_, 'lon')

    lat_[lat_ > 360] = np.nan
    lon_[lon_ > 360] = np.nan

    return lat_, lon_
def download_big_nc_array_in_parts(url_, variable_name, parts_=4):
    f_ = nc.Dataset(url_)

    var_shape = f_.variables[variable_name].shape
    print('downloading variable', variable_name, 'with shape:', var_shape)

    if len(var_shape) == 0:
        print('ERROR! variable is not an array')
        return None
    elif len(var_shape) == 1:
        if var_shape[0] == 1:
            print('ERROR! variable is a scalar')
            return None
        else:
            rows_per_part = int(var_shape[0] / parts_)
            if rows_per_part == 0:
                print('ERROR! variable size is too small to be divided, should be downloaded directly')
                return None
            else:
                output_array = np.zeros(var_shape[0])
                for part_ in range(parts_ - 1):
                    output_array[int(part_*rows_per_part):int((part_+1)*rows_per_part)] =\
                        f_.variables[variable_name][int(part_*rows_per_part):int((part_+1)*rows_per_part)]
                output_array[int((parts_ -1)*rows_per_part):] = \
                    f_.variables[variable_name][int((parts_ -1)*rows_per_part):]
                return output_array

    elif len(var_shape) == 2:
        rows_per_part = int(var_shape[1] / parts_)
        if rows_per_part == 0:
            print('ERROR! variable size is too small to be divided, should be downloaded directly')
            return None
        else:
            output_array = np.zeros((var_shape[0],var_shape[1]))
            for part_ in range(parts_ - 1):
                output_array[:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part)] = \
                    f_.variables[variable_name][:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part)]
            output_array[:,int((parts_ - 1) * rows_per_part):] = \
                f_.variables[variable_name][:,int((parts_ - 1) * rows_per_part):]
            return output_array

    elif len(var_shape) == 3:
        rows_per_part = int(var_shape[1] / parts_)
        if rows_per_part == 0:
            print('ERROR! variable size is too small to be divided, should be downloaded directly')
            return None
        else:
            output_array = np.zeros((var_shape[0],var_shape[1],var_shape[2]))
            for part_ in range(parts_ - 1):
                output_array[:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part),:] = \
                    f_.variables[variable_name][:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part),:]
            output_array[:,int((parts_ - 1) * rows_per_part):,:] = \
                f_.variables[variable_name][:,int((parts_ - 1) * rows_per_part):,:]
            return output_array

    elif len(var_shape) == 4:
        rows_per_part = int(var_shape[1] / parts_)
        if rows_per_part == 0:
            print('ERROR! variable size is too small to be divided, should be downloaded directly')
            return None
        else:
            output_array = np.zeros((var_shape[0],var_shape[1],var_shape[2],var_shape[3]))
            for part_ in range(parts_ - 1):
                output_array[:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part),:,:] = \
                    f_.variables[variable_name][:,int(part_ * rows_per_part):int((part_ + 1) * rows_per_part),:,:]
            output_array[:,int((parts_ - 1) * rows_per_part):,:,:] = \
                f_.variables[variable_name][:,int((parts_ - 1) * rows_per_part):,:,:]
            return output_array

    elif len(var_shape) > 4:
        print('ERROR! variable has more than 4 dimensions, not implemented for this many dimentions')
        return None
def get_himawari8_2000m_NCI(YYYYmmddHHMM_str, channel_number, output_format='png',
                            output_path='/g/k10/la6753/data/', row_start=0, row_stop=5500, col_start=0,
                            col_stop=5500):
    """
    gets array from himawari-8 netcdf files and extracts only the indicated channel at the indicated time. saves to output_path
    :param YYYYmmddHHMM_str: string with the time in four digits for year, two digits for months...
    :param channel_number: int or float with the number of the channel ('01'-'16')
    :param output_format: string with either 'png' or 'numpy'. If png the array will be saved used store_array_to_png, otherwise numpy.save will be used
    :param output_path: string with the path, or full filename to be used to save the file
    :param row_start: int with the row number to start the crop
    :param row_stop: int with the row number to stop the crop
    :param col_start: int with the coloumn number to start the crop
    :param col_stop: int with the coloumn number to stop the crop
    :return: None
    """
    channel_number_str = str(int(channel_number)).zfill(2)

    filename_ = '/g/data/rr5/satellite/obs/himawari8/FLDK/' + \
                YYYYmmddHHMM_str[:4] + '/' + YYYYmmddHHMM_str[4:6] + '/' + YYYYmmddHHMM_str[6:8] + \
                '/' + YYYYmmddHHMM_str[8:12] + \
                '/' + YYYYmmddHHMM_str + '00' + \
                '-P1S-ABOM_OBS_' \
                'B' + channel_number_str + \
                '-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

    if os.path.exists(filename_):

        f_ = nc.Dataset(filename_)

        variable_name = ''
        for var_key in f_.variables.keys():
            if len(var_key.split('channel')) > 1:
                variable_name = var_key
                break

        array_ = f_.variables[variable_name][0, row_start:row_stop, col_start:col_stop]

        if output_path[-1] == '/' or output_path[-1] == '\\':
            if output_format == 'png':
                output_filename = output_path + 'him_2000m_ch' + channel_number_str + '_' + YYYYmmddHHMM_str + '.png'
            else:
                output_filename = output_path + 'him_2000m_ch' + channel_number_str + '_' + YYYYmmddHHMM_str + '.npy'
        else:
            output_filename = output_path

        if output_format == 'png':
            store_array_to_png(array_, output_filename)
        else:
            np.save(output_filename, array_)

    else:
        print('File not available for time stamp:', YYYYmmddHHMM_str)


# ERA5
def create_virtual_sondes_from_ERA5(time_stamp_sec, lat_lon_tuple, era5_file_levels_ncFile, era5_file_surface_ncFile,
                                    max_time_delta_sec=21600, show_prints=True):
    close_level_file=False
    close_surface_file=False

    if type(era5_file_levels_ncFile) == str:
        era5_file_levels = nc.Dataset(era5_file_levels_ncFile)
        close_level_file = True
    else:
        era5_file_levels = era5_file_levels_ncFile
    if type(era5_file_surface_ncFile) == str:
        era5_file_surface = nc.Dataset(era5_file_surface_ncFile)
        close_surface_file = True
    else:
        era5_file_surface = era5_file_surface_ncFile

    time_era5_levels_sec = time_era5_to_seconds(era5_file_levels.variables['time'][:])
    time_era5_surface_sec = time_era5_to_seconds(era5_file_surface.variables['time'][:])
    r_era5_levels_1 = time_to_row_sec(time_era5_levels_sec, time_stamp_sec)
    r_era5_surface_1 = time_to_row_sec(time_era5_surface_sec, time_stamp_sec)

    if np.abs(time_era5_levels_sec[r_era5_levels_1] - time_stamp_sec) > max_time_delta_sec:
        if show_prints: print('error time gap is too large', )
        return None

    # find row and column for the lat lon
    lat_index, lon_index = find_index_from_lat_lon(era5_file_levels.variables['latitude'][:].data,
                                                   era5_file_levels.variables['longitude'][:].data,
                                                   lat_lon_tuple[0], lat_lon_tuple[1])


    if show_prints: print('creating input arrays')
    t_profile = kelvin_to_celsius(era5_file_levels.variables['t'][r_era5_levels_1, :, lat_index, lon_index].data)
    if show_prints: print('created t_array')
    td_profile = calculate_dewpoint_from_T_RH(t_profile, era5_file_levels.variables['r'][r_era5_levels_1, :, lat_index, lon_index].data)
    if show_prints: print('created Td_array')
    h_profile = era5_file_levels.variables['z'][r_era5_levels_1, :, lat_index, lon_index].data / gravity_
    if show_prints: print('created z_array')
    u_profile = era5_file_levels.variables['u'][r_era5_levels_1, :, lat_index, lon_index].data
    if show_prints: print('created u_array')
    v_profile = era5_file_levels.variables['v'][r_era5_levels_1, :, lat_index, lon_index].data
    if show_prints: print('created v_array')
    p_profile = era5_file_levels.variables['level'][:].data  # hPa
    if show_prints: print('created p_array')
    surface_p = era5_file_surface.variables['sp'][r_era5_surface_1, lat_index, lon_index] / 100 # / 100 to convert Pa to hPa
    if show_prints: print('created sp_array')



    # trim profiles from surface to top
    # find which levels should be included
    levels_total = 0
    for i_ in range(p_profile.shape[0]):
        if p_profile[i_] > surface_p:
            break
        levels_total += 1

    ####################################### find extrapolations
    surface_t = np.interp(np.log(surface_p), np.log(p_profile), t_profile)
    surface_td = np.interp(np.log(surface_p), np.log(p_profile), td_profile)
    surface_u = np.interp(np.log(surface_p), np.log(p_profile), u_profile)
    surface_v = np.interp(np.log(surface_p), np.log(p_profile), v_profile)
    surface_h = np.interp(np.log(surface_p), np.log(p_profile), h_profile)

    # create temp arrays
    T_array = np.zeros(levels_total + 1, dtype=float)
    Td_array = np.zeros(levels_total + 1, dtype=float)
    Q_array = np.zeros(levels_total + 1, dtype=float)
    U_array = np.zeros(levels_total + 1, dtype=float)
    V_array = np.zeros(levels_total + 1, dtype=float)
    H_array = np.zeros(levels_total + 1, dtype=float)
    P_array = np.zeros(levels_total + 1, dtype=float)

    T_array[:levels_total] = t_profile[:levels_total]
    Td_array[:levels_total] = td_profile[:levels_total]
    U_array[:levels_total] = u_profile[:levels_total]
    V_array[:levels_total] = v_profile[:levels_total]
    H_array[:levels_total] = h_profile[:levels_total]
    P_array[:levels_total] = p_profile[:levels_total]

    T_array[-1] = surface_t
    Td_array[-1] = surface_td
    U_array[-1] = surface_u
    V_array[-1] = surface_v
    H_array[-1] = surface_h
    P_array[-1] = surface_p

    if close_level_file:
        era5_file_levels.close()
    if close_surface_file:
        era5_file_surface.close()

    return P_array, H_array, T_array, Td_array, U_array, V_array
def era5_get_surface_interpolated_vars(era5_file_levels_ncFile, era5_file_surface_ncFile, show_prints=True,
                                       time_start_str_YYYYmmDDHHMM=None, time_stop_str_YYYYmmDDHHMM=None):
    close_level_file=False
    close_surface_file=False

    if type(era5_file_levels_ncFile) == str:
        era5_file_levels = nc.Dataset(era5_file_levels_ncFile)
        close_level_file = True
    else:
        era5_file_levels = era5_file_levels_ncFile
    if type(era5_file_surface_ncFile) == str:
        era5_file_surface = nc.Dataset(era5_file_surface_ncFile)
        close_surface_file = True
    else:
        era5_file_surface = era5_file_surface_ncFile

    time_era5_levels_sec = time_era5_to_seconds(era5_file_levels.variables['time'][:])


    # trim time
    r_1 = 0
    r_2 = -1
    if time_start_str_YYYYmmDDHHMM is not None:
        r_1 = time_to_row_str(time_era5_levels_sec, time_start_str_YYYYmmDDHHMM)
    if time_stop_str_YYYYmmDDHHMM is not None:
        r_2 = time_to_row_str(time_era5_levels_sec, time_stop_str_YYYYmmDDHHMM)

    time_era5_sec = time_era5_levels_sec[r_1:r_2]


    if show_prints: print('creating input arrays')
    t_profile = kelvin_to_celsius(era5_file_levels.variables['t'][r_1:r_2, 10:, :, :].data)
    if show_prints: print('created t_array')
    td_profile = calculate_dewpoint_from_T_RH(t_profile, era5_file_levels.variables['r'][r_1:r_2, 10:, :, :].data)
    if show_prints: print('created Td_array')
    h_profile = era5_file_levels.variables['z'][r_1:r_2, 10:, :, :].data / gravity_
    if show_prints: print('created z_array')
    u_profile = era5_file_levels.variables['u'][r_1:r_2, 10:, :, :].data
    if show_prints: print('created u_array')
    v_profile = era5_file_levels.variables['v'][r_1:r_2, 10:, :, :].data
    if show_prints: print('created v_array')
    p_profile = era5_file_levels.variables['level'][10:].data  # hPa
    if show_prints: print('created p_array')
    surface_p = era5_file_surface.variables['sp'][r_1:r_2, :, :] / 100 # / 100 to convert Pa to hPa
    if show_prints: print('created sp_array')
    q_profile = era5_file_levels.variables['q'][r_1:r_2, 10:, :, :].data
    if show_prints: print('created q_array')



    ####################################### find extrapolations
    surface_t = np.zeros((surface_p.shape), dtype=float)
    surface_td = np.zeros((surface_p.shape), dtype=float)
    surface_u = np.zeros((surface_p.shape), dtype=float)
    surface_v = np.zeros((surface_p.shape), dtype=float)
    surface_h = np.zeros((surface_p.shape), dtype=float)
    surface_q = np.zeros((surface_p.shape), dtype=float)

    if show_prints: print('starting interpolation of every point in time')
    for r_ in range(time_era5_sec.shape[0]):
        p_progress_bar(r_,time_era5_sec.shape[0])
        for lat_ in range(surface_p.shape[1]):
            for lon_ in range(surface_p.shape[2]):

                surface_t [r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), t_profile [r_,:,lat_,lon_])
                surface_td[r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), td_profile[r_,:,lat_,lon_])
                surface_u [r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), u_profile [r_,:,lat_,lon_])
                surface_v [r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), v_profile [r_,:,lat_,lon_])
                surface_h [r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), h_profile [r_,:,lat_,lon_])
                surface_q [r_,lat_,lon_] = np.interp(np.log(surface_p[r_,lat_,lon_]),
                                                     np.log(p_profile), q_profile [r_,:,lat_,lon_])



    if close_level_file:
        era5_file_levels.close()
    if close_surface_file:
        era5_file_surface.close()

    return surface_t, surface_td, surface_u, surface_v, surface_h, surface_q, time_era5_sec

# HYSPLIT
def hysplit_load_freq_endpoints(filename_, number_of_hours):


    file_obj = open(filename_,'r')

    line_list = file_obj.readlines()

    file_obj.close()

    file_traj_list = []
    traj_number = -1
    for line_inx, line_str in enumerate(line_list):
        if line_str == '     1 PRESSURE\n':
            traj_number += 1
            for r_ in range(number_of_hours + 1):
                new_line_list = line_list[line_inx + r_ + 1].split()
                new_line_list.append(traj_number)
                file_traj_list.append(new_line_list)


    arr_ = np.zeros((len(file_traj_list),12), dtype=float)
    for r_ in range(len(file_traj_list)):
        for c_ in range(12):
            arr_[r_,c_] = file_traj_list[r_][c_ + 2]

    return arr_
def hysplit_load_freq_endpoints_all(file_list):

    file_traj_list = []

    for filename_ in file_list:

        file_obj = open(filename_,'r')

        line_list = file_obj.readlines()

        file_obj.close()


        for line_inx, line_str in enumerate(line_list):
            if line_str == '     1 PRESSURE\n':
                for r_ in range(25):
                    file_traj_list.append(line_list[line_inx + r_ + 1].split())

    arr_ = np.zeros((len(file_traj_list),11), dtype=float)
    for r_ in range(len(file_traj_list)):
        for c_ in range(11):
            arr_[r_,c_] = file_traj_list[r_][c_ + 2]

    return arr_
def calculate_mean_time(file_list, lat_tuple, lon_tuple):
    # file_list_irn = sorted(glob.glob(str('E:\\hysplit_IRN\\' + '*.txt')))
    # file_list_uae = sorted(glob.glob(str('E:\\hysplit_UAE\\' + '*.txt')))
    # lat_tuple = tuple((24.889974, 26.201930))
    # lon_tuple = tuple((50.727086, 51.729315))

    hit_counter_list = []
    total_counter_list = []
    # month_list_list = []
    month_mean_time = []
    month_std_time = []

    month_probability_list = []

    for filename_ in file_list:
        arr_ = hysplit_load_freq_endpoints(filename_, 24)
        hit_counter = 0
        hit_age = []

        total_number_of_trajs = int(np.max(arr_[:,-1]))

        for traj_ in range(total_number_of_trajs + 1):
            for r_ in range(arr_.shape[0]):
                if arr_[r_,-1] == traj_:
                    if lat_tuple[0] < arr_[r_, 7] < lat_tuple[1] and lon_tuple[0] < arr_[r_, 8] < lon_tuple[1]:
                        hit_counter += 1
                        hit_age.append(arr_[r_, 6])
                        break



        hit_counter_list.append(hit_counter)
        total_counter_list.append(total_number_of_trajs)

        month_probability_list.append(100*hit_counter/total_number_of_trajs)

        # month_list_list.append(hit_age)
        month_mean_time.append(np.mean(hit_age))
        month_std_time.append(np.std(hit_age))

    return month_probability_list, np.array(month_mean_time), hit_counter_list, total_counter_list, np.array(month_std_time)


# BOM
def Lidar_compile_and_convert_txt_to_dict(main_folder_path):
    # main_folder_path = 'D:\Data\LIDAR Data\\'

    # create the full file list
    filename_list = []
    path_folders_list = next(os.walk(main_folder_path))[1]
    for sub_folder in path_folders_list:
        if sub_folder[0] == '2':
            path_sub_folders_list = next(os.walk(main_folder_path + sub_folder + '\\'))[1]
            for sub_sub_folder in path_sub_folders_list:
                path_sub_sub_sub = main_folder_path + sub_folder + '\\' + sub_sub_folder + '\\'
                ssss_filelist = sorted(glob.glob(str(path_sub_sub_sub + '*.*')))
                for filename_min in ssss_filelist:
                    filename_list.append(filename_min)
    total_files = len(filename_list)
    print(' number of files to compile:', str(total_files))

    # get first file to get shape
    convertion_output = Lidar_convert_txt_to_array(filename_list[0])
    range_shape = convertion_output[1].shape[0]

    # create arrays
    time_array = np.zeros(total_files)
    range_array = convertion_output[1][:,0]
    ch0_pr2 = np.zeros((total_files, range_shape), dtype=float)
    ch0_mrg = np.zeros((total_files, range_shape), dtype=float)
    ch1_pr2 = np.zeros((total_files, range_shape), dtype=float)
    ch1_mrg = np.zeros((total_files, range_shape), dtype=float)
    ch2_pr2 = np.zeros((total_files, range_shape), dtype=float)
    ch2_mrg = np.zeros((total_files, range_shape), dtype=float)
    print('arrays initialized')

    # populate arrays
    for i_, filename_ in enumerate(filename_list):
        p_progress(i_, total_files)
        convertion_output = Lidar_convert_txt_to_array(filename_)
        time_array[i_] = convertion_output[0]
        ch0_pr2[i_, :] = convertion_output[1][:,1]
        ch0_mrg[i_, :] = convertion_output[1][:,2]
        ch1_pr2[i_, :] = convertion_output[1][:,3]
        ch1_mrg[i_, :] = convertion_output[1][:,4]
        ch2_pr2[i_, :] = convertion_output[1][:,5]
        ch2_mrg[i_, :] = convertion_output[1][:,6]

    # move to dict
    output_dict = {}
    output_dict['time'] = time_array
    output_dict['range'] = range_array
    output_dict['ch0_pr2'] = ch0_pr2
    output_dict['ch0_mrg'] = ch0_mrg
    output_dict['ch1_pr2'] = ch1_pr2
    output_dict['ch1_mrg'] = ch1_mrg
    output_dict['ch2_pr2'] = ch2_pr2
    output_dict['ch2_mrg'] = ch2_mrg


    return output_dict
def Lidar_convert_txt_to_array(filename_):
    file_time_str =  filename_[-25:-6]
    time_stamp_seconds = time_str_to_seconds(file_time_str, '%Y-%m-%d_%H-%M-%S')

    # read the data into an array
    data_array_raw = genfromtxt(filename_,dtype=float, delimiter='\t',skip_header=133)

    # only keep one altitude column
    data_array_out = np.zeros((data_array_raw.shape[0], 7), dtype=float)
    data_array_out[:,0] = data_array_raw[:,0]
    data_array_out[:,1] = data_array_raw[:,1]
    data_array_out[:,2] = data_array_raw[:,2]
    data_array_out[:,3] = data_array_raw[:,4]
    data_array_out[:,4] = data_array_raw[:,5]
    data_array_out[:,5] = data_array_raw[:,7]
    data_array_out[:,6] = data_array_raw[:,8]
    return time_stamp_seconds, data_array_out
def compile_AWAP_precip_datafiles(file_list):
    # load first file to get shape
    print('loading file: ', file_list[0])
    arr_1, start_date_sec_1 = load_AWAP_data(file_list[0])
    rows_ = arr_1.shape[0]
    columns_ = arr_1.shape[1]

    # create lat and lon series
    series_lat = np.arange(-44.5, -9.95, 0.05)[::-1]
    series_lon = np.arange(112, 156.29, 0.05)

    # create time array
    output_array_time = np.zeros(len(file_list), dtype=float)

    # create output array
    output_array = np.zeros((len(file_list), rows_, columns_), dtype=float)

    # load first array data into output array
    output_array[0,:,:] = arr_1
    output_array_time[0] = start_date_sec_1

    # loop thru remainning files to populate ouput_array
    for t_, filename_ in enumerate(file_list[1:]):
        print('loading file: ', filename_)
        arr_t, start_date_sec_t = load_AWAP_data(filename_)
        output_array[t_+1, :, :] = arr_t
        output_array_time[t_+1] = start_date_sec_t

    return output_array, output_array_time, series_lat, series_lon
def load_AWAP_data(filename_):
    start_date_str = filename_.split('\\')[-1][:8]
    # stop_date_str = filename_.split('\\')[-1][8:16]
    start_date_sec = time_str_to_seconds(start_date_str, '%Y%m%d')

    arr_precip = np.genfromtxt(filename_, float, skip_header=6, skip_footer=18)

    return arr_precip , start_date_sec
def get_means_from_filelist(file_list, lat_lon_ar):
    # lat_lon_points_list = [ 147.8,
    #                         149,
    #                         -36.8,
    #                         -35.4]

    # box domain indexes
    index_c = [716, 740]
    index_r = [508, 536]

    series_lat = np.arange(-44.5, -9.95, 0.05)[::-1]
    series_lon = np.arange(112,156.3,0.05)

    lat_index_list, lon_index_list = find_index_from_lat_lon(series_lat, series_lon, lat_lon_ar[:,1], lat_lon_ar[:,0])


    time_secs_list = []

    precip_array = np.zeros((277,9),dtype=float)


    for r_, filename_ in enumerate(file_list):
        print('loading file: ', filename_)
        arr_precip, start_date_sec = load_AWAP_data(filename_)
        time_secs_list.append(start_date_sec)

        precip_array[r_, 0] = start_date_sec
        precip_array[r_, 1] = np.mean(arr_precip[index_r[0]:index_r[1]+1, index_c[0]:index_c[1]+1])

        for i_ in range(2,9):
            precip_array[r_, i_] = arr_precip[lat_index_list[i_-2],lon_index_list[i_-2]]

    save_array_to_disk(['box mean precip [mm]','1 precip [mm]','2 precip [mm]','3 precip [mm]',
                        '4 precip [mm]','5 precip [mm]','6 precip [mm]','7 precip [mm]'],
                       precip_array[:,0], precip_array[:,1:], 'C:\\_output\\test_fimi_2.csv')
    # save_HVF(['box','1','2','3','4','5','6','7'], precip_array, 'C:\\_output\\test_fimi_1.csv')

    print("done")

    return precip_array
def compile_BASTA_days_and_save_figure(directory_where_nc_file_are):
    # compile BASTA data per day and save plot (per day)

    time_format_basta = 'seconds since %Y-%m-%d %H:%M:%S'

    # directory_where_nc_file_are = '/home/luis/Data/BASTA/L0/12m5/'
    path_input = directory_where_nc_file_are

    file_label = path_input.split('/')[-4] + '_' + path_input.split('/')[-3] + '_' + path_input.split('/')[-2] + '_'

    file_list_all = sorted(glob.glob(str(path_input + '/*.nc')))

    first_day_str = file_list_all[0][-18:-10]
    last_day_str = file_list_all[-1][-18:-10]

    first_day_int = time_seconds_to_days(time_str_to_seconds(first_day_str,'%Y%m%d'))
    last_day_int = time_seconds_to_days(time_str_to_seconds(last_day_str,'%Y%m%d'))

    total_number_of_days = last_day_int - first_day_int

    print('The data in the folder encompasses', total_number_of_days, 'days')

    days_list_int = np.arange(first_day_int, last_day_int + 1)
    days_list_str = time_seconds_to_str(time_days_to_seconds(days_list_int),'%Y%m%d')

    for day_str in days_list_str:

        print('-|' * 20)
        file_list_day = sorted(glob.glob(str(path_input + file_label + day_str + '*.nc')))

        print('Compiling day',  day_str, len(file_list_day), 'files found for this day.')


        if len(file_list_day) > 0:

            filename_ = file_list_day[0]

            print('loading file:', filename_)

            netcdf_file_object = nc.Dataset(filename_, 'r')

            # variable_names = sorted(netcdf_file_object.variables.keys())

            time_raw = netcdf_file_object.variables['time'][:].copy()
            file_first_time_stamp = time_str_to_seconds(netcdf_file_object.variables['time'].units,
                                                        time_format_basta)

            compiled_time_days = time_seconds_to_days(np.array(time_raw, dtype=int) + file_first_time_stamp)
            compiled_raw_reflectivity_array = netcdf_file_object.variables['raw_reflectivity'][:].copy()
            compiled_range_array = netcdf_file_object.variables['range'][:].copy()

            netcdf_file_object.close()

            if len(file_list_day) > 1:
                for filename_ in file_list_day[1:]:

                    print('loading file:', filename_)

                    netcdf_file_object = nc.Dataset(filename_, 'r')


                    time_raw = netcdf_file_object.variables['time'][:].copy()

                    file_first_time_stamp = time_str_to_seconds(netcdf_file_object.variables['time'].units,
                                                                time_format_basta)

                    time_days = time_seconds_to_days(np.array(time_raw, dtype = int) + file_first_time_stamp)
                    compiled_time_days = np.append(compiled_time_days, time_days)
                    raw_reflectivity_array = netcdf_file_object.variables['raw_reflectivity'][:].copy()
                    compiled_raw_reflectivity_array = np.vstack((compiled_raw_reflectivity_array,
                                                                 raw_reflectivity_array))

                    netcdf_file_object.close()

            figure_output_name = path_input + file_label + day_str + '.png'
            print('saving figure to:', figure_output_name)
            p_arr_vectorized_2(compiled_raw_reflectivity_array, compiled_time_days, compiled_range_array/1000,
                               cmap_=default_cm, figsize_=(12, 8), vmin_=80, vmax_=140,
                               cbar_label='Raw Reflectivity dB', x_header='UTC',y_header='Range AGL [km]',
                               figure_filename=figure_output_name,
                               time_format_ = '%H')
def compile_BASTA_into_one_file(directory_where_nc_file_are):
    # compile BASTA data into one netcdf file

    time_format_basta = 'seconds since %Y-%m-%d %H:%M:%S'

    # directory_where_nc_file_are = '/home/luis/Data/BASTA/L0/12m5/'
    path_input = directory_where_nc_file_are

    file_list_all = sorted(glob.glob(str(path_input + '/*.nc')))

    # first_day_str = file_list_all[0][-18:-10]
    # last_day_str = file_list_all[-1][-18:-10]

    # first_day_int = time_seconds_to_days(time_str_to_seconds(first_day_str,'%Y%m%d'))
    # last_day_int = time_seconds_to_days(time_str_to_seconds(last_day_str,'%Y%m%d'))

    # days_list_int = np.arange(first_day_int, last_day_int + 1)

    # create copy of first file
    netcdf_file_object = nc.Dataset(file_list_all[-1], 'r')
    last_second_raw = netcdf_file_object.variables['time'][:][-1]
    file_first_time_stamp = time_str_to_seconds(netcdf_file_object.variables['time'].units,
                                                time_format_basta)
    netcdf_file_object.close()
    last_second_epoc = last_second_raw + file_first_time_stamp
    last_time_str = time_seconds_to_str(last_second_epoc, '%Y%m%d_%H%M%S')
    output_filename = file_list_all[0][:-3] + '_' + last_time_str + '.nc'
    shutil.copyfile(file_list_all[0], output_filename)
    print('Created output file with name:', output_filename)


    # open output file for appending data
    netcdf_output_file_object = nc.Dataset(output_filename, 'a')
    file_first_time_stamp_seconds_epoc = time_str_to_seconds(netcdf_output_file_object.variables['time'].units,
                                                             time_format_basta)

    variable_names = sorted(netcdf_output_file_object.variables.keys())
    # create references to variables in output file
    variable_objects_dict = {}
    for var_name in variable_names:
        variable_objects_dict[var_name] = netcdf_output_file_object.variables[var_name]


    for filename_ in file_list_all[1:]:

        print('-' * 5)
        print('loading file:', filename_)

        # open file
        netcdf_file_object = nc.Dataset(filename_, 'r')
        # create file's time series
        file_time_stamp_seconds_epoc = time_str_to_seconds(netcdf_file_object.variables['time'].units,
                                                                 time_format_basta)
        time_raw = netcdf_file_object.variables['time'][:].copy()
        time_seconds_epoc = np.array(time_raw, dtype=int) + file_time_stamp_seconds_epoc

        row_start = variable_objects_dict['time'].shape[0]
        row_end = time_raw.shape[0] + row_start

        # append time array
        variable_objects_dict['time'][row_start:row_end] = time_seconds_epoc - file_first_time_stamp_seconds_epoc
        # append raw_reflectivity array
        variable_objects_dict['raw_reflectivity'][row_start:row_end] = \
            netcdf_file_object.variables['raw_reflectivity'][:].copy()
        # append raw_velocity array
        variable_objects_dict['raw_velocity'][row_start:row_end] = \
            netcdf_file_object.variables['raw_velocity'][:].copy()



        # append all other variables that only time dependent
        for var_name in variable_names:
            if var_name != 'time' and var_name != 'range' and \
                    var_name != 'raw_reflectivity' and var_name != 'raw_velocity':
                if len(netcdf_file_object.variables[var_name].shape) == 1:
                    variable_objects_dict[var_name][row_start:row_end] = \
                        netcdf_file_object.variables[var_name][:].copy()


        netcdf_file_object.close()

    netcdf_output_file_object.close()

    print('done')
def load_BASTA_data_from_netcdf_to_arrays(filename_):
    # load BASTA data from netcdf to arrays

    # path_input = '/home/luis/Data/BASTA/L0/'
    # filename_ = path_input + 'BASTA_L0_12m5_20180606_071716_20180806_025422.nc'
    time_format_basta = 'seconds since %Y-%m-%d %H:%M:%S'

    # open file
    netcdf_file_object = nc.Dataset(filename_, 'r')

    # load time as seconds and days
    file_time_stamp_seconds_epoc = time_str_to_seconds(netcdf_file_object.variables['time'].units, time_format_basta)
    time_raw = netcdf_file_object.variables['time'][:].copy()
    time_seconds_epoc = np.array(time_raw, dtype=int) + file_time_stamp_seconds_epoc
    time_days_epoc = time_seconds_to_days(time_seconds_epoc)

    # append range array
    array_range = netcdf_file_object.variables['range'][:].copy()
    # append raw_reflectivity array
    array_raw_reflectivity = netcdf_file_object.variables['raw_reflectivity']#[:].copy()
    # append raw_velocity array
    array_raw_velocity = netcdf_file_object.variables['raw_velocity']#[:].copy()


    # close file
    # netcdf_file_object.close()

    return array_raw_reflectivity, array_raw_velocity, array_range, time_seconds_epoc, time_days_epoc
def BASTA_load_period_to_dict(start_time_YMDHM, stop_time_YMDHM, folder_path,
                                variable_names=('time', 'range', 'raw_reflectivity', 'raw_velocity')):
    time_format_basta = 'seconds since %Y-%m-%d %H:%M:%S'

    out_dict = {}
    temp_dict = {}
    variables_with_time_dimension = []

    if not 'time' in variable_names:
        variable_names_temp_list = ['time']
        for variable_name in variable_names:
            variable_names_temp_list.append(variable_name)
        variable_names = variable_names_temp_list

    # data_folder
    data_folder = folder_path

    # get all data files filenames
    file_list = sorted(glob.glob(str(data_folder + '\\*.nc')))
    file_times_tuple_list = []
    file_times_tuple_list_str = []
    for i_, filename_ in enumerate(file_list):
        file_time_str_start = filename_.split('_')[-2] + filename_.split('_')[-1].split('.')[0]
        file_time_sec_start = time_str_to_seconds(file_time_str_start, '%Y%m%d%H%M%S')
        if i_ < len(file_list) -1:
            file_time_str_stop = file_list[i_+1].split('_')[-2] + file_list[i_+1].split('_')[-1].split('.')[0]
            file_time_sec_stop = time_str_to_seconds(file_time_str_stop, '%Y%m%d%H%M%S')
        else:
            file_time_sec_stop = file_time_sec_start + (24*60*60)
        file_times_tuple_list.append(tuple((file_time_sec_start, file_time_sec_stop)))
        file_times_tuple_list_str.append(tuple((file_time_str_start, time_seconds_to_str(file_time_sec_stop,
                                                                                         '%Y%m%d%H%M%S'))))

    # select only files inside time range
    event_start_sec = time_str_to_seconds(start_time_YMDHM, '%Y%m%d%H%M')
    event_stop_sec = time_str_to_seconds(stop_time_YMDHM, '%Y%m%d%H%M')
    selected_file_list = []
    for file_index in range(len(file_list)):
        if event_start_sec <= file_times_tuple_list[file_index][0] <= event_stop_sec:
            selected_file_list.append(file_list[file_index])
        elif event_start_sec <= file_times_tuple_list[file_index][1] <= event_stop_sec:
            selected_file_list.append(file_list[file_index])
        elif file_times_tuple_list[file_index][0] <= event_start_sec <= file_times_tuple_list[file_index][1]:
            selected_file_list.append(file_list[file_index])
        elif file_times_tuple_list[file_index][0] <= event_stop_sec <= file_times_tuple_list[file_index][1]:
            selected_file_list.append(file_list[file_index])
    print('found files:')
    p_(selected_file_list)

    # load data
    if len(selected_file_list) == 0:
        print('No files inside time range!')
        return out_dict
    else:
        cnt = 0
        for filename_ in selected_file_list:
            if cnt == 0:
                nc_file = nc.Dataset(filename_, 'r')
                print('reading file:',filename_)
                for variable_name in variable_names:
                    if 'time' in nc_file.variables[variable_name].dimensions:
                        variables_with_time_dimension.append(variable_name)
                    if variable_name == 'time':
                        file_time_stamp_seconds_epoc = time_str_to_seconds(nc_file.variables['time'].units,
                                                                           time_format_basta)
                        time_raw = nc_file.variables['time'][:].copy()
                        time_seconds_epoc = np.array(time_raw, dtype=int) + file_time_stamp_seconds_epoc
                        temp_dict[variable_name] = time_seconds_epoc
                    else:
                        temp_dict[variable_name] = nc_file.variables[variable_name][:].filled(np.nan)
                nc_file.close()
                cnt += 1
            else:
                nc_file = nc.Dataset(filename_, 'r')
                print('reading file:', filename_)
                for variable_name in variable_names:
                    if 'time' in nc_file.variables[variable_name].dimensions:
                        variables_with_time_dimension.append(variable_name)
                        if len(nc_file.variables[variable_name].shape) == 1:

                            if variable_name == 'time':
                                file_time_stamp_seconds_epoc = time_str_to_seconds(nc_file.variables['time'].units,
                                                                                   time_format_basta)
                                time_raw = nc_file.variables['time'][:].copy()
                                time_seconds_epoc = np.array(time_raw, dtype=int) + file_time_stamp_seconds_epoc
                                temp_dict[variable_name] = np.hstack((temp_dict[variable_name], time_seconds_epoc))
                            else:
                                temp_dict[variable_name] = np.hstack((temp_dict[variable_name],
                                                                      nc_file.variables[variable_name][:].filled(np.nan)))
                        else:
                            temp_dict[variable_name] = np.vstack((temp_dict[variable_name],
                                                                  nc_file.variables[variable_name][:].filled(np.nan)))
                nc_file.close()

    # find row for start and end of event
    start_row = np.argmin(np.abs(temp_dict['time'] - event_start_sec))
    end_row = np.argmin(np.abs(temp_dict['time'] - event_stop_sec))

    for variable_name in variable_names:
        if variable_name in variables_with_time_dimension:
            out_dict[variable_name] = temp_dict[variable_name][start_row:end_row]
        else:
            out_dict[variable_name] = temp_dict[variable_name]

    return out_dict




def MRR_CFAD(range_array, Ze_array, bins_=(12, np.arange(-10, 40, 2)), normalize_height_wise = True, x_header='dBZe',
             y_header='Height [km]', custom_y_range_tuple=None, custom_x_range_tuple=None, figure_filename=None,
             cbar_label='', cmap_=default_cm, figsize_ = (10,6), title_str = '', contourF_=True, cbar_format='%.2f',
             vmin_=None,vmax_=None, grid_=True, fig_ax=None, show_cbar=True, level_threshold_perc=10,
             invert_y=False, levels=None,custom_ticks_x=None, custom_ticks_y=None, cbar_ax=None):

    if len(range_array.shape) == 1:
        temp_array = np.zeros((Ze_array.shape))
        for r_ in range(Ze_array.shape[0]):
            temp_array[r_,:] = range_array
        range_array = temp_array

    if type(bins_[0]) == int:
        if bins_[0] < 1:
            bins_ = (int(range_array.shape[1] * bins_[0]), bins_[1])

    hist_out = np.histogram2d(range_array.flatten()[~np.isnan(Ze_array.flatten())] / 1000,
                              Ze_array.flatten()[~np.isnan(Ze_array.flatten())],
                              normed=False, bins=bins_)
    hist_array, hist_r, hist_c = hist_out
    hist_r = (hist_r[:-1] + hist_r[1:]) * 0.5
    hist_c = (hist_c[:-1] + hist_c[1:]) * 0.5

    hist_r_2d = np.zeros((hist_array.shape), dtype=float)
    hist_c_2d = np.zeros((hist_array.shape), dtype=float)

    for r_ in range(hist_array.shape[0]):
        for c_ in range(hist_array.shape[1]):
            hist_r_2d[r_, c_] = hist_r[r_]
            hist_c_2d[r_, c_] = hist_c[c_]

    # normalize height wise
    if normalize_height_wise:
        heights_counts = np.sum(hist_array, axis=1)
        maximum_count_at_some_height = np.max(heights_counts)
        cbar_label_final = 'Height normalized frequency'
        for r_ in range(hist_array.shape[0]):
            if heights_counts[r_] < maximum_count_at_some_height * (level_threshold_perc/100):
                hist_array[r_, :] = np.nan
            else:
                 hist_array[r_, :] = hist_array[r_, :] / heights_counts[r_]
    else:
        cbar_label_final = 'Normalized frequency'

    if cbar_label == '': cbar_label = cbar_label_final

    fig_ax = p_arr_vectorized_3(hist_array, hist_c_2d, hist_r_2d, contourF_=contourF_, grid_=grid_,
                                custom_y_range_tuple=custom_y_range_tuple, custom_x_range_tuple=custom_x_range_tuple,
                                x_header=x_header, y_header=y_header, cmap_=cmap_, figsize_=figsize_, cbar_ax=cbar_ax,
                                cbar_label=cbar_label, title_str=title_str, vmin_=vmin_, vmax_=vmax_,levels=levels,
                                figure_filename=figure_filename, fig_ax=fig_ax,show_cbar=show_cbar, invert_y=invert_y,
                                custom_ticks_x=custom_ticks_x, custom_ticks_y=custom_ticks_y,cbar_format=cbar_format)
    return fig_ax, hist_array.T, hist_c[:-1], hist_r[:-1]


# parsivel
def create_DSD_plot(DSD_arr, time_parsivel_seconds, size_arr, events_period_str, figfilename='',
                    output_data=False, x_range=(0, 7.5), y_range=(-1, 3.1), figsize_=(5, 5)):
    size_series = size_arr[0, :]

    event_row_start = time_to_row_str(time_parsivel_seconds, events_period_str.split('_')[0])
    event_row_stop_ = time_to_row_str(time_parsivel_seconds, events_period_str.split('_')[1])

    # normalize
    DSD_arr_over_D = DSD_arr / size_arr

    DSD_arr_over_D_by_D = np.sum(DSD_arr_over_D, axis=1)

    DSD_arr_over_D_by_D_no_zero = DSD_arr_over_D_by_D * 1
    DSD_arr_over_D_by_D_no_zero[DSD_arr_over_D_by_D_no_zero == 0] = np.nan

    DSD_arr_over_D_by_D_log = np.log10(DSD_arr_over_D_by_D_no_zero)

    DSD_arr_over_D_by_D_log_event_1_bin = np.array(DSD_arr_over_D_by_D_log[event_row_start:event_row_stop_])

    DSD_arr_over_D_by_D_log_event_1_bin[~np.isnan(DSD_arr_over_D_by_D_log_event_1_bin)] = 1

    DSD_arr_over_D_by_D_log_event_1_bin_sum = np.nansum(DSD_arr_over_D_by_D_log_event_1_bin, axis=0)

    DSD_arr_over_D_by_D_log_event_1_meanbyD = np.nanmean(np.array(
        DSD_arr_over_D_by_D_log[event_row_start:event_row_stop_]), axis=0)

    DSD_arr_over_D_by_D_log_event_1_meanbyD[DSD_arr_over_D_by_D_log_event_1_bin_sum < 10] = np.nan

    fig, ax = plt.subplots(figsize=figsize_)
    ax.set_title('Mean value of drop concentrations in each diameter bin')
    ax.set_xlabel('D [mm]')
    ax.set_ylabel('log10 N(D) [m-3 mm-1]')
    ax.plot(size_series, DSD_arr_over_D_by_D_log_event_1_meanbyD, '-or', label='Event 1')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid()
    if figfilename != '':
        fig.savefig(figfilename, transparent=True, bbox_inches='tight')
        plt.close(fig)
    if output_data:
        return size_series, DSD_arr_over_D_by_D_log_event_1_meanbyD
def parsivel_nc_format_V2(input_filename, output_filename):
    """
    Transform the not so good nc V1 version produced by save_parsivel_arrays_to_netcdf to V2
    :param input_filename: output from save_parsivel_arrays_to_netcdf
    :param output_filename: a path and filename
    :return:
    """
    # create file
    netcdf_output_file_object = nc.Dataset(output_filename, 'w')
    print('created new file')
    netcdf_first_file_object = nc.Dataset(input_filename)

    # create attributes
    netcdf_output_file_object.setncattr('author', 'Luis Ackermann (ackermannluis@gmail.com')
    netcdf_output_file_object.setncattr('version', 'V2')
    netcdf_output_file_object.setncattr('created', time_seconds_to_str(time.time(), '%Y-%m-%d_%H:%M UTC'))
    print('added attributes')

    # create list for dimensions and variables
    dimension_names_list = sorted(netcdf_first_file_object.dimensions)
    variable_names_list = sorted(netcdf_first_file_object.variables)

    # create dimensions
    for dim_name in dimension_names_list:
        if dim_name == 'time':
            netcdf_output_file_object.createDimension('time', size=0)
            print('time', 'dimension created')
        else:
            netcdf_output_file_object.createDimension(dim_name,
                                                      size=netcdf_first_file_object.dimensions[dim_name].size)
            print(dim_name, 'dimension created')

    # create variables
    # time
    var_name = 'time'
    netcdf_output_file_object.createVariable(var_name, 'int64', (var_name,), zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units',
                                                            'seconds since ' + time_seconds_to_str(0, time_format_mod))
    time_parsivel_seconds = time_str_to_seconds(np.array(netcdf_first_file_object.variables[var_name][:], dtype=str),
                                                time_format_parsivel)
    netcdf_output_file_object.variables[var_name][:] = np.array(time_parsivel_seconds, dtype='int64')
    print('created time variable')

    # time_YmdHM
    var_name = 'YYYYmmddHHMM'
    netcdf_output_file_object.createVariable(var_name, 'str', ('time',), zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'YYYYmmddHHMM in string type')
    netcdf_output_file_object.variables[var_name][:] = np.array(netcdf_first_file_object.variables['time'][:],
                                                                dtype=str)
    print('created time_YmdHM variable')

    # particle_fall_speed
    var_name = 'particles_spectrum'
    if var_name in variable_names_list:
        netcdf_output_file_object.createVariable(var_name,
                                                 netcdf_first_file_object.variables[var_name].dtype,
                                                 netcdf_first_file_object.variables[var_name].dimensions, zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', 'particle counts per bin per minute')
        netcdf_output_file_object.variables[var_name].setncattr('description',
                                                                'for each time stamp, the array varies with respect'
                                                                ' to fall speed on the y axis (rows) starting from the top'
                                                                ' and varies with respect to size on the x axis (columns) '
                                                                'starting from the left')
        netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name][:].copy()
        print('created particles_spectrum variable')

        # particle_fall_speed
        var_name = 'particle_fall_speed'
        netcdf_output_file_object.createVariable(var_name,
                                                 netcdf_first_file_object.variables[var_name].dtype,
                                                 ('particle_fall_speed',), zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', 'm/s')
        netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name][:, 0].copy()
        print('created particle_fall_speed variable')

        # particle_size
        var_name = 'particle_size'
        netcdf_output_file_object.createVariable(var_name,
                                                 netcdf_first_file_object.variables[var_name].dtype,
                                                 ('particle_size',), zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', 'mm')
        netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name][0, :].copy()
        print('created particle_size variable')

    # precipitation_intensity
    var_name = 'precipitation_intensity'
    netcdf_output_file_object.createVariable(var_name,
                                             'float',
                                             netcdf_first_file_object.variables[
                                                 'Intensity of precipitation (mm|h)'].dimensions, zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'mm/h')
    netcdf_output_file_object.variables[var_name][:] = np.array(
        netcdf_first_file_object.variables['Intensity of precipitation (mm|h)'][:], dtype=float)
    print('created precipitation_intensity variable')

    # Weather_code_SYNOP_WaWa
    var_name = 'weather_code_SYNOP_WaWa'
    netcdf_output_file_object.createVariable(var_name,
                                             netcdf_first_file_object.variables['Weather code SYNOP WaWa'].dtype,
                                             netcdf_first_file_object.variables['Weather code SYNOP WaWa'].dimensions,
                                             zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'n/a')
    netcdf_output_file_object.variables[var_name][:] = \
        netcdf_first_file_object.variables['Weather code SYNOP WaWa'][:].copy()

    # Weather_code_SYNOP_WaWa
    var_name = 'weather_code_METAR_SPECI'
    netcdf_output_file_object.createVariable(var_name,
                                             netcdf_first_file_object.variables['Weather code METAR|SPECI'].dtype,
                                             netcdf_first_file_object.variables['Weather code METAR|SPECI'].dimensions,
                                             zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'n/a')
    netcdf_output_file_object.variables[var_name][:] = \
        netcdf_first_file_object.variables['Weather code METAR|SPECI'][:].copy()
    print('created weather_code_METAR_SPECI variable')

    # Weather_code_NWS
    var_name = 'weather_code_NWS'
    netcdf_output_file_object.createVariable(var_name,
                                             netcdf_first_file_object.variables['Weather code NWS'].dtype,
                                             netcdf_first_file_object.variables['Weather code NWS'].dimensions,
                                             zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'n/a')
    NWS_description = '''precip_type_dict = {
        'C': 'No Precip',
        'Kein Niederschlag': 'No Precip',
        'A': 'Hail',
        'L': 'Drizzle',
        'L+': 'heavy Drizzle',
        'L-': 'light Drizzle',
        'R': 'Rain',
        'R+': 'heavy Rain',
        'R-': 'light Rain',
        'RL': 'Drizzle and Rain',
        'RL+': 'heavy Drizzle and Rain',
        'RL-': 'light Drizzle and Rain',
        'RLS': 'Rain, Drizzle and Snow',
        'RLS+': 'heavy Rain, Drizzle and Snow',
        'RLS-': 'light Rain, Drizzle and Snow',
        'S': 'Snow',
        'S+': 'heavy Snow',
        'S-': 'light Snow',
        'SG': 'Snow Grains',
        'SP': 'Freezing Rain'
    }'''
    netcdf_output_file_object.variables[var_name].setncattr('description', NWS_description)
    netcdf_output_file_object.variables[var_name][:] = \
        netcdf_first_file_object.variables['Weather code NWS'][:].copy()
    print('created weather_code_NWS variable')

    # Radar_reflectivity (dBz)
    var_name = 'radar_reflectivity'
    netcdf_output_file_object.createVariable(var_name,
                                             'float',
                                             netcdf_first_file_object.variables['Radar reflectivity (dBz)'].dimensions,
                                             zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'dBz')
    netcdf_output_file_object.variables[var_name][:] = np.array(
        netcdf_first_file_object.variables['Radar reflectivity (dBz)'][:], dtype=float)
    print('created radar_reflectivity variable')

    # particle_count
    var_name = 'particle_count'
    netcdf_output_file_object.createVariable(var_name,
                                             'int64',
                                             netcdf_first_file_object.variables[
                                                 'Number of detected particles'].dimensions, zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', 'counts')
    netcdf_output_file_object.variables[var_name].setncattr('description', 'Number of detected particles per minute')
    netcdf_output_file_object.variables[var_name][:] = np.array(
        netcdf_first_file_object.variables['Number of detected particles'][:], dtype='int64')
    print('created particle_count variable')

    # particle_concentration_spectrum
    var_name = 'particle_concentration_spectrum'
    var_name_old = 'particle_concentration_spectrum_m-3'
    if var_name_old in variable_names_list:
        netcdf_output_file_object.createVariable(var_name,
                                                 netcdf_first_file_object.variables[var_name_old].dtype,
                                                 netcdf_first_file_object.variables[var_name_old].dimensions, zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', '1/m3')
        netcdf_output_file_object.variables[var_name].setncattr('description', 'particles per meter cube per class')
        netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name_old][:].copy()
        print('created particle_concentration_spectrum variable')

        # N_total
        var_name = 'N_total'
        var_name_old = 'particle_concentration_total_m-3'
        netcdf_output_file_object.createVariable(var_name,
                                                 netcdf_first_file_object.variables[var_name_old].dtype,
                                                 netcdf_first_file_object.variables[var_name_old].dimensions, zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', '1/m3')
        netcdf_output_file_object.variables[var_name].setncattr('description', 'total particles per meter cube')
        netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name_old][:].copy()
        print('created N_total variable')

        # psd
        var_name = 'psd'
        var_name_old = 'particle_concentration_spectrum_m-3'
        netcdf_output_file_object.createVariable(var_name,
                                                 'float',
                                                 ('time', 'particle_size',), zlib=True)
        netcdf_output_file_object.variables[var_name].setncattr('units', '1/m3')
        netcdf_output_file_object.variables[var_name].setncattr('description', 'particle size distribution, same as '
                                                                               'particle_concentration_spectrum but all speeds'
                                                                               'bins are summed, only varies with time and size')
        netcdf_output_file_object.variables[var_name][:] = np.sum(netcdf_first_file_object.variables[var_name_old][:],
                                                                  axis=1)
        print('created psd variable')

    # rain mask
    rain_only_list = ['R', 'R+', 'R-']
    RR_ = np.array(netcdf_first_file_object.variables['Intensity of precipitation (mm|h)'][:], dtype=float)
    NWS_ = netcdf_first_file_object.variables['Weather code NWS'][:].copy()
    rain_mask = np.zeros(RR_.shape[0], dtype=int) + 1
    for r_ in range(RR_.shape[0]):
        if RR_[r_] > 0 and NWS_[r_] in rain_only_list:
            rain_mask[r_] = 0
    var_name = 'rain_mask'
    netcdf_output_file_object.createVariable(var_name,
                                             'int',
                                             ('time',), zlib=True)
    netcdf_output_file_object.variables[var_name].setncattr('units', '0 if rain, 1 if not rain')
    netcdf_output_file_object.variables[var_name].setncattr('description', 'using the NWS code, only used R, R+ and R-')
    netcdf_output_file_object.variables[var_name][:] = rain_mask
    print('rain_mask')

    # close all files
    netcdf_output_file_object.close()
    netcdf_first_file_object.close()
def parsivel_sampling_volume(particle_size_2d, particle_fall_speed_2d):
    sampling_area = 0.18 * (0.03 - ((particle_size_2d/1000) / 2)) # m2
    sampling_time = 60 # seconds
    sampling_height = particle_fall_speed_2d * sampling_time # meters

    sampling_volume_2d = sampling_area * sampling_height # m3

    return sampling_volume_2d
def load_parsivel_txt_to_array(filename_, delimiter_=';'):
    # filename_ = 'C:\\_input\\parsivel_2018-07-26-00_2018-08-02-00_1.txt'

    size_scale = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,
                  2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5]
    speed_scale = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,
                   3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]

    speed_array = np.zeros((32,32), dtype=float)
    size_array = np.zeros((32, 32), dtype=float)

    for i in range(32):
        speed_array[:,i] = speed_scale
        size_array[i, :] = size_scale

    # read parsivel file
    spectrum_array_list = []
    data_array_list = []

    with open(filename_) as file_object:
        header_ = file_object.readline().split(delimiter_)
        line_str = file_object.readline()
        line_split = np.array(line_str.split(delimiter_))
        if len(line_split) == 17:
            line_split[16] = '0'
            data_array_list.append(line_split[:-1])
            spectrum_array_list.append(np.zeros((32,32)))
        elif len(line_split) > 17:
            line_split[16] = '0'
            data_array_list.append(line_split[:16])
            line_split[line_split == ''] = '0'
            spectrum_array_list.append(np.array(line_split[16:-1]).reshape((32, 32)))
        elif len(line_split) == 16:
            data_array_list.append(line_split[:-1])
            spectrum_array_list.append(np.zeros((32,32)))

        for line in file_object:
            line_split = np.array(line.split(delimiter_))
            if len(line_split) == 17:
                line_split[16] = '0'
                data_array_list.append(line_split[:-1])
                spectrum_array_list.append(np.zeros((32, 32)))
            elif len(line_split) > 17:
                line_split[16] = '0'
                data_array_list.append(line_split[:16])
                line_split[line_split == ''] = '0'
                spectrum_array_list.append(np.array(line_split[16:-1]).reshape((32, 32)))
            elif len(line_split) == 16:
                if line_split[0] != 'Date':
                    data_array_list.append(line_split[:-1])
                    spectrum_array_list.append(np.zeros((32, 32)))

    data_array = np.stack(data_array_list)
    spectrum_array = np.stack(spectrum_array_list).astype(float)
    t_list = []
    for t_ in range(data_array.shape[0]):
        t_list.append(data_array[t_][0] + '  ' + data_array[t_][1])

    if len(header_) == 16:
        # no spectra was set to record
        return data_array, None, t_list, size_array, speed_array, header_
    else:
        return data_array, spectrum_array, t_list, size_array, speed_array, header_

def save_parsivel_arrays_to_netcdf(raw_spectra_filename, nedcdf_output_filename,
                                   delimiter_=';', raw_time_format='%d.%m.%Y %H:%M:%S'):
    # save_parsivel_arrays_to_netcdf('C:\\_input\\parsivel_2018-07-26-00_2018-08-02-00_1.txt', 'C:\\_input\\parsivel_compiled_3.nc')

    print('reading txt to array')
    data_array, spectrum_array, t_list, size_array, speed_array, header_ = \
        load_parsivel_txt_to_array(raw_spectra_filename, delimiter_=delimiter_)
    print('arrays created')

    file_attributes_tuple_list = [('Compiled by', 'Luis Ackermann @: '  + str(datetime.datetime.now())),
                                  ('Data source', 'Parsivel Disdrometer'),
                                  ('time format', 'YYYYMMDDHHmm in uint64 data type, each ' +
                                                  'time stamp is the acumulated precip for one minute')]

    # time from str to int
    time_array = np.zeros(data_array.shape[0], dtype='<U12')
    # for t_ in range(data_array.shape[0]):
    #     time_array[t_] = int(t_list[t_][6:10] + # YYYY
    #                            t_list[t_][3:5] + # MM
    #                            t_list[t_][:2] + # DD
    #                            t_list[t_][12:14] + # HH
    #                            t_list[t_][15:17]) # mm
    for t_ in range(data_array.shape[0]):
        time_array[t_] = int(time_seconds_to_str(time_str_to_seconds(t_list[t_],raw_time_format),
                                                time_format_parsivel))


    pollutant_attributes_tuple_list = [('units', 'particles per minute')]

    # create output file
    file_object_nc4 = nc.Dataset(nedcdf_output_filename,'w')#,format='NETCDF4_CLASSIC')
    print('output file started')

    # create dimensions
    file_object_nc4.createDimension('particle_fall_speed', speed_array.shape[0])
    file_object_nc4.createDimension('particle_size', size_array.shape[1])
    file_object_nc4.createDimension('time', time_array.shape[0])


    # create dimension variables
    file_object_nc4.createVariable('particle_fall_speed', 'f4', ('particle_fall_speed','particle_size',), zlib=True)
    file_object_nc4.createVariable('particle_size', 'f4', ('particle_fall_speed','particle_size',), zlib=True)
    file_object_nc4.createVariable('time', 'u8', ('time',), zlib=True)


    # populate dimension variables
    file_object_nc4.variables['time'][:] = time_array[:]
    file_object_nc4.variables['particle_fall_speed'][:] = speed_array[:]
    file_object_nc4.variables['particle_size'][:] = size_array[:]


    # create particles_spectrum array
    if spectrum_array is not None:
        file_object_nc4.createVariable('particles_spectrum', 'u2',
                                       ('time', 'particle_fall_speed', 'particle_size',), zlib=True)

        # populate
        file_object_nc4.variables['particles_spectrum'][:] = spectrum_array[:]


        # create particle_concentration_spectrum_m-3
        # get sampling volume
        sampling_volume_2d = parsivel_sampling_volume(size_array, speed_array)
        particle_concentration_spectrum = spectrum_array / sampling_volume_2d
        # create variable
        file_object_nc4.createVariable('particle_concentration_spectrum_m-3', 'float32',
                                       ('time', 'particle_fall_speed', 'particle_size',), zlib=True)
        # populate
        file_object_nc4.variables['particle_concentration_spectrum_m-3'][:] = particle_concentration_spectrum[:]

        # create particle_concentration_total_m-3
        particle_concentration_total = np.nansum(np.nansum(particle_concentration_spectrum, axis=-1), axis=-1)
        # create variable
        file_object_nc4.createVariable('particle_concentration_total_m-3', 'float32',
                                       ('time', ), zlib=True)
        # populate
        file_object_nc4.variables['particle_concentration_total_m-3'][:] = particle_concentration_total[:]

        for attribute_ in pollutant_attributes_tuple_list:
            setattr(file_object_nc4.variables['particles_spectrum'], attribute_[0], attribute_[1])

    # create other data variables
    for i_, head_ in enumerate(header_[:-1]):
        var_name = head_.replace('/','|')
        print('storing var name: ' , var_name)
        temp_ref = file_object_nc4.createVariable(var_name, str, ('time',), zlib=True)
        temp_ref[:] = data_array[:, i_]


    for attribute_ in file_attributes_tuple_list:
        setattr(file_object_nc4, attribute_[0], attribute_[1])



    file_object_nc4.close()

    print('Done!')
def load_parsivel_from_nc(netcdf_filename):
    netcdf_file_object = nc.Dataset(netcdf_filename, 'r')
    file_var_values_dict = {}

    variable_name_list = netcdf_file_object.variables.keys()

    for var_ in variable_name_list:
        file_var_values_dict[var_] = netcdf_file_object.variables[var_][:].copy()

    netcdf_file_object.close()
    return file_var_values_dict, variable_name_list
def parsivel_plot_spectrum_counts(arr_, title_='', x_range_tuple=(0, 6), y_range_tuple=(0, 10), save_filename=None,
                                  contourF=False, bins_=(0,2,5,10,20,50,100,200), fig_size=(5,5)):
    cmap_parsivel = ListedColormap(['white', 'yellow', 'orange', 'lime', 'darkgreen',
                                    'aqua', 'purple', 'navy', 'red'], 'indexed')

    size_scale = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,
                  2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5]
    speed_scale = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,
                   3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]

    speed_array = np.zeros((32,32), dtype=float)
    size_array = np.zeros((32, 32), dtype=float)

    for i in range(32):
        speed_array[:,i] = speed_scale
        size_array[i, :] = size_scale

    spectrum_array_color = np.zeros((arr_.shape[0], arr_.shape[1]), dtype=float)
    bin_labels = []
    i_ = 0
    for i_, bin_ in enumerate(bins_):
        spectrum_array_color[arr_ > bin_] = i_ + 1
        bin_labels.append(str(bin_))
    bin_labels[i_] = '>' + bin_labels[i_]

    fig, ax = plt.subplots(figsize=fig_size)
    if contourF:
        quad1 = ax.contourf(size_array, speed_array, spectrum_array_color, cmap=cmap_parsivel,
                              vmin=0, vmax=8)
    else:
        quad1 = ax.pcolormesh(size_array, speed_array, spectrum_array_color, cmap=cmap_parsivel,
                              vmin=0, vmax=8)

    ax.set_ylim(y_range_tuple)
    ax.set_xlim(x_range_tuple)

    ax.set_xlabel('particle size [mm]')
    ax.set_ylabel('particle speed [m/s]')
    ax.set_title(title_)
    cbar_label = 'Particles per bin'

    cb2 = fig.colorbar(quad1)#, ticks=[0,1,2,3,4,5,6,7])
    ticks_ = np.linspace(0.5, i_+0.5, len(bins_))
    cb2.set_ticks(ticks_)
    cb2.set_ticklabels(bin_labels)
    cb2.ax.set_ylabel(cbar_label)

    if save_filename is None:
        plt.show()
    else:
        fig.savefig(save_filename, transparent=True, bbox_inches='tight')
        plt.close(fig)
    return fig, ax
def parsivel_plot_spectrum_DSD(arr_, title_='', x_range_tuple=(0, 6), y_range_tuple=(0, 10), save_filename=None,
                               contourF=False, fig_size=(5,5), cmap_=default_cm, cbar_label='DSD [m-3]',
                               nozeros_=True, vmin_=None, vmax_=None,):

    size_scale = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,
                  2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5]
    speed_scale = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,
                   3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]

    speed_array = np.zeros((32,32), dtype=float)
    size_array = np.zeros((32, 32), dtype=float)

    for i in range(32):
        speed_array[:,i] = speed_scale
        size_array[i, :] = size_scale


    if nozeros_:
        arr_ = np.array(arr_)
        arr_[arr_ == 0] = np.nan

    fig, ax = plt.subplots(figsize=fig_size)
    if contourF:
        quad1 = ax.contourf(size_array, speed_array, arr_, cmap=cmap_)
    else:
        quad1 = ax.pcolormesh(size_array, speed_array, arr_, cmap=cmap_, vmin=vmin_, vmax=vmax_)

    ax.set_ylim(y_range_tuple)
    ax.set_xlim(x_range_tuple)

    ax.set_xlabel('particle size [mm]')
    ax.set_ylabel('particle speed [m/s]')
    ax.set_title(title_)

    cb2 = fig.colorbar(quad1)
    cb2.ax.set_ylabel(cbar_label)

    if save_filename is None:
        plt.show()
    else:
        fig.savefig(save_filename, transparent=True, bbox_inches='tight')
        plt.close(fig)
    return fig, ax
def calculate_cumulative_precipitation_parsivel(parsivel_precipitation_mm_per_hour, parsivel_time_sec, time_period_str):
    return np.nansum(
        parsivel_precipitation_mm_per_hour[time_to_row_str(parsivel_time_sec, time_period_str.split('_')[0]):
                                           time_to_row_str(parsivel_time_sec, time_period_str.split('_')[1])]) / 60

def calculate_D_m(N_D, D_series):
    D_grad = np.gradient(D_series)
    D_m = np.nansum((N_D * (D_series**4) * D_grad))  /  np.nansum((N_D * (D_series ** 3) * D_grad))
    return D_m
def calculate_LWC(N_D, D_series):
    D_grad = np.gradient(D_series)
    water_density = 1E6 # g/m3
    LWC_ = (np.pi * water_density / 6) *  np.nansum((N_D * (D_series**3) * D_grad))
    return LWC_

# Holographic microscope
def convert_raw_to_array(filename_):
    print('converting file: ' + filename_.split('/')[-1])
    A = np.fromfile(filename_, dtype='uint8')
    evenEl = A[1::2]
    oddEl = A[0::2]
    B = 256 * evenEl + oddEl
    width = 2592
    height = 1944
    I = B.reshape(height, width)
    return I
def create_video_from_filelist(file_list, output_filename, cmap_):
    width = 2592
    height = 1944
    array_3d = np.zeros((len(file_list), height, width), dtype='uint8')
    time_list = []
    for t_, filename_ in enumerate(file_list):
        array_3d[t_,:,:] = convert_raw_to_array(filename_)
        time_list.append(filename_[-21:-4])


    create_video_animation_from_3D_array(array_3d, output_filename, colormap_= cmap_, title_list=time_list,
                                         axes_off=True, show_colorbar=False, interval_=500)


def convert_array_to_png_array(array_):
    # shape
    rows_ = array_.shape[0]
    columns_ = array_.shape[1]

    # nan layer
    array_nan =  np.zeros((rows_, columns_), dtype='uint8')
    array_nan[array_ != array_] = 100

    # replace nans
    array_[array_ != array_] = 0

    # convert to all positive
    array_positive = np.abs(array_)

    # sign layer
    array_sign = np.zeros((rows_, columns_), dtype='uint8')
    array_sign[array_ >= 0] = 100

    # zeros array
    array_zeros = np.zeros((rows_, columns_), dtype='uint8')
    array_zeros[array_positive != 0] = 1

    # sub 1 array
    array_sub1 = np.zeros((rows_, columns_), dtype='uint8')
    array_sub1[array_positive<1] = 1
    array_sub1 = array_sub1 * array_zeros

    # power array
    exp_ = np.array(np.log10(array_positive), dtype=int)
    exp_[array_zeros==0] = 0

    # integral array
    array_integral = array_positive / 10 ** np.array(exp_, dtype=float)

    # array_layer_1
    array_layer_1 = np.array(((array_sub1 * 9) + 1) * array_integral * 10, dtype='uint8') + array_sign

    # array_layer_2
    array_layer_2 = np.array(((array_integral * ((array_sub1 * 9) + 1) * 10)
                              - np.array(array_integral * ((array_sub1 * 9) + 1) * 10, dtype='uint8')) * 100,
                             dtype='uint8')
    array_layer_2 = array_layer_2 + array_nan

    # power sign layer
    exp_ = exp_ - array_sub1
    array_power_sign = np.zeros((rows_, columns_), dtype='uint8')
    array_power_sign[exp_ >= 0] = 100

    # array_layer_3
    array_layer_3 = np.abs(exp_) + array_power_sign

    # initialize out array
    out_array = np.zeros((rows_, columns_, 3), dtype='uint8')

    # dump into out array
    out_array[:, :, 0] = array_layer_1
    out_array[:, :, 1] = array_layer_2
    out_array[:, :, 2] = array_layer_3

    return out_array



# netcdf file handling
def netCDF_crop_timewise(input_filename, time_stamp_start_str_YYYYmmDDHHMM, time_stamp_stop_str_YYYYmmDDHHMM,
                         output_filename=None, vars_to_keep=None, time_dimension_name='time'):
    """
    Creates a copy of an input netCDF4 file with only a subset of the data
    :param input_filename: netCDF4 file with path
    :param time_stamp_start_str_YYYYmmDDHHMMSS: String in YYYYmmDDHHMMSS format
    :param time_stamp_stop_str_YYYYmmDDHHMMSS:
    :param output_filename: filename with path and .nc extension. If none, output file will be in same folder as input
    :param vars_to_keep: list of variable names in str to be kept in output copy. If none, all variables will be copied
    :param time_dimension_name:  name of time dimension
    :return: 0 if good, filename if error
    """
    error_file = 0

    try:

        nc_input_file = nc.Dataset(input_filename)
        time_array = nc_input_file.variables[time_dimension_name][:].copy()
        nc_input_file.close()

        r_1 = time_to_row_str(time_array, time_stamp_start_str_YYYYmmDDHHMM)
        r_2 = time_to_row_str(time_array, time_stamp_stop_str_YYYYmmDDHHMM)

        dict_ = load_netcdf_to_dictionary(input_filename, var_list=vars_to_keep,
                                          time_tuple_start_stop_row=(r_1,r_2), time_dimension_name=time_dimension_name)

        if output_filename is None:
            output_filename = input_filename[:-3] + '_trimmed_' + str(r_1) + '_' + str(r_2) + '.nc'

        save_dictionary_to_netcdf(dict_, output_filename)


    except BaseException as error_msg:
        print(error_msg)
        error_file = input_filename

    return error_file
def add_variable_to_netcdf_file(nc_filename, variables_dict):
    """
    Opens and adds a variable(s) to the file. Will not add new dimensions.
    :param nc_filename: str including path
    :param variables_dict:
    must be a dictionary with keys as variables. inside each variables key should have a dictionary
    inside with variable names as keys
    Each var most have a data key equal to a numpy array (can be masked) and a attribute key
    Each var most have a dimensions key equal to a tuple, in the same order as the array's dimensions
    Each var most have a attributes key equal to a list of tuples with name and description text
    :return: None
    """
    # check if dict_ has the right format

    # create dimension and variables lists
    vars_list = variables_dict.keys()
    for var_ in vars_list:
        if 'dimensions' in variables_dict[var_].keys():
            pass
        else:
            print('dictionary has the wrong format, ' + var_ + 'variable is missing its dimensions')
            return
        if 'attributes' in variables_dict[var_].keys():
            pass
        else:
            print('dictionary has the wrong format, ' + var_ + 'variable is missing its attributes')
            return

    # open file
    file_obj = nc.Dataset(nc_filename,'a')
    print('file openned, do not close this threat or file might be corrupted')

    try:
        # check that variable shapes agree with destination file
        for var_ in vars_list:
            dim_list = list(variables_dict[var_]['dimensions'])
            var_shape = variables_dict[var_]['data'].shape
            for i_, dim_ in enumerate(dim_list):
                if dim_ in sorted(file_obj.dimensions):
                    if var_shape[i_] == file_obj.dimensions[dim_].size:
                        pass
                    else:
                        print('Variable', var_, 'has dimension', dim_,
                              'of different size compared to destination file\nfile closed')
                        file_obj.close()
                        return
                else:
                    print('Variable', var_, 'has dimension', dim_,
                          'which does not exist in destination file\nfile closed')
                    file_obj.close()
                    return

            # create variables
            print('creating', var_, 'variable')
            file_obj.createVariable(var_,
                                    variables_dict[var_]['data'].dtype,
                                    variables_dict[var_]['dimensions'], zlib=True)

            # populate variables
            file_obj.variables[var_][:] = variables_dict[var_]['data']

            for var_attr in variables_dict[var_]['attributes']:
                if var_attr[0] == '_FillValue' or var_attr[0] == 'fill_value':
                    pass
                else:
                    setattr(file_obj.variables[var_], var_attr[0], var_attr[1])

            print('created', var_, 'variable')

    except BaseException as error_msg:
        file_obj.close()
        print('error, file closed\n', error_msg)


    print('All good, closing file')
    file_obj.close()
    print('Done!')
def save_dictionary_to_netcdf(dict_, output_filename):
    """
    Saves a dictionary with the right format to a netcdf file. First dim will be set to unlimited.
    :param dict_: must have a dimensions key, a variables key, and a attributes key.
    dimensions key should have a list of the names of the dimensions
    variables key should have a dictionary inside with variable names as keys
    attributes key should have a list of tuples inside, with the name of the attribute and description in each tuple
    Each var most have a data key equal to a numpy array (can be masked) and a attribute key
    Each var most have a dimensions key equal to a tuple, in the same order as the array's dimensions
    all attributes are tuples with name and description text
    :param output_filename: should include full path and extension
    :return: None
    """
    # check if dict_ has the right format
    if 'variables' in dict_.keys():
        pass
    else:
        print('dictionary has the wrong format, missing variables key')
        return
    if 'dimensions' in dict_.keys():
        pass
    else:
        print('dictionary has the wrong format, missing dimensions key')
        return
    if 'attributes' in dict_.keys():
        pass
    else:
        print('dictionary has the wrong format, missing attributes key')
        return
    # create dimension and variables lists
    vars_list = dict_['variables'].keys()
    dims_list = dict_['dimensions']
    for dim_ in dims_list:
        if dim_ in vars_list:
            pass
        else:
            print('dictionary has the wrong format, ' + dim_ + 'dimension is missing from variables')
    for var_ in vars_list:
        if 'dimensions' in dict_['variables'][var_].keys():
            pass
        else:
            print('dictionary has the wrong format, ' + var_ + 'variable is missing its dimensions')
            return
        if 'attributes' in dict_['variables'][var_].keys():
            pass
        else:
            print('dictionary has the wrong format, ' + var_ + 'variable is missing its attributes')
            return

    # create output file
    file_obj = nc.Dataset(output_filename,'w')#,format='NETCDF4_CLASSIC')
    print('output file started')

    # populate file's attributes
    for attribute_ in dict_['attributes']:
        setattr(file_obj, attribute_[0], attribute_[1])


    # create dimensions
    for i_, dim_ in enumerate(dims_list):
        if i_ == 0:
            file_obj.createDimension(dim_, size=0)
        else:
            shape_index = np.argwhere(np.array(dict_['variables'][dim_]['dimensions']) == dim_)[0][0]
            file_obj.createDimension(dim_, dict_['variables'][dim_]['data'].shape[shape_index])
    print('dimensions created')


    # create variables
    for var_ in vars_list:
        print('creating', var_, 'variable')
        file_obj.createVariable(var_,
                                dict_['variables'][var_]['data'].dtype,
                                dict_['variables'][var_]['dimensions'], zlib=True)

        # populate variables
        file_obj.variables[var_][:] = dict_['variables'][var_]['data']



        for var_attr in dict_['variables'][var_]['attributes']:
            if isinstance(var_attr, str):
                setattr(file_obj.variables[var_], dict_['variables'][var_]['attributes'][0],
                        dict_['variables'][var_]['attributes'][1])
                break
            else:
                if var_attr[0] == '_FillValue' or var_attr[0] == 'fill_value':
                    pass
                else:
                    setattr(file_obj.variables[var_], var_attr[0], var_attr[1])
        print('created', var_, 'variable')

    print('storing data to disk and closing file')
    file_obj.close()
    print('Done!')
def load_netcdf_to_dictionary(filename_, var_list=None, time_tuple_start_stop_row=None, time_dimension_name='time'):
    """
    creates a dictionary from a netcdf file, with the following format
    :param filename_: filename with path of a netCDF4 file
    :param var_list: list of variables to be loaded, if none, all variables will be loaded
    :param time_tuple_start_stop_str: tuple with two time rows, time dimension will be trimmed r_1:r_2
    :param time_dimension_name:  name of time dimension
    :return: dict_: have a dimensions key, a variables key, and a attributes key.
    Each var have a data key equal to a numpy array (can be masked) and a attribute key
    Each var have a dimensions key equal to a tuple, in the same order as the array's dimensions
    all attributes are tuples with name and description text
    """
    # create output dict
    out_dict = {}

    # open file
    file_obj = nc.Dataset(filename_, 'r')  # ,format='NETCDF4_CLASSIC')
    print('output file started')

    # get file's attr
    file_att_list_tuple = []
    for attr_ in file_obj.ncattrs():
        file_att_list_tuple.append((attr_, file_obj.getncattr(attr_)))
    out_dict['attributes'] = file_att_list_tuple

    # get dimensions
    out_dict['dimensions'] = sorted(file_obj.dimensions)

    # get variables
    if var_list is None:
        var_list = sorted(file_obj.variables)
    out_dict['variables'] = {}

    # create variables
    for var_ in var_list:
        out_dict['variables'][var_] = {}


        if time_tuple_start_stop_row is not None:
            if time_dimension_name in file_obj.variables[var_].dimensions:
                out_dict['variables'][var_]['data'] = file_obj.variables[var_][time_tuple_start_stop_row[0]:
                                                                               time_tuple_start_stop_row[1]]
            else:
                out_dict['variables'][var_]['data'] = file_obj.variables[var_][:]
        else:
            out_dict['variables'][var_]['data'] = file_obj.variables[var_][:]

        out_dict['variables'][var_]['attributes'] = file_obj.variables[var_].ncattrs()
        var_att_list_tuple = []
        for attr_ in file_obj.variables[var_].ncattrs():
            var_att_list_tuple.append((attr_, file_obj.variables[var_].getncattr(attr_)))
        out_dict['variables'][var_]['attributes'] = var_att_list_tuple

        out_dict['variables'][var_]['dimensions'] = file_obj.variables[var_].dimensions

        print('read variable', var_)
    file_obj.close()
    print('Done!')

    return out_dict
def merge_multiple_netCDF_by_time_dimension(directory_where_nc_file_are_in_chronological_order, output_path='',
                                            output_filename=None, time_variable_name='time', time_dimension_name=None,
                                            vars_to_keep=None, nonTimeVars_check_list=None,
                                            key_search_str='', seek_in_subfolders=False, force_file_list=None):

    if force_file_list is not None:
        file_list_all = sorted(force_file_list)
    else:
        if seek_in_subfolders:
            if key_search_str == '':
                file_list_all = sorted(list_files_recursive(directory_where_nc_file_are_in_chronological_order))
            else:
                file_list_all = sorted(list_files_recursive(directory_where_nc_file_are_in_chronological_order,
                                                            filter_str=key_search_str))

        else:
            file_list_all = sorted(glob.glob(str(directory_where_nc_file_are_in_chronological_order
                                                 + '*' + key_search_str + '*.nc')))

    print('Files to be merged (in this order):')
    parameter_list = ''
    for i, parameter_ in enumerate(file_list_all):
        parameter_list = str(parameter_list) + str(i) + " ---> " + str(parameter_) + '\n'
    print(parameter_list)

    # create copy of first file
    if output_filename is None:
        if output_path == '':
            output_filename = file_list_all[0][:-3] + '_merged.nc'
        else:
            output_filename = output_path + file_list_all[0].split('\\')[-1][:-3] + '_merged.nc'


    # define time variable and dimension
    if time_dimension_name is None:
        time_dimension_name = time_variable_name

    # check if time dimension is unlimited
    netcdf_first_file_object = nc.Dataset(file_list_all[0], 'r')

    if netcdf_first_file_object.dimensions[time_dimension_name].size == 0 and vars_to_keep is None:
        # all good, just make copy of file with output_filename name
        netcdf_first_file_object.close()
        shutil.copyfile(file_list_all[0], output_filename)
        print('first file in merger list has unlimited time dimension, copy created with name:', output_filename)
    else:
        # not so good, create new file and copy everything from first, make time dimension unlimited...
        netcdf_output_file_object = nc.Dataset(output_filename, 'w')
        print('first file in merger list does not have unlimited time dimension, new file created with name:',
              output_filename)

        # copy main attributes
        attr_list = netcdf_first_file_object.ncattrs()
        for attr_ in attr_list:
            netcdf_output_file_object.setncattr(attr_, netcdf_first_file_object.getncattr(attr_))
        print('main attributes copied')

        # create list for dimensions and variables
        dimension_names_list = sorted(netcdf_first_file_object.dimensions)
        if vars_to_keep is None:
            variable_names_list = sorted(netcdf_first_file_object.variables)
        else:
            variable_names_list = vars_to_keep

        # create dimensions
        for dim_name in dimension_names_list:
            if dim_name == time_dimension_name:
                netcdf_output_file_object.createDimension(time_dimension_name, size=0)
                print(time_variable_name, 'dimension created')
            else:
                netcdf_output_file_object.createDimension(dim_name,
                                                         size=netcdf_first_file_object.dimensions[dim_name].size)
                print(dim_name, 'dimension created')

        # create variables
        for var_name in variable_names_list:
            # create
            netcdf_output_file_object.createVariable(var_name,
                                                     netcdf_first_file_object.variables[var_name].dtype,
                                                     netcdf_first_file_object.variables[var_name].dimensions, zlib=True)
            print(var_name, 'variable created')

            # copy the attributes
            attr_list = netcdf_first_file_object.variables[var_name].ncattrs()
            for attr_ in attr_list:
                netcdf_output_file_object.variables[var_name].setncattr(attr_,
                                                                        netcdf_first_file_object.variables[
                                                                            var_name].getncattr(attr_))
            print('variable attributes copied')

            # copy the data to the new file
            netcdf_output_file_object.variables[var_name][:] = netcdf_first_file_object.variables[var_name][:].copy()
            print('variable data copied')

            print('-=' * 20)


        # close all files
        netcdf_output_file_object.close()
        netcdf_first_file_object.close()


    print('starting to copy other files into merged file')
    vars_list = variable_names_list

    for filename_ in file_list_all[1:]:
        # open output file for appending data
        netcdf_output_file_object = nc.Dataset(output_filename, 'a')

        print('-' * 5)
        print('loading file:', filename_)

        # open hourly file
        netcdf_file_object = nc.Dataset(filename_, 'r')
        # get time array
        time_hourly = np.array(netcdf_file_object.variables[time_variable_name][:], dtype=float)

        row_start = netcdf_output_file_object.variables[time_variable_name].shape[0]
        row_end = time_hourly.shape[0] + row_start

        # append time array
        netcdf_output_file_object.variables[time_variable_name][row_start:row_end] = time_hourly

        # append all other variables that only time dependent
        for var_name in vars_list:
            if var_name != time_variable_name:
                if time_dimension_name in netcdf_output_file_object.variables[var_name].dimensions:
                    netcdf_output_file_object.variables[var_name][row_start:row_end] = \
                        netcdf_file_object.variables[var_name][:].copy()

        # check non time dependent variables for consistency
        vars_list_sub = sorted(netcdf_file_object.variables)
        if vars_list_sub != sorted(netcdf_first_file_object.variables):
            print('Alert! Variables in first file are different than other files')
            print('first file variables:')
            p_(sorted(netcdf_first_file_object.variables))
            print(filename_, 'file variables:')
            p_(vars_list_sub)

        if nonTimeVars_check_list is not None:
            for var_name in nonTimeVars_check_list:
                if np.nansum(np.abs(netcdf_file_object.variables[var_name][:].copy() -
                                    netcdf_output_file_object.variables[var_name][:].copy())) != 0:
                    print('Alert!', var_name, 'from file:', filename_, 'does not match the first file')

                    # copy the attributes
                    netcdf_output_file_object.variables[var_name].setncattr(
                        'values from file ' + filename_, netcdf_file_object.variables[var_name][:].copy()
                    )

        netcdf_file_object.close()

        netcdf_output_file_object.close()

    print('done')
def load_netcdf_file_variable(filename_, variable_name_list=None):
    netcdf_file_object = nc.Dataset(filename_, 'r')

    file_attributes_dict = {}
    file_var_values_dict = {}
    file_var_attrib_dict = {}
    file_dim_dict = {}

    if variable_name_list is None: variable_name_list = list(netcdf_file_object.variables)

    for atr_ in netcdf_file_object._attributes:
        file_attributes_dict[atr_] = netcdf_file_object._attributes[atr_]

    for dim_ in netcdf_file_object.dimensions:
        file_dim_dict[dim_] = netcdf_file_object.dimensions[dim_]

    for var_ in variable_name_list:
        file_var_values_dict[var_] = netcdf_file_object.variables[var_][:].copy()
        for atr_ in netcdf_file_object.variables[var_]._attributes:
            file_var_attrib_dict[var_] = netcdf_file_object.variables[var_]._attributes[atr_]

    netcdf_file_object.close()

    return file_attributes_dict, file_var_values_dict, file_var_attrib_dict, file_dim_dict
def save_array_list_as_netcdf(array_list, name_list, units_list, attributes_list, out_filename):
    file_object = nc.Dataset(out_filename, 'w')
    # file_object.history = 'Created for a test'

    for variable_ in range(len(array_list)):
        dim_list_name = []
        for dim_ in range(len(array_list[variable_].shape)):
            dim_name = str(variable_) + '_' + str(dim_)
            dim_list_name.append(dim_name)
            file_object.createDimension(dim_name, array_list[variable_].shape[dim_])

        dtype_ = str(array_list[variable_].dtype)[0]

        file_object.createVariable( name_list[variable_], dtype_, tuple(dim_list_name) )



        setattr(file_object.variables[name_list[variable_]], 'units',units_list[variable_])

        file_object.variables[name_list[variable_]] = array_list[variable_]
        # temp_variable_handle[:] = array_list[variable_][:]


    for atri_ in attributes_list:
        setattr(file_object, atri_[0], atri_[1])


    file_object.close()
def save_time_series_as_netcdf(array_list, name_list, units_list, attributes_list, out_filename):
    file_object = nc.Dataset(out_filename, 'w')

    # create time dimension
    file_object.createDimension('time', array_list[0].shape[0])

    for variable_ in range(len(array_list)):
        dtype_ = str(array_list[variable_].dtype)[0]
        if dtype_ == '<': dtype_ = 'S1'
        file_object.createVariable(name_list[variable_], dtype_, ('time',))

        setattr(file_object.variables[name_list[variable_]], 'units',units_list[variable_])

        file_object.variables[name_list[variable_]][:] = array_list[variable_][:]
        # temp_variable_handle[:] = array_list[variable_][:]


    for atri_ in attributes_list:
        setattr(file_object, atri_[0], atri_[1])


    file_object.close()
def save_emissions_to_new_netcdf(out_filename, emissions_array, pollutant_name, time_array, lat_array, lon_array,
                                 file_attributes_tuple_list, pollutant_attributes_tuple_list):
    file_object = nc.Dataset(out_filename, 'w')

    # create dimensions
    file_object.createDimension('lat', lat_array.shape[0])
    file_object.createDimension('lon', lon_array.shape[0])
    file_object.createDimension('time', time_array.shape[0])

    # create dimension variables
    file_object.createVariable('time', str(time_array.dtype)[0], ('time', ))
    file_object.createVariable('lat', str(lat_array.dtype)[0], ('lat',))
    file_object.createVariable('lon', str(lon_array.dtype)[0], ('lon',))

    # populate dimension variables
    file_object.variables['time'][:] = time_array[:]
    file_object.variables['lat'][:] = lat_array[:]
    file_object.variables['lon'][:] = lon_array[:]

    # create emission array
    file_object.createVariable(pollutant_name, str(emissions_array.dtype)[0], ('time', 'lat', 'lon',))
    # populate
    file_object.variables[pollutant_name][:] = emissions_array[:]

    for attribute_ in file_attributes_tuple_list:
        setattr(file_object, attribute_[0], attribute_[1])

    for attribute_ in pollutant_attributes_tuple_list:
        setattr(file_object.variables[pollutant_name], attribute_[0], attribute_[1])

    file_object.close()
def save_emissions_to_existing_netcdf(out_filename, emissions_array, pollutant_name, attributes_tuple_list):
    file_object = nc.Dataset(out_filename, 'a')

    file_object.createVariable(pollutant_name, str(emissions_array.dtype)[0], ('time', 'lat', 'lon',))
    file_object.variables[pollutant_name][:] = emissions_array[:]

    setattr(file_object.variables[pollutant_name], 'pollutant name', pollutant_name)
    for attribute_ in attributes_tuple_list:
        setattr(file_object.variables[pollutant_name], attribute_[0], attribute_[1])


    file_object.close()
def WRF_emission_file_modify(filename_, variable_name, cell_index_west_east, cell_index_south_north, new_value):
    netcdf_file_object = nc.Dataset(filename_, 'a')

    current_array = netcdf_file_object.variables[variable_name][0,0,:,:].copy()

    current_value = current_array[cell_index_south_north, cell_index_west_east]
    print(current_value)

    current_array[cell_index_south_north, cell_index_west_east] = new_value

    netcdf_file_object.variables[variable_name][0,0,:,:] = current_array[:,:]

    netcdf_file_object.close()
def find_wrf_3d_cell_from_latlon_to_south_north_west_east(lat_, lon_, wrf_output_filename,
                                                          wrf_lat_variablename='XLAT', wrf_lon_variablename='XLONG',
                                                          flatten_=False):

    netcdf_file_object_wrf = nc.Dataset(wrf_output_filename, 'r')
    wrf_lat_array = netcdf_file_object_wrf.variables[wrf_lat_variablename][:,:].copy()
    wrf_lon_array = netcdf_file_object_wrf.variables[wrf_lon_variablename][:,:].copy()
    netcdf_file_object_wrf.close()

    wrf_abs_distance = ( (np.abs(wrf_lat_array - lat_)**2) + (np.abs(wrf_lon_array - lon_)**2) )**0.5

    if flatten_:
        return np.argmin(wrf_abs_distance)
    else:
        return np.unravel_index(np.argmin(wrf_abs_distance), wrf_abs_distance.shape)


# specialized tools
def vectorize_array(array_):
    output_array = np.zeros((array_.shape[0] * array_.shape[1], 3), dtype=float)
    for r_ in range(array_.shape[0]):
        for c_ in range(array_.shape[1]):
            output_array[r_,0] = r_
            output_array[r_, 1] = c_
            output_array[r_, 2] = array_[r_,c_]
    return output_array
def exceedance_rolling(arr_time_seconds, arr_values, standard_, rolling_period, return_rolling_arrays=False):
    ## assumes data is in minutes and in same units as standard
    time_secs_1h, values_mean_disc_1h = mean_discrete(arr_time_seconds, arr_values, 3600, arr_time_seconds[0], min_data=45)
    values_rolling_mean = row_average_rolling(values_mean_disc_1h, rolling_period)
    counter_array = np.zeros(values_rolling_mean.shape[0])
    counter_array[values_rolling_mean > standard_] = 1
    total_number_of_exceedances = np.sum(counter_array)

    #create date str array
    T_ = np.zeros((time_secs_1h.shape[0],5),dtype='<U32')
    for r_ in range(time_secs_1h.shape[0]):
        if time_secs_1h[r_] == time_secs_1h[r_]:
            T_[r_] = time.strftime("%Y_%m_%d",time.gmtime(time_secs_1h[r_])).split(',')

    exceedance_date_list = []
    for r_, rolling_stamp in enumerate(values_rolling_mean):
        if rolling_stamp > standard_:
            exceedance_date_list.append(T_[r_])
    exc_dates_array = np.array(exceedance_date_list)
    exc_dates_array_unique = np.unique(exc_dates_array)

    if return_rolling_arrays:
        return total_number_of_exceedances, exc_dates_array_unique, time_secs_1h, values_rolling_mean
    else:
        return total_number_of_exceedances, exc_dates_array_unique



# ozonesonde and radiosonde related
def load_sonde_data(filename_, mode_='PBL'):  ##Loads data and finds inversions, creates I_
    # global V_, M_, H_, ASL_, time_header, I_, I_line
    # global ASL_avr, L_T, L_RH, time_string, time_days, time_seconds, year_, flight_name


    ## user defined variables
    delimiter_ = ','
    error_flag = -999999
    first_data_header = 'Day_[GMT]'
    day_column_number = 0
    month_column_number = 1
    year_column_number = 2
    hour_column_number = 3
    minute_column_number = 4
    second_column_number = 5
    # time_header = 'Local Time'  # defining time header

    # main data array
    sample_data = filename_

    # look for data start (header size)
    with open(sample_data) as file_read:
        header_size = -1
        r_ = 0
        for line_string in file_read:
            if (len(line_string) >= len(first_data_header) and
                        line_string[:len(first_data_header)] == first_data_header):
                header_size = r_
                break
            r_ += 1
        if header_size == -1:
            print('no data found!')
            sys.exit()

    data_array = np.array(genfromtxt(sample_data,
                                     delimiter=delimiter_,
                                     skip_header=header_size,
                                     dtype='<U32'))
    # defining header  and data arrays
    M_ = data_array[1:, 6:].astype(float)
    H_ = data_array[0, 6:]
    ASL_ = M_[:, -1]
    # year_ = data_array[1, year_column_number]
    ASL_[ASL_ == error_flag] = np.nan

    # defining time arrays
    time_str = data_array[1:, 0].astype('<U32')
    for r_ in range(time_str.shape[0]):
        time_str[r_] = (str(data_array[r_ + 1, day_column_number]) + '-' +
                           str(data_array[r_ + 1, month_column_number]) + '-' +
                           str(data_array[r_ + 1, year_column_number]) + '_' +
                           str(data_array[r_ + 1, hour_column_number]) + ':' +
                           str(data_array[r_ + 1, minute_column_number]) + ':' +
                           str(data_array[r_ + 1, second_column_number]))

    time_days = np.array([mdates.date2num(datetime.datetime.utcfromtimestamp(
                                calendar.timegm(time.strptime(time_string_record, '%d-%m-%Y_%H:%M:%S'))))
                                for time_string_record in time_str])
    time_seconds = time_days_to_seconds(time_days)
    V_ = M_.astype(float)
    V_[V_ == error_flag] = np.nan

    T_avr = np.ones(V_[:, 1].shape)
    RH_avr = np.ones(V_[:, 1].shape)
    ASL_avr = np.ones(V_[:, 1].shape)
    L_T = np.zeros(V_[:, 1].shape)
    L_RH = np.zeros(V_[:, 1].shape)
    I_ = np.zeros(V_[:, 1].shape)
    I_[:] = np.nan
    # rolling average of T RH and ASL
    mean_size = 7  # 5
    for r_ in range(mean_size, V_[:, 1].shape[0] - mean_size):
        T_avr[r_] = np.nanmean(V_[r_ - mean_size: r_ + mean_size, 1])
        RH_avr[r_] = np.nanmean(V_[r_ - mean_size: r_ + mean_size, 2])
        ASL_avr[r_] = np.nanmean(ASL_[r_ - mean_size: r_ + mean_size])
    for r_ in range(mean_size, V_[:, 1].shape[0] - mean_size):
        if (ASL_avr[r_ + 1] - ASL_avr[r_]) > 0:
            L_T[r_] = ((T_avr[r_ + 1] - T_avr[r_]) /
                       (ASL_avr[r_ + 1] - ASL_avr[r_]))
            L_RH[r_] = ((RH_avr[r_ + 1] - RH_avr[r_]) /
                        (ASL_avr[r_ + 1] - ASL_avr[r_]))

    # define location of inversion
    # PBL or TSI
    if mode_ == 'PBL':
        for r_ in range(mean_size, V_[:, 1].shape[0] - mean_size):
            if L_T[r_] > 7 and L_RH[r_] < -20:  # PBL = 7,20 / TSI = 20,200
                I_[r_] = 1

        # get one of I_ only per layer
        temperature_gap = .4  # kilometres
        I_line = np.zeros((1, 3))  # height, time, intensity
        if np.nansum(I_) > 1:
            r_ = -1
            while r_ < I_.shape[0] - mean_size:
                r_ += 1
                if I_[r_] == 1 and ASL_avr[r_] < 4:
                    layer_temp = T_avr[r_]
                    layer_h = ASL_avr[r_]
                    layer_time = time_seconds[r_]
                    for rr_ in range(r_, I_.shape[0] - mean_size):
                        if T_avr[rr_] < layer_temp - temperature_gap:
                            delta_h = ASL_avr[rr_] - layer_h
                            altitude_ = layer_h
                            stanking_temp = np.array([altitude_, layer_time, delta_h])
                            I_line = np.row_stack((I_line, stanking_temp))
                            r_ = rr_
                            break

        if np.max(I_line[:, 0]) != 0:
            I_line = I_line[1:, :]
        else:
            I_line[:, :] = np.nan
    else:
        for r_ in range(mean_size, V_[:, 1].shape[0] - mean_size):
            if L_T[r_] > 20 and L_RH[r_] < -200:  # PBL = 7,20 / TSI = 20,200
                I_[r_] = 1

        # get one of I_ only per layer
        temperature_gap = .4  # kilometres
        I_line = np.zeros((1, 3))  # height, time, intensity
        if np.nansum(I_) > 1:
            r_ = -1
            while r_ < I_.shape[0] - mean_size:
                r_ += 1
                if I_[r_] == 1 and 4 < ASL_avr[r_] < 8:
                    layer_temp = T_avr[r_]
                    layer_h = ASL_avr[r_]
                    layer_time = time_seconds[r_]
                    for rr_ in range(r_, I_.shape[0] - mean_size):
                        if T_avr[rr_] < layer_temp - temperature_gap:
                            delta_h = ASL_avr[rr_] - layer_h
                            altitude_ = layer_h
                            stanking_temp = np.array([altitude_, layer_time, delta_h])
                            I_line = np.row_stack((I_line, stanking_temp))
                            r_ = rr_
                            break

        if np.max(I_line[:, 0]) != 0:
            I_line = I_line[1:, :]
        else:
            I_line[:, :] = np.nan

    return H_, V_, time_days, time_seconds, I_, I_line, L_T, L_RH
def plot_X1_X2_Y(X1_blue, X2_green, Y):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    ax1.plot(X1_blue, Y, s=5, color='b', edgecolor='none')
    ax1.axvline(0, c='k')
    ax2.scatter(X2_green, Y, s=5, color='g', edgecolor='none')
    ax2.axvline(0, c='k')

    plt.show()
def plot_T_RH_I_(V_, I_line):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    ASL_ = V_[:, -1]

    ax1.set_ylabel('ASL')
    ax1.set_xlabel('Temp')
    ax2.set_xlabel('RH')

    ax1.scatter(V_[:, 1], ASL_, s=5, color='b', edgecolor='none')
    ax1.axvline(0, c='k')
    RH_temp = V_[:, 2]
    RH_temp = RH_temp
    ax2.scatter(RH_temp, ASL_, s=5, color='g', edgecolor='none')
    ax2.axvline(0, c='k')
    for x in range(I_line.shape[0]):
        plt.axhline(I_line[x, 0], c='r')
    plt.show()
def plot_ThetaVirtual_I_(V_, I_line):
    fig, ax1 = plt.subplots()

    ASL_ = V_[:, -1]

    ax1.set_ylabel('ASL')
    ax1.set_xlabel('Virtual Potential Temperature [K]')

    ax1.scatter(V_[:, 5], ASL_, s=5, color='b', edgecolor='none')


    for x in range(I_line.shape[0]):
        plt.axhline(I_line[x, 0], c='r')

    plt.show()
def last_lat_lon_alt_ozonesonde(filename_):
    data_array = genfromtxt(filename_, delimiter=',', dtype='<U32', skip_header=23)
    return data_array[-1,31], data_array[-1,32], data_array[-1,33], data_array[-1,0]
def load_khancoban_sondes(filename_):
    line_number = -1
    dict_ = {}
    dict_['filename'] = filename_.split('\\')[-1]
    dict_['date'] = '20' + filename_.split('\\')[-1][2:]
    profile_header = []
    profile_units = []
    profile_data = []
    with open(filename_) as file_object:
        for line in file_object:
            line_number += 1
            line_items = line.split()

            if  17 <= line_number <= 35:
                profile_header.append(line_items[0])
                profile_units.append(line_items[1])

            if line_number >= 39 and len(line_items)>1:
                profile_data.append(line_items)

    profile_array = np.zeros((len(profile_data), len(profile_data[0])), dtype=float)

    for r_ in range(len(profile_data)):
        profile_array[r_, :] = profile_data[r_]

    for c_ in range(len(profile_header)):
        dict_[profile_header[c_]] = {}
        dict_[profile_header[c_]]['data'] = profile_array[:, c_]
        dict_[profile_header[c_]]['units'] = profile_units[c_]


    return dict_
def convert_khan_sonde_data_to_skewt_dict(khan_dict, sonde_name):

    # create time array in seconds since epoc
    date_seconds = time_str_to_seconds(khan_dict[sonde_name]['date'], '%Y%m%d.0%H')
    time_sonde_sec = date_seconds + khan_dict[sonde_name]['time']['data']



    mydata_0=dict(zip(('hght','pres','temp','dwpt', 'sknt', 'drct', 'relh', 'time', 'lati', 'long'),
                      (khan_dict[sonde_name]['Height']['data'],
                       khan_dict[sonde_name]['P']['data'],
                       kelvin_to_celsius(khan_dict[sonde_name]['T']['data']),
                       kelvin_to_celsius(khan_dict[sonde_name]['TD']['data']),
                       ws_ms_to_knots(khan_dict[sonde_name]['FF']['data']),
                       khan_dict[sonde_name]['DD']['data'],
                       khan_dict[sonde_name]['RH']['data'],
                       time_sonde_sec,
                       khan_dict[sonde_name]['Lat']['data'],
                       khan_dict[sonde_name]['Lon']['data']
                       )))
    return mydata_0


# data averaging
def average_all_data_files(filename_, number_of_seconds, WD_index = None, WS_index = None,
                           min_data_number=None, cumulative_parameter_list=None):
    header_, values_ = load_time_columns(filename_)
    time_sec = time_days_to_seconds(values_[:,0])

    # wind tratment
    if WD_index is not None and WS_index is not None:
        print('wind averaging underway for parameters: ' + header_[WD_index] + ' and ' + header_[WS_index])
        # converting wind parameters to cartesian
        WD_ = values_[:,WD_index]
        WS_ = values_[:,WS_index]
        North_, East_ = polar_to_cart(WD_, WS_)
        values_[:,WD_index] = North_
        values_[:,WS_index] = East_

    # averaging
    if min_data_number is None: min_data_number = int(number_of_seconds/60 * .75)
    if cumulative_parameter_list is None:
        Index_mean,Values_mean = mean_discrete(time_sec, values_[:,2:], number_of_seconds,
                                               time_sec[0], min_data = min_data_number,
                                               cumulative_parameter_indx= None)
    else:
        Index_mean,Values_mean = mean_discrete(time_sec, values_[:,2:], number_of_seconds,
                                               time_sec[0], min_data = min_data_number,
                                               cumulative_parameter_indx=np.array(cumulative_parameter_list) - 2)


    if WD_index is not None and WS_index is not None:
        # converting wind parameters to polar
        North_ = Values_mean[:,WD_index - 2]
        East_ = Values_mean[:,WS_index - 2]
        WD_, WS_ = cart_to_polar(North_, East_)
        Values_mean[:,WD_index - 2] = WD_
        Values_mean[:,WS_index - 2] = WS_

    output_filename = filename_.split('.')[0]
    output_filename += '_' + str(int(number_of_seconds/60)) + '_minute_mean' + '.csv'
    save_array_to_disk(header_[2:], Index_mean, Values_mean, output_filename)
    print('Done!')
    print('saved at: ' + output_filename)
def median_discrete(Index_, Values_, avr_size, first_index, min_data=1, position_=0.0):
    # Index_: n by 1 numpy array to look for position,
    # Values_: n by m numpy array, values to be averaged
    # avr_size in same units as Index_,
    # first_index is the first discrete index on new arrays.
    # min_data is minimum amount of data for average to be made (optional, default = 1)
    # position_ will determine where is the stamp located; 0 = beginning, .5 = mid, 1 = top (optional, default = 0)
    # this will average values from Values_ that are between Index_[n:n+avr_size)
    # will return: Index_averaged, Values_averaged

    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(Index_.shape[0]-1):
        if Index_[x]==Index_[x] and Index_[x+1]==Index_[x+1]:
            if Index_[x+1] < Index_[x]:
                always_ascending = 0
    if always_ascending == 0:
        MM_ = np.column_stack((Index_,Values_))
        MM_sorted = MM_[MM_[:,0].argsort()] # sort by first column
        Index_ = MM_sorted[:,0]
        Values_ = MM_sorted[:,1:]

    # error checking!
    if Index_.shape[0] != Values_.shape[0]:
        return None, None
    if Index_[-1] < first_index:
        return None, None
    if min_data < 1:
        return None, None

    # initialize averaged matrices
    final_index = np.nanmax(Index_)
    total_averaged_rows = int((final_index-first_index)/avr_size) + 1
    if len(Values_.shape) == 1:
        Values_median = np.zeros(total_averaged_rows)
        Values_median[:] = np.nan
    else:
        Values_median = np.zeros((total_averaged_rows,Values_.shape[1]))
        Values_median[:,:] = np.nan
    Index_averaged = np.zeros(total_averaged_rows)
    for r_ in range(total_averaged_rows):
        Index_averaged[r_] = first_index + (r_ * avr_size)

    Index_averaged -= (position_ * avr_size)
    Values_25pr = np.array(Values_median)
    Values_75pr = np.array(Values_median)
    Std_ = np.array(Values_median)


    indx_avr_r = -1
    last_raw_r = 0
    r_raw_a = 0
    r_raw_b = 1
    while indx_avr_r <= total_averaged_rows-2:
        indx_avr_r += 1
        indx_a = Index_averaged[indx_avr_r]
        indx_b = Index_averaged[indx_avr_r] + avr_size
        stamp_population = 0
        for r_raw in range(last_raw_r,Index_.shape[0]):
            if indx_a <= Index_[r_raw] < indx_b:
                if stamp_population == 0: r_raw_a = r_raw
                r_raw_b = r_raw + 1
                stamp_population += 1
            if Index_[r_raw] >= indx_b:
                last_raw_r = r_raw
                break
        if stamp_population >= min_data:
            if len(Values_.shape) == 1:
                Values_median[indx_avr_r] = np.nanmedian(Values_[r_raw_a:r_raw_b])
                Values_25pr[indx_avr_r] = np.nanmedian(Values_[r_raw_a:r_raw_b])
                Values_75pr[indx_avr_r] = np.nanmedian(Values_[r_raw_a:r_raw_b])
                Std_[indx_avr_r] = np.nanstd(Values_[r_raw_a:r_raw_b])
            else:
                for c_ in range(Values_.shape[1]):
                    Values_median[indx_avr_r,c_] = np.nanmedian(Values_[r_raw_a:r_raw_b,c_])
                    Values_25pr[indx_avr_r,c_] = np.nanpercentile(Values_[r_raw_a:r_raw_b,c_],25)
                    Values_75pr[indx_avr_r,c_] = np.nanpercentile(Values_[r_raw_a:r_raw_b,c_],75)
                    Std_[indx_avr_r] = np.nanstd(Values_[r_raw_a:r_raw_b],c_)


    Index_averaged = Index_averaged + (position_ * avr_size)

    return Index_averaged,Values_median,Values_25pr,Values_75pr, Std_
def mean_discrete(Index_, Values_, avr_size, first_index,
                  min_data=1, position_=0., cumulative_parameter_indx=None, last_index=None, show_progress=True):
    """
    this will average values from Values_ that are between Index_[n:n+avr_size)
    :param Index_: n by 1 numpy array to look for position,
    :param Values_: n by m numpy array, values to be averaged
    :param avr_size: in same units as Index_
    :param first_index: is the first discrete index on new arrays.
    :param min_data: is minimum amount of data for average to be made (optional, default = 1)
    :param position_: will determine where is the stamp located; 0 = beginning, .5 = mid, 1 = top (optional, default = 0)
    :param cumulative_parameter_indx: in case there is any column in Values_ to be summed, not averaged. Most be a list
    :param last_index: in case you want to force the returned series to some fixed period/length
    :return: Index_averaged, Values_averaged
    """

    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(Index_.shape[0]-1):
        if Index_[x]==Index_[x] and Index_[x+1]==Index_[x+1]:
            if Index_[x+1] < Index_[x]:
                always_ascending = 0
    if always_ascending == 0:
        MM_ = np.column_stack((Index_,Values_))
        MM_sorted = MM_[MM_[:,0].argsort()] # sort by first column
        Index_ = MM_sorted[:,0]
        Values_ = MM_sorted[:,1:]

    # error checking!
    if Index_.shape[0] != Values_.shape[0]:
        print('error during shape check! Index_.shape[0] != Values_.shape[0]')
        return None, None
    if Index_[-1] < first_index:
        print('error during shape check! Index_[-1] < first_index')
        return None, None
    if min_data < 1:
        print('error during shape check! min_data < 1')
        return None, None

    # initialize averaged matrices
    if last_index is None:
        final_index = np.nanmax(Index_)
    else:
        final_index = last_index

    total_averaged_rows = int((final_index-first_index)/avr_size) + 1
    if len(Values_.shape) == 1:
        Values_mean = np.zeros(total_averaged_rows)
        Values_mean[:] = np.nan
    else:
        Values_mean = np.zeros((total_averaged_rows,Values_.shape[1]))
        Values_mean[:,:] = np.nan
    Index_averaged = np.zeros(total_averaged_rows)
    for r_ in range(total_averaged_rows):
        Index_averaged[r_] = first_index + (r_ * avr_size)

    Index_averaged -= (position_ * avr_size)

    indx_avr_r = -1
    last_raw_r = 0
    r_raw_a = 0
    r_raw_b = 1
    while indx_avr_r <= total_averaged_rows-2:
        if show_progress: p_progress_bar(indx_avr_r, total_averaged_rows-2, extra_text='averaged')
        indx_avr_r += 1
        indx_a = Index_averaged[indx_avr_r]
        indx_b = Index_averaged[indx_avr_r] + avr_size
        stamp_population = 0
        for r_raw in range(last_raw_r,Index_.shape[0]):
            if indx_a <= Index_[r_raw] < indx_b:
                if stamp_population == 0: r_raw_a = r_raw
                r_raw_b = r_raw + 1
                stamp_population += 1
            if Index_[r_raw] >= indx_b:
                last_raw_r = r_raw
                break
        if stamp_population >= min_data:
            if len(Values_.shape) == 1:
                if cumulative_parameter_indx is not None:
                    Values_mean[indx_avr_r] = np.nansum(Values_[r_raw_a:r_raw_b])
                else:
                    Values_mean[indx_avr_r] = np.nanmean(Values_[r_raw_a:r_raw_b])
            else:
                for c_ in range(Values_.shape[1]):
                    if cumulative_parameter_indx is not None:
                        if c_ in cumulative_parameter_indx:
                            Values_mean[indx_avr_r, c_] = np.nansum(Values_[r_raw_a:r_raw_b, c_])
                        else:
                            Values_mean[indx_avr_r, c_] = np.nanmean(Values_[r_raw_a:r_raw_b, c_])
                    else:
                        Values_mean[indx_avr_r,c_] = np.nanmean(Values_[r_raw_a:r_raw_b,c_])

    Index_averaged = Index_averaged + (position_ * avr_size)

    return Index_averaged,Values_mean
def mean_discrete_std(Index_, Values_, avr_size, first_index, min_data=1, position_=0.):
    # Index_: n by 1 numpy array to look for position,
    # Values_: n by m numpy array, values to be averaged
    # avr_size in same units as Index_,
    # first_index is the first discrete index on new arrays.
    # min_data is minimum amount of data for average to be made (optional, default = 1)
    # position_ will determine where is the stamp located; 0 = beginning, .5 = mid, 1 = top (optional, default = 0)
    # this will average values from Values_ that are between Index_[n:n+avr_size)
    # will return: Index_averaged, Values_averaged

    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(Index_.shape[0]-1):
        if Index_[x]==Index_[x] and Index_[x+1]==Index_[x+1]:
            if Index_[x+1] < Index_[x]:
                always_ascending = 0
    if always_ascending == 0:
        MM_ = np.column_stack((Index_,Values_))
        MM_sorted = MM_[MM_[:,0].argsort()] # sort by first column
        Index_ = MM_sorted[:,0]
        Values_ = MM_sorted[:,1:]

    # error checking!
    if Index_.shape[0] != Values_.shape[0]:
        return None, None
    if Index_[-1] < first_index:
        return None, None
    if min_data < 1:
        return None, None

    # initialize averaged matrices
    final_index = np.nanmax(Index_)
    total_averaged_rows = int((final_index-first_index)/avr_size) + 1
    if len(Values_.shape) == 1:
        Values_mean = np.zeros(total_averaged_rows)
        Values_mean[:] = np.nan
    else:
        Values_mean = np.zeros((total_averaged_rows,Values_.shape[1]))
        Values_mean[:,:] = np.nan
    Index_averaged = np.zeros(total_averaged_rows)
    for r_ in range(total_averaged_rows):
        Index_averaged[r_] = first_index + (r_ * avr_size)

    Index_averaged -= (position_ * avr_size)
    Std_ = np.array(Values_mean)

    indx_avr_r = -1
    last_raw_r = 0
    r_raw_a = 0
    r_raw_b = 1
    while indx_avr_r <= total_averaged_rows-2:
        indx_avr_r += 1
        indx_a = Index_averaged[indx_avr_r]
        indx_b = Index_averaged[indx_avr_r] + avr_size
        stamp_population = 0
        for r_raw in range(last_raw_r,Index_.shape[0]):
            if indx_a <= Index_[r_raw] < indx_b:
                if stamp_population == 0: r_raw_a = r_raw
                r_raw_b = r_raw + 1
                stamp_population += 1
            if Index_[r_raw] >= indx_b:
                last_raw_r = r_raw
                break
        if stamp_population >= min_data:
            if len(Values_.shape) == 1:
                Values_mean[indx_avr_r] = np.nanmean(Values_[r_raw_a:r_raw_b])
                Std_[indx_avr_r] = np.nanstd(Values_[r_raw_a:r_raw_b])
            else:
                for c_ in range(Values_.shape[1]):
                    Values_mean[indx_avr_r,c_] = np.nanmean(Values_[r_raw_a:r_raw_b,c_])
                    Std_[indx_avr_r] = np.nanstd(Values_[r_raw_a:r_raw_b],c_)

    Index_averaged = Index_averaged + (position_ * avr_size)

    return Index_averaged,Values_mean,Std_
def sum_discrete_3D_array(Index_, array_3D, sum_size, first_index, min_data=1, position_=0.):
    # Index_: n by 1 numpy array to look for position,
    # Values_: n by m numpy array, values to be averaged
    # avr_size in same units as Index_,
    # first_index is the first discrete index on new arrays.
    # min_data is minimum amount of data for average to be made (optional, default = 1)
    # position_ will determine where is the stamp located; 0 = beginning, .5 = mid, 1 = top (optional, default = 0)
    # this will average values from Values_ that are between Index_[n:n+avr_size)
    # will return: Index_averaged, Values_averaged

    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(Index_.shape[0]-1):
        if Index_[x]==Index_[x] and Index_[x+1]==Index_[x+1]:
            if Index_[x+1] < Index_[x]:
                always_ascending = 0
    if always_ascending == 0:
        print('Error, index must always be ascending')
        return None, None

    # error checking!
    if Index_.shape[0] != array_3D.shape[0]:
        print('Error, axes 0 of 3D array must be equal to Index size')
        return None, None
    if Index_[-1] < first_index:
        print('Error, first')
        return None, None

    # initialize averaged matrices
    final_index = np.nanmax(Index_)
    total_summed_rows = int((final_index-first_index)/sum_size) + 1

    Values_sum = np.zeros((total_summed_rows, array_3D.shape[1], array_3D.shape[2]))
    Values_sum[:,:,:] = np.nan
    Index_summed = np.zeros(total_summed_rows)
    for r_ in range(total_summed_rows):
        Index_summed[r_] = first_index + (r_ * sum_size)

    Index_summed -= (position_ * sum_size)

    indx_sum_r = -1
    last_raw_r = 0
    r_raw_a = 0
    r_raw_b = 1
    while indx_sum_r <= total_summed_rows-2:
        indx_sum_r += 1
        indx_a = Index_summed[indx_sum_r]
        indx_b = Index_summed[indx_sum_r] + sum_size
        stamp_population = 0
        for r_raw in range(last_raw_r,Index_.shape[0]):
            if indx_a <= Index_[r_raw] < indx_b:
                if stamp_population == 0: r_raw_a = r_raw
                r_raw_b = r_raw + 1
                stamp_population += 1
            if Index_[r_raw] >= indx_b:
                last_raw_r = r_raw
                break
        if stamp_population >= min_data:
            Values_sum[indx_sum_r,:,:] = np.nansum(array_3D[r_raw_a:r_raw_b,:,:],axis=0)
    Index_summed = Index_summed + (position_ * sum_size)

    return Index_summed,Values_sum
def row_average_rolling(arr_, average_size):
    result_ = np.array(arr_) * np.nan

    for r_ in range(arr_.shape[0] +1 - int(average_size)):
        result_[r_] = np.nanmean(arr_[r_ : r_ + average_size])

    return result_
def row_average_discrete_1D(arr_, average_size):
    result_ = np.zeros(int(arr_.shape[0]/average_size)) * np.nan

    for r_ in range(result_.shape[0]):
        result_[r_] = np.nanmean(arr_[int(r_* average_size) : int(r_* average_size) + average_size], axis=0)

    return result_
def row_average_discrete_2D(arr_, average_size):
    result_ = np.zeros((int(arr_.shape[0]/average_size), arr_.shape[1])) * np.nan

    for r_ in range(result_.shape[0]):
        result_[r_,:] = np.nanmean(arr_[int(r_* average_size) : int(r_* average_size) + average_size], axis=0)

    return result_
def row_average_discrete_3D(arr_, average_size):
    result_ = np.zeros((int(arr_.shape[0]/average_size), arr_.shape[1], arr_.shape[2])) * np.nan

    for r_ in range(result_.shape[0]):
        result_[r_,:,:] = np.nanmean(arr_[int(r_* average_size) : int(r_* average_size) + average_size], axis=0)

    return result_
def column_average_discrete_2D(arr_, average_size):
    result_ = np.zeros((arr_.shape[0], int(arr_.shape[1]/average_size))) * np.nan

    for c_ in range(result_.shape[1]):
        result_[:, c_] = np.nanmean(arr_[:, int(c_* average_size) : int(c_* average_size) + average_size], axis=1)

    return result_
def column_average_discrete_3D(arr_, average_size):
    result_ = np.zeros((arr_.shape[0], int(arr_.shape[1]/average_size), arr_.shape[2])) * np.nan

    for c_ in range(result_.shape[1]):
        result_[:, c_,:] = np.nanmean(arr_[:, int(c_* average_size) : int(c_* average_size) + average_size,:], axis=1)

    return result_
def average_all_data_files_monthly(filename_, number_of_seconds, min_data_number = None,
                                   WD_index = None, WS_index = None, cumulative_parameter_list=None):
    header_, values_ = load_time_columns(filename_)
    time_sec = time_days_to_seconds(values_[:,0])

    # wind tratment
    if WD_index is not None and WS_index is not None:
        print('wind averaging underway for parameters: ' + header_[WD_index] + ' and ' + header_[WS_index])
        # converting wind parameters to cartesian
        WD_ = values_[:,WD_index]
        WS_ = values_[:,WS_index]
        North_, East_ = polar_to_cart(WD_, WS_)
        values_[:,WD_index] = North_
        values_[:,WS_index] = East_

    # averaging
    if min_data_number is None: min_data_number = int(number_of_seconds/60 * .75)
    if cumulative_parameter_list is None:
        Index_mean,Values_mean = mean_discrete(time_sec, values_[:,2:], number_of_seconds,
                                               time_sec[0], min_data = min_data_number,
                                               cumulative_parameter_indx= None)
    else:
        Index_mean,Values_mean = mean_discrete(time_sec, values_[:,2:], number_of_seconds,
                                               time_sec[0], min_data = min_data_number,
                                               cumulative_parameter_indx=np.array(cumulative_parameter_list) - 2)


    if WD_index is not None and WS_index is not None:
        # converting wind parameters to polar
        North_ = Values_mean[:,WD_index - 2]
        East_ = Values_mean[:,WS_index - 2]
        WD_, WS_ = cart_to_polar(North_, East_)
        Values_mean[:,WD_index - 2] = WD_
        Values_mean[:,WS_index - 2] = WS_

    output_filename = filename_.split('.')[0]
    output_filename += '_' + str(int(number_of_seconds/60)) + '_minute_mean' + '.csv'
    save_array_to_disk(header_[2:], Index_mean, Values_mean, output_filename)
    print('Done!')
    print('saved at: ' + output_filename)
def rolling_window(array_, window_size):
    shape = array_.shape[:-1] + (array_.shape[-1] - window_size + 1, window_size)
    strides = array_.strides + (array_.strides[-1],)
    return np.lib.stride_tricks.as_strided(array_, shape=shape, strides=strides)



# wind direction related
def polar_to_cart(WD_, WS_):
    WD_rad = np.radians(WD_)
    North_ = WS_ * np.cos(WD_rad)
    East_ = WS_ * np.sin(WD_rad)
    return North_, East_
def cart_to_polar(North_, East_):
    try:
        WS_ = np.sqrt(North_**2 + East_**2)
        WD_with_neg = np.degrees(np.arctan2(East_, North_))
        mask_ = np.zeros(WD_with_neg.shape[0])
        mask_[WD_with_neg < 0] = 360
        WD_ = WD_with_neg + mask_
    except:
        WS_ = np.sqrt(North_**2 + East_**2)
        WD_with_neg = np.degrees(np.arctan2(East_, North_))
        mask_ = 0
        if WD_with_neg < 0:
            mask_ = 360
        WD_ = WD_with_neg + mask_
    return WD_, WS_


# time transforms
def combine_by_index(reference_index, var_index, var_values):
    """
    finds point from var_index to each reference_index point, has to be exact, if not found then nan
    :param reference_index: 1d array
    :param var_index: 1d array of same size as var_values
    :param var_values: 1d or 2d array of same size as var_index
    :return: reindexed_var_values of same size as reference_index
    """

    rows_ = reference_index.shape[0]
    if len(var_values.shape) == 1:
        reindexed_var_values = np.zeros(rows_) * np.nan

        for r_ in range(rows_):
            p_progress(r_, rows_)
            where_ = np.where(var_index == reference_index[r_])[0]
            if len(where_) > 0:
                reindexed_var_values[r_] = var_values[where_[0]]

        return reindexed_var_values
    else:
        reindexed_var_values = np.zeros((rows_, var_values.shape[1])) * np.nan

        for r_ in range(rows_):
            p_progress(r_, rows_)
            where_ = np.where(var_index == reference_index[r_])[0]
            if len(where_) > 0:
                reindexed_var_values[r_, :] = var_values[where_[0], :]

        return reindexed_var_values
def time_seconds_to_days(time_in_seconds):
    return mdates.epoch2num(time_in_seconds)
def time_days_to_seconds(time_in_days):
    return mdates.num2epoch(time_in_days)
def time_str_to_seconds(time_str, time_format):
    # defining time arrays
    if isinstance(time_str, str):
        time_seconds = calendar.timegm(time.strptime(time_str, time_format))
    else:
        time_seconds = np.array([calendar.timegm(time.strptime(time_string_record, time_format))
                                 for time_string_record in time_str])
    return time_seconds
def time_seconds_to_str(time_in_seconds, time_format):
    try:
        x = len(time_in_seconds)
        if isinstance(time_in_seconds, list):
            time_in_seconds = np.array(time_in_seconds)
        temp_array = np.zeros(time_in_seconds.shape[0],dtype="<U32")
        for r_ in range(time_in_seconds.shape[0]):
            temp_array[r_] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime(time_format)
        return temp_array
    except:
        return datetime.datetime.utcfromtimestamp(time_in_seconds).strftime(time_format)
def time_seconds_to_5C_array(time_in_seconds):
    if isinstance(time_in_seconds, int):
        out_array = np.zeros(5, dtype=int)
        out_array[0] = datetime.datetime.utcfromtimestamp(time_in_seconds).strftime('%Y')
        out_array[1] = datetime.datetime.utcfromtimestamp(time_in_seconds).strftime('%m')
        out_array[2] = datetime.datetime.utcfromtimestamp(time_in_seconds).strftime('%d')
        out_array[3] = datetime.datetime.utcfromtimestamp(time_in_seconds).strftime('%H')
        out_array[4] = datetime.datetime.utcfromtimestamp(time_in_seconds).strftime('%M')
    else:
        out_array = np.zeros((time_in_seconds.shape[0], 5), dtype=int)
        for r_ in range(time_in_seconds.shape[0]):
            out_array[r_, 0] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime('%Y')
            out_array[r_, 1] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime('%m')
            out_array[r_, 2] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime('%d')
            out_array[r_, 3] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime('%H')
            out_array[r_, 4] = datetime.datetime.utcfromtimestamp(time_in_seconds[r_]).strftime('%M')
    return out_array
def time_era5_to_seconds(time_in_era5):
    time_in_era5_in_seconds = np.array(time_in_era5, dtype=float) * 60 * 60
    time_format_era5 = 'hours since %Y-%m-%d %H:%M'
    time_seconds_start = calendar.timegm(time.strptime('hours since 1900-01-01 00:00', time_format_era5))
    time_seconds_epoc = time_in_era5_in_seconds + time_seconds_start
    return time_seconds_epoc
def time_seconds_to_struct(time_in_seconds):
    time_struct_list = []
    for t_ in time_in_seconds:
        time_struct_list.append(time.gmtime(t_))
    return time_struct_list
def time_to_row_str(time_array_seconds, time_stamp_str_YYYYmmDDHHMM):
    time_stamp_seconds = time_str_to_seconds(time_stamp_str_YYYYmmDDHHMM, time_format_parsivel)
    row_ = np.argmin(np.abs(time_array_seconds - time_stamp_seconds))
    return row_
def time_to_row_sec(time_array_seconds, time_stamp_sec):
    row_ = np.argmin(np.abs(time_array_seconds - time_stamp_sec))
    return row_
def time_period_to_row_tuple(time_array_seconds, time_stamp_start_stop_str_YYYYmmDDHHMM):
    time_start_seconds = time_str_to_seconds(time_stamp_start_stop_str_YYYYmmDDHHMM.split('_')[0], time_format_parsivel)
    time_stop_seconds = time_str_to_seconds(time_stamp_start_stop_str_YYYYmmDDHHMM.split('_')[1], time_format_parsivel)
    row_1 = np.argmin(np.abs(time_array_seconds - time_start_seconds))
    row_2 = np.argmin(np.abs(time_array_seconds - time_stop_seconds))
    return row_1, row_2
def convert_any_time_type_to_days(time_series, print_show=False):
    time_days_normal_range = [727000, 748000]
    time_secs_normal_range = [646800000, 2540240000]

    # check if it is a str
    if isinstance(time_series, str):
        # try each known str_time_format and return time_seconds_to_days()
        for time_str_format in time_str_formats:
            try:
                time_in_secs = time_str_to_seconds(time_series, time_str_format)
                return time_seconds_to_days(time_in_secs)
            except:
                pass
        if print_show: print('could not find correct time string format! returning nan')
        return np.nan

    # if not str, check if it is a single number
    if isinstance(time_series, float) or isinstance(time_series, int):
        if time_secs_normal_range[0] < time_series < time_secs_normal_range[1]:
            return time_seconds_to_days(time_series)
        elif time_days_normal_range[0] < time_series < time_days_normal_range[1]:
            return time_series
        else:
            if print_show: print('could not find correct time number correction! returning nan')
            return np.nan
    else:
        # multiple items
        # check if series of strs
        try:
            if isinstance(time_series[0], str):
                # try each known str_time_format and return time_seconds_to_days()
                for time_str_format in time_str_formats:
                    try:
                        time_in_secs = time_str_to_seconds(time_series, time_str_format)
                        return time_seconds_to_days(time_in_secs)
                    except:
                        pass
                if print_show: print('could not find correct time string format! returning None')
                return None
            else:
                # get max and min
                time_series_min = np.nanmin(time_series)
                time_series_max = np.nanmax(time_series)

                if time_secs_normal_range[0] < time_series_min and time_series_max < time_secs_normal_range[1]:
                    return time_seconds_to_days(time_series)
                elif time_days_normal_range[0] < time_series_min and time_series_max < time_days_normal_range[1]:
                    return time_series
                else:
                    if print_show: print('could not find correct time number correction! returning None')
                    return None

        except:
            if print_show: print('unknown type of data, returning None')
            return None
def time_rman_blist_to_seconds(rman_2D_b_array, time_format='%H:%M:%S %d/%m/%Y'):
    """
    takes bite arrays and converts to seconds
    :param rman_2D_b_array: array where each row is a time stamp and columns are a character in bite format
    :param time_format: string that defines the structure of the characters in each time stamp
    :return: seconds array
    """

    time_str_list = []
    for row_ in range(rman_2D_b_array.shape[0]):
        t_str = ''
        for i in rman_2D_b_array[row_]:
            t_str = t_str + i.decode('UTF-8')
        time_str_list.append(t_str)

    time_seconds = time_str_to_seconds(time_str_list, time_format)

    return time_seconds
def create_time_series_seconds(start_time_str, stop_time_str, step_size):
    start_time_sec = float(time_days_to_seconds(convert_any_time_type_to_days(start_time_str)))
    stop__time_sec = float(time_days_to_seconds(convert_any_time_type_to_days(stop_time_str )))

    time_list = []

    t_ = start_time_sec

    while t_ < stop__time_sec:
        time_list.append(t_)
        t_ += step_size
    return np.array(time_list)
def day_night_discrimination(hour_of_day,values_,day_hours_range_tuple_inclusive):
    day_ = np.array(values_) * np.nan
    night_ = np.array(values_) * np.nan
    for r_ in range(values_.shape[0]):
        if day_hours_range_tuple_inclusive[0] <= hour_of_day[r_] <= day_hours_range_tuple_inclusive[1]:
            day_[r_,:] = values_[r_,:]
        else:
            night_[r_,:] = values_[r_,:]
    return day_, night_
def create_time_stamp_list_between_two_times(datetime_start_str,
                                             datetime_end_str,
                                             time_steps_in_sec,
                                             input_time_format='%Y%m%d%H%M',
                                             output_list_format='%Y%m%d%H%M'):

    datetime_start_sec = time_str_to_seconds(datetime_start_str, input_time_format)
    datetime_end_sec = time_str_to_seconds(datetime_end_str, input_time_format)
    number_of_images = (datetime_end_sec - datetime_start_sec) / time_steps_in_sec
    datetime_list_str = []
    for time_stamp_index in range(int(number_of_images)):
        datetime_list_str.append(time_seconds_to_str(datetime_start_sec + (time_stamp_index * time_steps_in_sec),
                                                     output_list_format))

    return datetime_list_str



# animation
def update_animation_img(frame_number, img_animation, ax_, frame_list, title_list):
    p_progress_bar(frame_number, len(frame_list), extra_text='of video created')
    try:
        new_frame = frame_list[frame_number,:,:]
    except:
        new_frame = frame_list[frame_number]
    img_animation.set_data(new_frame)
    ax_.set_title(str(title_list[frame_number]))
    # ax_.set_xlabel(title_list[frame_number])
    return img_animation
def update_animation_img_pcolormesh(frame_number, img_animation, ax_, frame_list, title_list):
    p_progress(frame_number, len(frame_list), extra_text='of video created')
    try:
        new_frame = frame_list[frame_number,:,:]
    except:
        new_frame = frame_list[frame_number]
    img_animation.set_array(new_frame.ravel())
    ax_.set_title(str(title_list[frame_number]))
    return img_animation
def update_animation_img_img_list(frame_number, img_animation, ax_, frame_list, title_list):
    p_progress(frame_number, len(frame_list), extra_text='of video created')
    new_frame = frame_list[frame_number]
    img_animation.set_data(new_frame)
    ax_.set_title(str(title_list[frame_number]))
    # ax_.set_xlabel(title_list[frame_number])
    return img_animation
def update_animation_img_scatter_list(frame_number, img_plot, sca_plot, ax_img,
                                      frame_list, scatter_array_x, scatter_array_y, title_list):
    p_progress(frame_number, len(frame_list), extra_text='of video created')
    new_frame_img = frame_list[frame_number]
    new_frame_sca_x = scatter_array_x[:frame_number]
    new_frame_sca_y = scatter_array_y[:frame_number]
    img_plot.set_data(new_frame_img)
    sca_plot.set_data(new_frame_sca_x, new_frame_sca_y)
    ax_img.set_title(str(title_list[frame_number]))
    # ax_.set_xlabel(title_list[frame_number])
    return img_plot
def animate_parsivel(frame_number, t_list, size_array, speed_array, spectrum_array_color, cmap_parsivel, img_plot, ax):
    img_plot.remove()
    img_plot = ax.pcolormesh(size_array, speed_array, spectrum_array_color[frame_number, :, :],
                             cmap=cmap_parsivel, vmin=0, vmax=8)
    ax.set_title(str(t_list[frame_number]))
    return img_plot
def create_video_animation_from_array_list(array_list, out_filename, colormap_=default_cm, extend_='', interval_=50,
                                           dpi_=200, show_=False, save_=True, cbar_label='', title_list=None):
    fig, ax_ = plt.subplots()

    min_ = np.nanmin(array_list)
    max_ = np.nanmax(array_list)

    if title_list is None:
        title_list_ = np.arange(len(array_list))
    else:
        title_list_ = title_list
    if extend_=='':
        img_figure = ax_.imshow(array_list[0], interpolation='none', cmap=colormap_, vmin=min_, vmax=max_)
    else:
        img_figure = ax_.imshow(array_list[0], interpolation='none', cmap=colormap_, vmin=min_, vmax=max_,
                        extent=[extend_[1], extend_[3], extend_[2], extend_[0]])
    color_bar = fig.colorbar(img_figure)
    color_bar.ax.set_ylabel(cbar_label)

    img_animation = FuncAnimation(fig, update_animation_img, len(array_list), fargs=(img_figure, ax_, array_list, title_list_), interval=interval_)

    if show_: plt.show()
    if save_:
        img_animation.save(out_filename, metadata={'artist':'Guido'}, dpi=dpi_)
        plt.close(fig)

    print('Done')
def create_video_animation_from_3D_array(array_, out_filename, colormap_=default_cm, extend_='', interval_=50, dpi_=200,
                                         show_=False, save_=True, cbar_label='', title_list=None, format_='%.2f',
                                         axes_off=False, show_colorbar=True, vmin_=None, vmax_=None):
    fig, ax_ = plt.subplots()

    if vmin_ is None: vmin_ = np.nanmin(array_)
    if vmax_ is None: vmax_ = np.nanmax(array_)

    if title_list is None or len(title_list) != array_.shape[0]:
        title_list_ = np.arange(array_.shape[0])
    else:
        title_list_ = title_list
    if extend_=='':
        img_figure = ax_.imshow(array_[0,:,:], interpolation='none', cmap=colormap_, vmin=vmin_, vmax=vmax_)
    else:
        img_figure = ax_.imshow(array_[0,:,:], interpolation='none', cmap=colormap_, vmin=vmin_, vmax=vmax_,
                                extent=[extend_[1], extend_[3], extend_[2], extend_[0]])
    if show_colorbar:
        color_bar = fig.colorbar(img_figure,format=format_)
        color_bar.ax.set_ylabel(cbar_label)

    if axes_off: ax_.set_axis_off()

    img_animation = FuncAnimation(fig, update_animation_img, array_.shape[0], fargs=(img_figure, ax_, array_, title_list_), interval=interval_)

    if show_: plt.show()
    if save_:
        # img_animation.save(out_filename, writer='ffmpeg', codec='rawvideo')
        img_animation.save(out_filename, metadata={'artist':'Guido'}, dpi=dpi_)
        plt.close(fig)

    print('Done')
def create_video_animation_from_img_arrays_list(array_list, out_filename, interval_=50, dpi_=200, show_=False,
                                                save_=True, title_list=None):
    fig, ax_ = plt.subplots()

    if title_list is None:
        title_list_ = np.arange(len(array_list))
    else:
        title_list_ = title_list
    img_figure = ax_.imshow(array_list[0], interpolation='none')

    ax_.set_axis_off()

    img_animation = FuncAnimation(fig, update_animation_img_img_list, len(array_list),
                                  fargs=(img_figure, ax_, array_list, title_list_), interval=interval_)

    if show_: plt.show()
    if save_:
        img_animation.save(out_filename, metadata={'artist':'Guido'}, dpi=dpi_)
        plt.close(fig)

    print('Done')
def create_video_animation_from_3D_array_pcolormesh(array_values, array_x, array_y, out_filename, colormap_=default_cm,
                                                    interval_=50, dpi_=200, show_=False, save_=True,
                                                    cbar_label='', title_list=None,format_='%.2f', axes_off=False,
                                                    show_colorbar=True, x_header='', y_header='',
                                                    custom_y_range_tuple=None, custom_x_range_tuple=None,
                                                    vmin_=None, vmax_=None):
    fig, ax_ = plt.subplots()

    if vmin_ is None: vmin_ = np.nanmin(array_values)
    if vmax_ is None: vmax_ = np.nanmax(array_values)

    if title_list is None or len(title_list) != array_values.shape[0]:
        title_list_ = np.arange(array_values.shape[0])
    else:
        title_list_ = title_list

    img_figure = ax_.pcolormesh(array_x, array_y, array_values[0,:,:], cmap=colormap_,
                                vmin=vmin_, vmax=vmax_)#, shading='gouraud')
    ax_.set_xlabel(x_header)
    ax_.set_ylabel(y_header)

    if custom_y_range_tuple is not None: ax_.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax_.set_xlim(custom_x_range_tuple)

    if show_colorbar:
        color_bar = fig.colorbar(img_figure,format=format_)
        color_bar.ax.set_ylabel(cbar_label)

    if axes_off: ax_.set_axis_off()


    img_animation = FuncAnimation(fig, update_animation_img_pcolormesh, frames=array_values.shape[0],
                                  fargs=(img_figure, ax_, array_values, title_list_), interval=interval_)

    if show_: plt.show()
    if save_:
        # img_animation.save(out_filename, writer='ffmpeg', codec='rawvideo')
        img_animation.save(out_filename, metadata={'artist':'Guido'}, dpi=dpi_)
        plt.close(fig)

    print('Done')




# display / plotting
def p_plot(X_series,Y_,
           S_=5, c_='', label_=None,
           x_header=None,y_header=None, t_line=False, grid_=False, cus_loc =None, cmap_=default_cm,
           custom_y_range_tuple=None, custom_x_range_tuple=None, figsize_ = (10,6), save_fig=False, figure_filename='',
           custom_x_ticks_start_end_step=None, custom_y_ticks_start_end_step=None, extra_text='', title_str = '',
           time_format_=None, x_as_time=True, c_header=None, add_line=False, linewidth_=2, fig_ax=None,
           line_color='black', invert_y=False, invert_x=False, log_x=False,log_y=False, transparent_=True,
           density_=False, t_line_1_1 = True, t_line_color = 'r', fit_function=None, show_cbar=False,
           text_box_str=None, text_box_loc=None, skewt=False, filled_arr=None,
           legend_show=False, legend_loc='upper left'):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        if skewt:
            fig = plt.figure(figsize=figsize_)
            ax = fig.add_subplot(111, projection='skewx')
        else:
            fig, ax = plt.subplots(figsize=figsize_)

    x_is_time_cofirmed = True
    if x_as_time==True and density_==False and invert_x==False and log_x==False:
        X_ = convert_any_time_type_to_days(X_series)
        if X_ is None:
            X_ = X_series
            x_is_time_cofirmed = False
    else:
        X_ = X_series
        x_is_time_cofirmed = False


    if skewt:
        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        if c_ == '': c_ = 'black'
        ax.semilogy(X_, Y_, color=c_)

        # Disables the log-formatting that comes with semilogy
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(np.linspace(100, 1000, 10))
        ax.set_ylim(1050, 100)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlim(-50, 50)
        x_as_time = False
        ax.grid(True)
    else:
        if density_:
            ax = p_density_scatter(X_, Y_, s = S_, fig_ax=[fig, ax], cmap_=cmap_, show_cbar=show_cbar)
        else:
            if c_=='':
                if add_line:
                    ax.scatter(X_, Y_, s=S_, lw=0, c='black')
                    ax.plot(X_, Y_, c=line_color, linewidth=linewidth_, label=label_)
                    if filled_arr is not None:
                        ax.fill_between(X_, Y_, filled_arr, facecolor=line_color, interpolate=True)
                else:
                    ax.scatter(X_, Y_, s=S_, lw=0, c='black', label=label_)
            elif type(c_) == str:
                if add_line:
                    ax.plot(X_, Y_, c=c_, linewidth=linewidth_, label=label_)
                    ax.scatter(X_, Y_, s=S_, lw=0, c=c_)
                    if filled_arr is not None:
                        ax.fill_between(X_, Y_, filled_arr, facecolor=line_color, interpolate=True)
                else:
                    ax.scatter(X_, Y_, s=S_, lw=0, c=c_, label=label_)
            else:
                im = ax.scatter(X_,Y_, s = S_, lw = 0,  c = c_, cmap = cmap_)
                color_bar = fig.colorbar(im,fraction=0.046, pad=0.04)
                if c_header is not None: color_bar.ax.set_ylabel(c_header)
    if x_header is not None: ax.set_xlabel(x_header)
    if y_header is not None: ax.set_ylabel(y_header)
    # ax.yaxis.set_ticks(np.arange(180, 541, 45))
    if grid_:
        ax.grid(True)
    if t_line:
        Rsqr = plot_trend_line(ax, X_, Y_, c=t_line_color, alpha=1, cus_loc = cus_loc,
                        extra_text=extra_text, t_line_1_1= t_line_1_1, fit_function=fit_function)
    else:
        Rsqr = None

    if invert_y:
        ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()
    if log_x:
        ax.set_xscale("log")#, nonposy='clip')
    if log_y:
        ax.set_yscale("log")#, nonposy='clip')

    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None:
        if x_as_time == True and density_ == False and invert_x == False and log_x == False and x_is_time_cofirmed == True:
            r_1 = convert_any_time_type_to_days(custom_x_range_tuple[0])
            r_2 = convert_any_time_type_to_days(custom_x_range_tuple[1])
            ax.set_xlim((r_1,r_2))
        else:
            ax.set_xlim(custom_x_range_tuple)

    if custom_x_ticks_start_end_step is not None:
        ax.xaxis.set_ticks(np.arange(custom_x_ticks_start_end_step[0], custom_x_ticks_start_end_step[1],
                                     custom_x_ticks_start_end_step[2]))
    if custom_y_ticks_start_end_step is not None:
        ax.yaxis.set_ticks(np.arange(custom_y_ticks_start_end_step[0], custom_y_ticks_start_end_step[1],
                                     custom_y_ticks_start_end_step[2]))

    if x_as_time==True and density_==False and invert_x==False and log_x==False and x_is_time_cofirmed==True:

        if time_format_ is None:
            plot_format_mayor = mdates.DateFormatter(time_format_mod)
        else:
            plot_format_mayor = mdates.DateFormatter(time_format_)
        ax.xaxis.set_major_formatter(plot_format_mayor)

    if legend_show:
        ax.legend(loc=legend_loc)
    ax.set_title(title_str)

    if text_box_str is not None:
        if text_box_loc is None:
            x_1 = ax.axis()[0]
            y_2 = ax.axis()[3]

            text_color = 'black'
            ax.text(x_1, y_2 , str(text_box_str),
                           horizontalalignment='left',verticalalignment='top',color=text_color)
        else:
            x_1 = text_box_loc[0]
            y_2 = text_box_loc[1]
            text_color = 'black'
            ax.text(x_1, y_2 , str(text_box_str),
                           horizontalalignment='left',verticalalignment='top',color=text_color)

    if save_fig or figure_filename != '':
        if figure_filename == '':
            name_ = str(calendar.timegm(time.gmtime()))[:-2]
            fig.savefig(path_output + 'image_' + name_ + '.png', transparent=True, bbox_inches='tight')
        else:
            fig.savefig(figure_filename, transparent=transparent_, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, Rsqr
def p_(header_):
    # parameters list
    print('-' * 20)
    print('Parameters: ')
    parameter_list = ''
    fill_len = len(str(len(header_)))
    for i, parameter_ in enumerate(header_):
        parameter_list =  str(parameter_list) + str(i).rjust(fill_len) + " ---> " + str(parameter_) + '\n'
    print(parameter_list)
    print('-' * 20)
def p_progress(current_count, total_count, display_each_percent=10, extra_text='done'):
    if total_count <= display_each_percent:
        if total_count > 0:
            print(int(100 * current_count / total_count), '%', extra_text)
    else:
        total_count_corrected = int(total_count / display_each_percent) * display_each_percent
        if display_each_percent * current_count / total_count_corrected % 1 == 0:
            if 0 < int(100 * current_count / total_count_corrected) <= 100:
                print(int(100 * current_count / total_count_corrected), '%', extra_text)
def p_progress_bar(current_count, total_count, extra_text='done'):
    display_each_percent = 5
    units_ = int(100 / display_each_percent)

    if current_count == 0:
        print('|' + ' ' *  units_ + '| %', extra_text, end="", flush=True)

    if current_count == total_count -1:
        print('\r', end='')
        print('|' + '-' *  units_ + '| %', extra_text + '!finished!')
    else:
        if total_count <= units_:
            if total_count > 0:
                print('\r', end='')
                print('|', end="", flush=True)
                str_ = '-' * current_count
                str_ = str_ + ' ' * (units_ - current_count)
                print(str_, end="", flush=True)
                print('| % ', extra_text, end="", flush=True)
        else:
            percentage_ = int((current_count / total_count) * 100)
            if percentage_ / display_each_percent % 1 == 0:
                if 0 < percentage_ <= 100:
                    print('\r', end='')
                    print('|', end="", flush=True)
                    str_ = '-' * int(percentage_ / display_each_percent)
                    str_ = str_ + ' ' * (units_ - int(percentage_ / display_each_percent))
                    print(str_, end="", flush=True)
                    print('| % ', extra_text, end="", flush=True)


def p_hist(data_, figsize_ = (10,6), fig_ax=None, title_str='', x_header=None, y_header=None, x_bins=None):
    if len(data_.shape) > 1:
        data_display = data_.flatten()
    else:
        data_display = data_

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize_)


    ax.hist(data_display[~np.isnan(data_display)],x_bins)

    ax.set_title(title_str)
    if x_header is not None: ax.set_xlabel(x_header)
    if y_header is not None: ax.set_ylabel(y_header)

    return fig, ax
def get_chart_range(ax):
    x_1 = ax.axis()[0]
    x_2 = ax.axis()[1]
    y_1 = ax.axis()[2]
    y_2 = ax.axis()[3]
    return x_1,x_2,y_1,y_2

def p_arr_vectorized(A_, cmap_=default_cm, figsize_= (10,6), vmin_=None,vmax_=None, cbar_label = ''):
    fig, ax = plt.subplots(figsize=figsize_)

    if vmin_ is None: vmin_ = np.nanmin(A_)
    if vmax_ is None: vmax_ = np.nanmax(A_)

    y_, x_ = np.mgrid[0:A_.shape[0], 0:A_.shape[1]]

    surf_ = ax.pcolormesh(x_, y_, A_, cmap=cmap_, vmin=vmin_, vmax=vmax_)

    color_bar = fig.colorbar(surf_)
    color_bar.ax.set_ylabel(cbar_label)

    return fig, ax
def p_arr_vectorized_2(array_v, array_x, array_y,custom_y_range_tuple=None, custom_x_range_tuple=None,
                       x_header='', y_header='', cbar_label = '', title_str='',
                       cmap_=default_cm, figsize_= (10,6), vmin_=None,vmax_=None,
                       figure_filename = None, time_format_ = None):
    fig, ax = plt.subplots(figsize=figsize_)
    # plt.close(fig)

    if vmin_ is None: vmin_ = np.nanmin(array_v)
    if vmax_ is None: vmax_ = np.nanmax(array_v)

    if len(array_x.shape) == 1:
        array_y_reshaped = np.zeros((array_v.shape[0], array_v.shape[1]), dtype=float)
        array_x_reshaped = np.zeros((array_v.shape[0], array_v.shape[1]), dtype=float)

        for r_ in range(array_v.shape[0]):
            array_y_reshaped[r_, :] = array_y
        for c_ in range(array_v.shape[1]):
            array_x_reshaped[:, c_] = array_x

    else:
        array_y_reshaped = array_y
        array_x_reshaped = array_x

    surf_ = ax.pcolormesh(array_x_reshaped, array_y_reshaped, array_v, cmap=cmap_, vmin=vmin_, vmax=vmax_)

    color_bar = fig.colorbar(surf_)
    color_bar.ax.set_ylabel(cbar_label)
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    ax.set_title(title_str)
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)

    if time_format_ is not None:
        plot_format_mayor = mdates.DateFormatter(time_format_)
        ax.xaxis.set_major_formatter(plot_format_mayor)

    if figure_filename is not None:
        fig.savefig(figure_filename, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return


    return fig, ax
def p_arr_vectorized_3(array_v, array_x, array_y,
                       custom_y_range_tuple=None, custom_x_range_tuple=None,
                       custom_ticks_x=None, custom_ticks_y=None,
                       x_header='', y_header='', cbar_label = '', title_str='', contour_=False, contourF_=False,
                       cmap_=default_cm, figsize_= (10,6), vmin_=None, vmax_=None, show_cbar=True, cbar_format='%.2f',
                       figure_filename = None, grid_=False, time_format_ = None, fig_ax=None,
                       colorbar_tick_labels_list=None, show_x_ticks=True, show_y_ticks=True,cbar_ax=None,
                       invert_y=False, invert_x=False, levels=None, text_box_str=None,text_box_loc=None):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize_)

    if vmin_ is None: vmin_ = np.nanmin(array_v)
    if vmax_ is None: vmax_ = np.nanmax(array_v)

    if len(array_x.shape) == 1:
        array_x_reshaped = np.zeros((array_v.shape[0], array_v.shape[1]), dtype=float)
        for c_ in range(array_v.shape[1]):
            array_x_reshaped[:, c_] = array_x
    else:
        array_x_reshaped = array_x
    array_x = array_x_reshaped

    if len(array_y.shape) == 1:
        array_y_reshaped = np.zeros((array_v.shape[0], array_v.shape[1]), dtype=float)
        for r_ in range(array_v.shape[0]):
            array_y_reshaped[r_, :] = array_y
    else:
        array_y_reshaped = array_y

    array_y = array_y_reshaped
    if time_format_ is not None:
        array_x = convert_any_time_type_to_days(array_x_reshaped)

    if contour_:
        surf_ = ax.contour(array_x, array_y, array_v, levels=levels, cmap=cmap_, vmin=vmin_, vmax=vmax_)
    elif contourF_:
        surf_ = ax.contourf(array_x, array_y, array_v, levels=levels, cmap=cmap_, vmin=vmin_, vmax=vmax_)
    else:
        surf_ = ax.pcolormesh(array_x, array_y, array_v, cmap=cmap_, vmin=vmin_, vmax=vmax_)

    if show_cbar:
        if cbar_ax is None:
            color_bar = fig.colorbar(surf_, format=cbar_format)
        else:
            color_bar = fig.colorbar(surf_, format=cbar_format, cax=cbar_ax)
        color_bar.ax.set_ylabel(cbar_label)

        if colorbar_tick_labels_list is not None:
            ticks_ = np.linspace(0.5, len(colorbar_tick_labels_list) - 0.5, len(colorbar_tick_labels_list))
            color_bar.set_ticks(ticks_)
            color_bar.set_ticklabels(colorbar_tick_labels_list)

    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    ax.set_title(title_str)
    ax.grid(grid_)


    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)

    if time_format_ is not None:
        plot_format_mayor = mdates.DateFormatter(time_format_)
        ax.xaxis.set_major_formatter(plot_format_mayor)
        ax.format_coord = lambda x, y: 'x=%s, y=%g, v=%g' % (plot_format_mayor(x),
                                                             y,
                                                             array_v[int(np.argmin(np.abs(array_x[:, 0] - x))), int(
                                                                 np.argmin(np.abs(array_y[0, :] - y)))])
    else:
        ax.format_coord = lambda x, y: 'x=%1.2f, y=%g, v=%g' % (x,
                                                                y,
                                                                array_v[
                                                                    int(np.argmin(np.abs(array_x[:, 0] - x))), int(
                                                                        np.argmin(np.abs(array_y[0, :] - y)))])

    if not show_x_ticks:
        plt.setp(ax.get_xticklabels(), visible=False)
    if not show_y_ticks:
        plt.setp(ax.get_yticklabels(), visible=False)


    if invert_y:
        ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()

    if custom_ticks_x is not None: ax.xaxis.set_ticks(custom_ticks_x)
    if custom_ticks_y is not None: ax.yaxis.set_ticks(custom_ticks_y)

    if text_box_str is not None:
        if text_box_loc is None:
            x_1 = ax.axis()[0]
            y_2 = ax.axis()[3]

            text_color = 'black'
            ax.text(x_1, y_2 , str(text_box_str),
                           horizontalalignment='left',verticalalignment='top',color=text_color)
        else:
            x_1 = text_box_loc[0]
            y_2 = text_box_loc[1]
            text_color = 'black'
            ax.text(x_1, y_2 , str(text_box_str),
                           horizontalalignment='left',verticalalignment='top',color=text_color)



    if figure_filename is not None:
        fig.savefig(figure_filename, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return

    return fig, ax, surf_
def p_arr(A_, cmap_=default_cm, extend_x1_x2_y1_y2 =(0,1), figsize_= (10, 6), aspect_='auto', rot_=0, title_str = '',
          vmin_=None, vmax_=None, cbar_label = '', x_as_time=False, time_format_='%H:%M %d%b%y', save_fig=False,
          figure_filename='', x_header='',y_header='', x_ticks_tuple=None, y_ticks_tuple=None, fig_ax=None,
          origin_='upper', colorbar_tick_labels_list=None, tick_label_format='plain', tick_offset=False):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize_)

    A_copy = np.array(A_)
    if vmin_ is not None: A_copy[A_copy < vmin_] = vmin_
    if vmax_ is not None: A_copy[A_copy > vmax_] = vmax_

    if rot_ != 0:
        A_copy = np.rot90(A_copy, rot_)

    if len(extend_x1_x2_y1_y2)==2:
        img_ = ax.imshow(A_copy, interpolation='none', cmap=cmap_, aspect= aspect_, vmin=vmin_, vmax=vmax_, origin=origin_)
    else:
        img_ = ax.imshow(A_copy, interpolation='none', cmap=cmap_, aspect= aspect_, origin=origin_, vmin=vmin_, vmax=vmax_,
                         extent=[extend_x1_x2_y1_y2[0], extend_x1_x2_y1_y2[1], extend_x1_x2_y1_y2[2], extend_x1_x2_y1_y2[3]])

    color_bar = fig.colorbar(img_)
    color_bar.ax.set_ylabel(cbar_label)


    if colorbar_tick_labels_list is not None:
        ticks_ = np.linspace(0.5, len(colorbar_tick_labels_list) - 0.5, len(colorbar_tick_labels_list))
        color_bar.set_ticks(ticks_)
        color_bar.set_ticklabels(colorbar_tick_labels_list)


    if x_as_time:
        plot_format_mayor = mdates.DateFormatter(time_format_)
        ax.xaxis.set_major_formatter(plot_format_mayor)

    ax.set_title(title_str)
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)

    if x_ticks_tuple is not None:
        ax.xaxis.set_ticks(np.arange(x_ticks_tuple[0], x_ticks_tuple[1], x_ticks_tuple[2]))
    if y_ticks_tuple is not None:
        ax.yaxis.set_ticks(np.arange(y_ticks_tuple[0], y_ticks_tuple[1], y_ticks_tuple[2]))

    ax.ticklabel_format(useOffset=tick_offset, style='plain')
    plt.tight_layout()

    if save_fig or figure_filename != '':
        if figure_filename == '':
            name_ = str(calendar.timegm(time.gmtime()))[:-2]
            fig.savefig(path_output + 'image_' + name_ + '.png', transparent=True, bbox_inches='tight')
        else:
            fig.savefig(figure_filename, transparent=False, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig, ax, img_, color_bar
def p_plot_colored_lines(x_array, y_array, color_array, tick_labels_list, fig_ax=None, figsize_= (10, 6),
                         x_header='', y_header='', figure_filename = None, time_format='', cbar_show=True,
                         custom_y_range_tuple=None, custom_x_range_tuple=None, grid_=False, cbar_ax=None,
                         cmap = listed_cm):
    # plot rain rate colored by rain type
    points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize_)

    # Use a boundary norm instead
    norm = BoundaryNorm(np.arange(len(tick_labels_list)+1), cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_array)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    if cbar_show:
        if cbar_ax is None:
            cb2 = fig.colorbar(line, ax=ax)
        else:
            cb2 = fig.colorbar(line, cax=cbar_ax)

        ticks_ = np.linspace(0.5, len(tick_labels_list) - 0.5, len(tick_labels_list))
        cb2.set_ticks(ticks_)
        cb2.set_ticklabels(tick_labels_list)

    # x_array = convert_any_time_type_to_days(x_array)

    ax.set_xlim(x_array.min(),
                x_array.max())
    ax.set_ylim(y_array.min(), y_array.max())
    ax.set_ylabel(y_header)
    ax.set_xlabel(x_header)
    ax.grid(grid_)
    if time_format != '':
        plot_format_mayor = mdates.DateFormatter(time_format)
        ax.xaxis.set_major_formatter(plot_format_mayor)
    # plt.xticks(rotation=45)
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)
    if figure_filename is not None:
        fig.savefig(figure_filename , transparent=True, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def plot_3D_scatter(x_series, y_series, z_series, label_names_tuples_xyz=tuple(''), size_ = 15, color_='b'):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x_series, y_series, z_series,s=size_,c=color_,lw = 0)


    if len(label_names_tuples_xyz) == 3:
        ax.set_xlabel(label_names_tuples_xyz[0])
        ax.set_ylabel(label_names_tuples_xyz[1])
        ax.set_zlabel(label_names_tuples_xyz[2])



    plt.show()
    return fig, ax
def plot_3D_stacket_series_lines(x_z_series_list, y_series=None, y_as_time=False, time_format=time_format,
                                 log_z=False, invert_z=False,
                                 custom_x_range_tuple=None, custom_y_range_tuple=None, custom_z_range_tuple=None,
                                 label_names_tuples_xyz=tuple(''), color_='b'):
    fig = plt.figure()
    ax = Axes3D(fig)

    if y_series is None:
        y_series = np.arange(len(x_z_series_list))

    for t_ in range(len(x_z_series_list)):
        y_ = np.ones(len(x_z_series_list[t_][0])) * y_series[t_]
        ax.plot(x_z_series_list[t_][0], y_, x_z_series_list[t_][1], c=color_)


    if len(label_names_tuples_xyz) == 3:
        ax.set_xlabel(label_names_tuples_xyz[0])
        ax.set_ylabel(label_names_tuples_xyz[1])
        ax.set_zlabel(label_names_tuples_xyz[2])

    if y_as_time:
        plot_format_mayor = mdates.DateFormatter(time_format)
        ax.yaxis.set_major_formatter(plot_format_mayor)

    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_z_range_tuple is not None: ax.set_zlim(custom_z_range_tuple)

    if log_z:
        ax.set_zscale("log")#, nonposy='clip')
    if invert_z:
        ax.invert_zaxis()

    ax.yaxis.set_ticks(y_series)

    plt.show()
    return fig, ax
def plot_shared_x_axis(X_Y_list, S_=5, x_header=None,y_header_list=None, t_line=False, grid_=False, cus_loc =None,
                       c_='', custom_y_range_tuple=None, custom_x_range_tuple=None, figsize_ = (10,6), save_fig=False,
                       figure_filename='',title_str = '', cmap_=default_cm, sharex=True, sharey=False,
                       custom_x_ticks_start_end_step=None, custom_y_ticks_start_end_step=None, rot_y_label=90,
                       time_format_='%H:%M %d%b%y', x_as_time=False, add_line=False, linewidth_=2,
                       invert_y=False, invert_x=False, log_x=False,log_y=False, transparent_=True):

    fig, (ax_list) = plt.subplots(nrows=len(X_Y_list), sharex=sharex, sharey=sharey, figsize=figsize_)

    if c_=='':
        n = int(len(X_Y_list))
        color_list = cmap_(np.linspace(0, 1, n))
        for series_number in range(len(X_Y_list)):
            ax_list[series_number].scatter(X_Y_list[series_number][0],X_Y_list[series_number][1],
                                           c= color_list[series_number], s = S_, lw = 0)
            if add_line:
                ax_list[series_number].plot(X_Y_list[series_number][0], X_Y_list[series_number][1],
                                            c=color_list[series_number], linewidth=linewidth_)
    else:
        for series_number in range(len(X_Y_list)):
            ax_list[series_number].scatter(X_Y_list[series_number][0],X_Y_list[series_number][1],
                                                s = S_, lw = 0,  c = c_)

    if x_header is not None: ax_list[-1].set_xlabel(x_header)

    for series_number in range(len(X_Y_list)):
        if y_header_list is not None:
            ax_list[series_number].set_ylabel(y_header_list[series_number], rotation=rot_y_label)
        if grid_:
            ax_list[series_number].grid(True)
        if t_line:
            plot_trend_line(ax_list[series_number], X_Y_list[series_number][0],X_Y_list[series_number][1],
                            order=1, c='r', alpha=1, cus_loc = cus_loc)

        if custom_y_range_tuple is not None: ax_list[series_number].set_ylim(custom_y_range_tuple)
        if custom_x_range_tuple is not None: ax_list[series_number].set_xlim(custom_x_range_tuple)

        if custom_x_ticks_start_end_step is not None:
            ax_list[series_number].xaxis.set_ticks(np.arange(custom_x_ticks_start_end_step[0],
                                                             custom_x_ticks_start_end_step[1],
                                                             custom_x_ticks_start_end_step[2]))
        if custom_y_ticks_start_end_step is not None:
            ax_list[series_number].yaxis.set_ticks(np.arange(custom_y_ticks_start_end_step[0],
                                                             custom_y_ticks_start_end_step[1],
                                                             custom_y_ticks_start_end_step[2]))

        if x_as_time:
            plot_format_mayor = mdates.DateFormatter(time_format_)
            ax_list[series_number].xaxis.set_major_formatter(plot_format_mayor)

        if invert_y:
            ax_list[series_number].invert_yaxis()
        if invert_x:
            ax_list[series_number].invert_xaxis()
        if log_x:
            ax_list[series_number].set_xscale("log", nonposy='clip')
        if log_y:
            ax_list[series_number].set_yscale("log", nonposy='clip')

    for series_number in range(len(X_Y_list)-1):
        plt.setp(ax_list[series_number].get_xticklabels(), visible=False)

    ax_list[0].set_title(title_str)
    fig.tight_layout()

    if save_fig or figure_filename != '':
        if figure_filename == '':
            name_ = str(calendar.timegm(time.gmtime()))[:-2]
            fig.savefig(path_output + 'image_' + name_ + '.png', transparent=True, bbox_inches='tight')
        else:
            fig.savefig(figure_filename, transparent=transparent_, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax_list
def plot_shared_y_axis(X_Y_list, S_=5, x_header_list=None, y_header=None, t_line=False, grid_=False, cus_loc=None,
                       c_='', custom_y_range_tuple=None, custom_x_range_tuple=None, figsize_=(10, 6), save_fig=False,
                       figure_filename='', title_str='', cmap_=default_cm, sharex=False, sharey=True,
                       custom_x_ticks_start_end_step=None, custom_y_ticks_start_end_step=None,
                       time_format_='%H:%M %d%b%y', x_as_time=False, add_line=False, linewidth_=2,
                       invert_y=False, invert_x=False, log_x=False, log_y=False, transparent_=True):
    fig, (ax_list) = plt.subplots(ncolumns=len(X_Y_list), sharex=sharex, sharey=sharey, figsize=figsize_)

    if c_ == '':
        n = int(len(X_Y_list))
        color_list = cmap_(np.linspace(0, 1, n))
        for series_number in range(len(X_Y_list)):
            ax_list[series_number].scatter(X_Y_list[series_number][0], X_Y_list[series_number][1],
                                           c=color_list[series_number], s=S_, lw=0)
            if add_line:
                ax_list[series_number].plot(X_Y_list[series_number][0], X_Y_list[series_number][1],
                                            c=color_list[series_number], linewidth=linewidth_)
    else:
        for series_number in range(len(X_Y_list)):
            ax_list[series_number].scatter(X_Y_list[series_number][0], X_Y_list[series_number][1],
                                           s=S_, lw=0, c=c_[series_number], cmap=cmap_)

    if y_header is not None: ax_list[0].set_ylabel(y_header)

    for series_number in range(len(X_Y_list)):
        if x_header_list is not None:
            ax_list[series_number].set_ylabel(x_header_list[series_number])
        if grid_:
            ax_list[series_number].grid(True)
        if t_line:
            plot_trend_line(ax_list[series_number], X_Y_list[series_number][0], X_Y_list[series_number][1],
                            order=1, c='r', alpha=1, cus_loc=cus_loc)

        if custom_y_range_tuple is not None: ax_list[series_number].set_ylim(custom_y_range_tuple)
        if custom_x_range_tuple is not None: ax_list[series_number].set_xlim(custom_x_range_tuple)

        if custom_x_ticks_start_end_step is not None:
            ax_list[series_number].xaxis.set_ticks(np.arange(custom_x_ticks_start_end_step[0],
                                                             custom_x_ticks_start_end_step[1],
                                                             custom_x_ticks_start_end_step[2]))
        if custom_y_ticks_start_end_step is not None:
            ax_list[series_number].yaxis.set_ticks(np.arange(custom_y_ticks_start_end_step[0],
                                                             custom_y_ticks_start_end_step[1],
                                                             custom_y_ticks_start_end_step[2]))

        if x_as_time:
            plot_format_mayor = mdates.DateFormatter(time_format_)
            ax_list[series_number].xaxis.set_major_formatter(plot_format_mayor)

        if invert_y:
            ax_list[series_number].invert_yaxis()
        if invert_x:
            ax_list[series_number].invert_xaxis()
        if log_x:
            ax_list[series_number].set_xscale("log", nonposy='clip')
        if log_y:
            ax_list[series_number].set_yscale("log", nonposy='clip')

    for series_number in range(len(X_Y_list) - 1):
        plt.setp(ax_list[series_number+1].get_xticklabels(), visible=False)

    ax_list[0].set_title(title_str)

    fig.tight_layout()
    if save_fig or figure_filename != '':
        if figure_filename == '':
            name_ = str(calendar.timegm(time.gmtime()))[:-2]
            fig.savefig(path_output + 'image_' + name_ + '.png', transparent=True, bbox_inches='tight')
        else:
            fig.savefig(figure_filename, transparent=transparent_, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax_list

def scatter_custom_size(X_,Y_,S_, x_header=None,y_header=None, t_line=False, grid_=False, cus_loc =None, c_='',
                        custom_y_range_tuple=None, custom_x_range_tuple=None, figsize_ = (10,6), save_fig=False,
                        custom_x_ticks_start_end_step=None, custom_y_ticks_start_end_step=None, extra_text='',
                        time_format_='%H:%M %d%b%y', x_as_time=False, c_header=None, add_line=False, linewidth_=2,
                        line_color='black'):
    fig, ax = plt.subplots(figsize=figsize_)
    if c_=='':
        ax.scatter(X_,Y_, s = S_, lw = 0, c = 'black')
        if add_line:
            ax.plot(X_, Y_, c=line_color, linewidth=linewidth_)
    else:
        im = ax.scatter(X_,Y_, s = S_, lw = 0,  c = c_, cmap = default_cm)
        color_bar = fig.colorbar(im,fraction=0.046, pad=0.04)
        if c_header is not None: color_bar.ax.set_ylabel(c_header)
    if x_header is not None: ax.set_xlabel(x_header)
    if y_header is not None: ax.set_ylabel(y_header)
    # ax.yaxis.set_ticks(np.arange(180, 541, 45))
    if grid_:
        ax.grid(True)
    if t_line:
        plot_trend_line(ax, X_, Y_, order=1, c='r', alpha=1, cus_loc = cus_loc, extra_text=extra_text)

    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)

    if custom_x_ticks_start_end_step is not None:
        ax.xaxis.set_ticks(np.arange(custom_x_ticks_start_end_step[0], custom_x_ticks_start_end_step[1], custom_x_ticks_start_end_step[2]))
    if custom_y_ticks_start_end_step is not None:
        ax.yaxis.set_ticks(np.arange(custom_y_ticks_start_end_step[0], custom_y_ticks_start_end_step[1], custom_y_ticks_start_end_step[2]))

    if x_as_time:
        plot_format_mayor = mdates.DateFormatter(time_format_)
        ax.xaxis.set_major_formatter(plot_format_mayor)

    if save_fig:
        name_ = str(calendar.timegm(time.gmtime()))[:-2]
        fig.savefig(path_output + 'image_' + name_ + '.png',transparent=True, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax
def Display_emission_array(filename_, variable_name):
    netcdf_file_object = nc.Dataset(filename_, 'r')
    p_arr(netcdf_file_object.variables[variable_name][0, 0, ::-1, :])

    netcdf_file_object.close()
def power_plot(X_, Y_, Size_=5, x_header='',y_header='', trend_line=False, show_line=False, lw_=2, grid_=False,
               cus_loc = '', c_='', custom_y_range_tuple=None, custom_x_range_tuple=None, cbar_label = ''):
    fig, ax = plt.subplots()
    if c_=='':
        ax.scatter(X_,Y_, s = Size_, lw = 0, c = 'black')
        if show_line:
            ax.plot(X_,Y_, lw = lw_, color = 'black')
    else:
        im = ax.scatter(X_,Y_, s = Size_, lw = 0,  c = c_, cmap = default_cm)
        ax.plot(X_,Y_, lw = lw_,  c = c_, cmap = default_cm)
        color_bar = fig.colorbar(im,fraction=0.046, pad=0.04)
        color_bar.ax.set_ylabel(cbar_label)
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    if grid_:
        ax.grid(True)
    if trend_line:
        plot_trend_line(ax, X_, Y_, order=1, c='r', alpha=1, cus_loc = cus_loc)

    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)
    plt.show()
    return fig, ax
def power_plot_with_error(X_, Y_, yerr_, Size_=5, c_='', x_header='',y_header='', trend_line=False, lw_=2, grid_=False,
                          cus_loc = '', custom_y_range_tuple=None, custom_x_range_tuple=None, cbar_label = ''):
    fig, ax = plt.subplots()
    if c_=='':
        ax.scatter(X_,Y_, s = Size_, lw = 0, c = 'black')
        ax.errorbar(X_,Y_, yerr=yerr_, color = 'black')
    else:
        im = ax.scatter(X_,Y_, s = Size_, lw = 0,  c = c_, cmap = default_cm)
        ax.plot(X_,Y_, lw = lw_,  c = c_, cmap = default_cm)
        color_bar = fig.colorbar(im,fraction=0.046, pad=0.04)
        color_bar.ax.set_ylabel(cbar_label)
    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    if grid_:
        ax.grid(True)
    if trend_line:
        plot_trend_line(ax, X_, Y_, order=1, c='r', alpha=1, cus_loc = cus_loc)

    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)
    plt.show()
def plot_preview_x_as_time(header_,days_,values_):

    plot_format_mayor = mdates.DateFormatter('%H:%M %d%b%y')
    fig, ax = plt.subplots()
    if len(values_.shape) > 1:
        for c_ in range(values_.shape[1]):
            ax.plot_date(days_,values_[:,c_], markersize=2, markeredgewidth=0, label=header_[c_])
    else:
        ax.plot_date(days_,values_,'ko-', markersize=2, markeredgewidth=0, label=header_)
    ax.xaxis.set_major_formatter(plot_format_mayor)
    plt.show()
def plot_values_x_as_time(header_,values_,x_array,y_list,
                          legend_=False, plot_fmt_str0='%H:%M %d%b%y'):
    color_list = default_cm(np.linspace(0,1,len(y_list)))
    plot_format_mayor = mdates.DateFormatter(plot_fmt_str0)
    fig, ax = plt.subplots()
    for c_,y_ in enumerate(y_list):
        color_ = color_list[c_]
        ax.plot(x_array,values_[:,y_], color = color_,label=header_[y_])
    ax.xaxis.set_major_formatter(plot_format_mayor)
    fig.tight_layout()
    if legend_: ax.legend(loc=(.95,.0))
    plt.show()
def plot_trend_line(axes_, xd, yd, c='r', alpha=1, cus_loc = None, text_color='black',
                    extra_text='', t_line_1_1=True, fit_function=None):
    """Make a line of best fit"""
    #create clean series
    x_, y_ = coincidence(xd,yd)



    if fit_function is not None:
        params = curve_fit(fit_function, x_, y_)
        print('fitted parameters')
        print(params[0])

        fit_line_x = np.arange(int(np.nanmin(x_)),int(np.nanmax(x_))+1,.1)
        plotting_par_list = [fit_line_x]
        for fit_par in params[0]:
            plotting_par_list.append(fit_par)
        funt_par = tuple(plotting_par_list)
        fit_line_y = fit_function(*funt_par)
        axes_.plot(fit_line_x, fit_line_y, c, alpha=alpha)


        # calculate R2
        plotting_par_list = [x_]
        params_str_ = ''
        for i_, fit_par in enumerate(params[0]):
            if extra_text == '':
                params_str_ = params_str_ + 'fit parameters ' + str(i_+1) + ': ' + '$%0.2f$' % (fit_par) + '\n'
            else:
                params_str_ = params_str_ + extra_text + '$%0.2f$' % (fit_par) + '\n'
            plotting_par_list.append(fit_par)
        funt_par = tuple(plotting_par_list)
        fit_line_y = fit_function(*funt_par)
        residuals = y_ - fit_line_y
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_ - np.mean(y_))**2)
        Rsqr = float(1 - (ss_res / ss_tot))

        # Plot R^2 value
        x_1 = np.nanmin(x_)
        y_2 = np.nanmax(y_)
        error_text = '$R^2 = %0.2f$' % Rsqr
        if cus_loc is None:
            axes_.text(x_1, y_2 , params_str_ + error_text,
                       horizontalalignment='left',verticalalignment='top',color=text_color)
        else:
            axes_.text(cus_loc[0], cus_loc[1] , params_str_ + error_text,
                       horizontalalignment='left',verticalalignment='top',color=text_color)

    else:
        # Calculate trend line
        coeffs = np.polyfit(x_, y_, 1)
        intercept = coeffs[-1]
        slope = coeffs[-2]
        minxd = np.nanmin(x_)
        maxxd = np.nanmax(x_)

        xl = np.array([minxd, maxxd])
        yl = slope * xl + intercept

        # Plot trend line
        axes_.plot(xl, yl, c, alpha=alpha)

        # Calculate R Squared
        p = np.poly1d(coeffs)
        ybar = np.sum(y_) / len(y_)
        ssreg = np.sum((p(x_) - ybar) ** 2)
        sstot = np.sum((y_ - ybar) ** 2)
        Rsqr = float(ssreg / sstot)

        # Plot R^2 value
        x_1 = np.nanmin(x_)
        y_2 = np.nanmax(y_)
        if intercept >= 0:
            if extra_text=='':
                equat_text = '$Y = %0.2f*x + %0.2f$' % (slope,intercept)
            else:
                equat_text = extra_text + '\n' + '$Y = %0.2f*x + %0.2f$' % (slope,intercept)
        else:
            if extra_text=='':
                equat_text = '$Y = %0.2f*x %0.2f$' % (slope,intercept)
            else:
                equat_text = extra_text + '\n' + '$Y = %0.2f*x %0.2f$' % (slope,intercept)
        error_text = '$R^2 = %0.2f$' % Rsqr
        if cus_loc is None:
            axes_.text(x_1, y_2 , equat_text + '\n' + error_text,
                       horizontalalignment='left',verticalalignment='top',color=text_color)
        else:
            axes_.text(cus_loc[0], cus_loc[1] , equat_text + '\n' + error_text,
                       horizontalalignment='left',verticalalignment='top',color=text_color)
    # plot 1:1 line if true
    if t_line_1_1:
        xy_min = np.min([np.nanmin(x_),np.nanmin(y_)])
        xy_max = np.max([np.nanmax(x_),np.nanmax(y_)])
        axes_.plot([xy_min, xy_max], [xy_min, xy_max], 'k--')

    return Rsqr

def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
def p_density_scatter( x_ , y_, fig_ax = None, cmap_=default_cm, sort = True, bins = 20, show_cbar=False, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """

    x, y = coincidence(x_ , y_)

    if fig_ax is None :
        fig , ax = plt.subplots()
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data ,
                 np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    im = ax.scatter( x, y, c=z, cmap=cmap_, lw=0, **kwargs)

    if show_cbar:
        color_bar = fig.colorbar(im, fraction=0.046, pad=0.04)

    return ax


# diurnal variations
def diurnal_variability_boxplot(time_in_seconds, y_, fig_ax=None, x_header='Hours', y_header='',figure_filename='',
                                bin_size_hours=1, min_bin_population=10, start_value=0, figsize_=(10,6), title_str=''):
    # convert time to struct
    time_hour = np.zeros(time_in_seconds.shape[0], dtype=float)
    time_mins = np.zeros(time_in_seconds.shape[0], dtype=float)
    time_secs = np.zeros(time_in_seconds.shape[0], dtype=float)

    for r_ in range(time_in_seconds.shape[0]):
        time_hour[r_] = time.gmtime(time_in_seconds[r_])[3]
        time_mins[r_] = time.gmtime(time_in_seconds[r_])[4]
        time_secs[r_] = time.gmtime(time_in_seconds[r_])[5]

    time_hours = time_hour + (time_mins + (time_secs/60))/60

    # get coincidences only
    x_val,y_val = coincidence(time_hours, y_)
    # combine x and y in matrix
    M_ = np.column_stack((x_val,y_val))
    # always ascending to increase efficiency
    M_sorted = M_[M_[:,0].argsort()] # sort by first column
    M_ = M_sorted
    # convert data to list of bins
    y_binned = []
    x_binned = []

    start_bin_edge = start_value

    last_row = 0
    last_row_temp = last_row
    while start_bin_edge <= np.nanmax(x_val):
        y_val_list = []
        for row_ in range(last_row, M_.shape[0]):
            if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size_hours:
                if M_[row_, 1] == M_[row_, 1]:
                    y_val_list.append(M_[row_, 1])
                    last_row_temp = row_
            if M_[row_, 0] >= start_bin_edge + bin_size_hours:
                last_row_temp = row_
                break
        x_binned.append(start_bin_edge + (bin_size_hours / 2))
        if len(y_val_list) >= min_bin_population:
            y_binned.append(y_val_list)
        else:
            y_binned.append([])
        start_bin_edge += bin_size_hours
        last_row = last_row_temp
    # start figure
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize_)
    # add series
    if bin_size_hours >= 1:
        x_binned_int = np.array(x_binned)
    else:
        x_binned_int = x_binned
    ax.boxplot(y_binned, 0, '', whis=[5,95], positions = x_binned_int,
                       showmeans = True, widths =bin_size_hours * .9, manage_xticks=False)
    # if user selected x axes as hour
    ax.xaxis.set_ticks(np.arange(0, 24, 3))

    ax.set_xlabel(x_header)
    ax.set_ylabel(y_header)
    ax.set_title(title_str)

    if figure_filename != '':
        fig.savefig(figure_filename, transparent=True, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, x_binned_int, y_binned

def plot_box_from_values(values_x, values_y, x_label=None, y_label=None, bin_size=1, min_bin_population=10,
                         fit_function = None, fit_fuction_by='mean', log_x=False,log_y=False,
                         custom_y_range_tuple = None, custom_x_range_tuple = None,
                         force_start=None, force_end=None, show_means=True,
                         notch=False, sym='', whis=(5,95)):
    x_val_original = values_x
    y_val_original = values_y

    # get coincidences only
    x_val,y_val = coincidence(x_val_original,y_val_original)

    # start figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # combine x and y in matrix
    M_ = np.column_stack((x_val,y_val))
    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(x_val.shape[0]-1):
        if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
            if x_val[x+1] < x_val[x]:
                always_ascending = 0
    if always_ascending == 0:
        M_sorted = M_[M_[:,0].argsort()] # sort by first column
        M_ = M_sorted
    # convert data to list of bins
    y_binned = []
    x_binned = []
    if force_start is None:
        start_bin_edge = np.nanmin(x_val)
    else:
        start_bin_edge = force_start
    if force_end is None:
        stop_bin = np.nanmax(x_val)
    else:
        stop_bin = force_end
    last_row = 0
    last_row_temp = last_row
    while start_bin_edge <= stop_bin:
        y_val_list = []
        for row_ in range(last_row, M_.shape[0]):
            if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                if M_[row_, 1] == M_[row_, 1]:
                    y_val_list.append(M_[row_, 1])
                    last_row_temp = row_
            if M_[row_, 0] >= start_bin_edge + bin_size:
                last_row_temp = row_
                break
        x_binned.append(start_bin_edge)
        if len(y_val_list) >= min_bin_population:
            y_binned.append(y_val_list)
        else:
            y_binned.append([])
        start_bin_edge += bin_size
        last_row = last_row_temp
    if bin_size == 1:
        x_binned_arr = np.array(x_binned, dtype=int)
    else:
        x_binned_arr = np.array(x_binned)
    # add series
    box_dict = ax.boxplot(y_binned, notch=notch, sym=sym, whis=whis, positions = x_binned_arr,
                          showmeans = show_means, widths = bin_size * .9)
    # axes labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if fit_function is not None:
        # get mean only list
        if fit_fuction_by=='mean':
            y_s = []
            for y_bin in y_binned:
                y_s.append(np.nanmean(y_bin))
        elif fit_fuction_by=='median':
            y_s = []
            for y_bin in y_binned:
                y_s.append(np.nanmedian(y_bin))
        elif fit_fuction_by=='max':
            y_s = []
            for y_bin in y_binned:
                y_s.append(np.nanmax(y_bin))
        elif fit_fuction_by=='min':
            y_s = []
            for y_bin in y_binned:
                y_s.append(np.nanmin(y_bin))
        else:
            print('error, only possible fit_by are mean, median, max, min')
            return

        x_,y_= coincidence(x_binned_arr,y_s)

        # axes labels
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

        if log_x:
            ax.set_xscale("log")  # , nonposy='clip')
        if log_y:
            ax.set_yscale("log")  # , nonposy='clip')


        params = curve_fit(fit_function, x_, y_)
        print('fitted parameters')
        print('%0.3f, %0.3f' % (params[0][0], params[0][1]))

        # calculate R2
        plotting_par_list = [x_]
        params_str_ = ''
        for i_, fit_par in enumerate(params[0]):
            plotting_par_list.append(fit_par)
        funt_par = tuple(plotting_par_list)
        fit_line_y = fit_function(*funt_par)
        residuals = y_ - fit_line_y
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_ - np.mean(y_))**2)
        Rsqr = float(1 - (ss_res / ss_tot))
        print('R2 = %0.2f' % Rsqr)

        fit_line_x = np.arange(0,int(np.max(x_))+1,.1)
        plotting_par_list = [fit_line_x]
        for fit_par in params[0]:
            plotting_par_list.append(fit_par)
        funt_par = tuple(plotting_par_list)
        fit_line_y = fit_function(*funt_par)
        # fit_line_y =  (a_ * (fit_line_x ** 3)) + (b_ * (fit_line_x ** 2)) + (c_ * fit_line_x) + d_

        ax.plot(fit_line_x,fit_line_y,'k')
        # ax.yaxis.set_ticks(np.arange(0, 2800, 200))

        for i in range(len(x_)):
            print('%0.2f, %0.2f' % (x_[i], y_[i]))

        print('-' * 20)

    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)


    plt.show()

    medians_ = []
    for i_ in box_dict['medians']:
        medians_.append(i_.get_ydata()[0])
    medians_ = np.array(medians_)

    means_ = []
    for i_ in box_dict['means']:
        means_.append(i_.get_ydata()[0])
    means_ = np.array(means_)


    return fig, ax, box_dict, x_binned_arr, medians_, means_
def plot_diurnal_multi(values_array, header_array, x_index, y_index_list,add_line=None, median_=False,
                       bin_size=1, min_bin_population=10, legend_= True, y_label='',legend_loc=(.70,.80),
                       custom_y_range_tuple=None, custom_x_range_tuple=None, lw_=2,
                       return_stats=False, print_stats=False):
    color_list = default_cm(np.linspace(0,1,len(y_index_list)))
    # stats holder
    stats_list_x = []
    stats_list_y = []
    # start figure
    fig, ax = plt.subplots()#figsize=(16, 10))
    for c_, parameter_index in enumerate(y_index_list):
        color_ = color_list[c_]
        x_val_original = values_array[:,x_index]
        y_val_original = values_array[:,parameter_index]

        # get coincidences only
        x_val,y_val = coincidence(x_val_original,y_val_original)

       # combine x and y in matrix
        M_ = np.column_stack((x_val,y_val))
        # checking if always ascending to increase efficiency
        always_ascending = 1
        for x in range(x_val.shape[0]-1):
            if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
                if x_val[x+1] < x_val[x]:
                    always_ascending = 0
        if always_ascending == 0:
            M_sorted = M_[M_[:,0].argsort()] # sort by first column
            M_ = M_sorted
        # convert data to list of bins
        y_binned = []
        x_binned = []
        start_bin_edge = np.nanmin(x_val)
        last_row = 0
        last_row_temp = last_row
        while start_bin_edge <= np.nanmax(x_val):
            y_val_list = []
            for row_ in range(last_row, M_.shape[0]):
                if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                    if M_[row_, 1] == M_[row_, 1]:
                        y_val_list.append(M_[row_, 1])
                        last_row_temp = row_
                if M_[row_, 0] >= start_bin_edge + bin_size:
                    last_row_temp = row_
                    break
            x_binned.append(start_bin_edge)
            if len(y_val_list) >= min_bin_population:
                y_binned.append(y_val_list)
            else:
                y_binned.append([])
            start_bin_edge += bin_size
            last_row = last_row_temp
        # if bin_size >= 1:
        #     x_binned_int = np.array(x_binned, dtype=int)
        # else:
        #     x_binned_int = x_binned

        # get mean only list
        y_means = []
        for y_bin in y_binned:
            if median_:
                y_means.append(np.median(y_bin))
            else:
                y_means.append(np.mean(y_bin))

        x_,y_= coincidence(np.array(x_binned),np.array(y_means))

        # store stats
        stats_list_x.append(x_)
        stats_list_y.append(y_)

        # print x and y
        if print_stats:
            print(header_array[parameter_index])
            for i in range(len(x_)):
                print(x_[i],y_[i])
            print('-' * 10)
        # add means series
        ax.plot(x_, y_, color=color_, label=header_array[parameter_index], lw=lw_)

    # axes labels
    ax.set_xlabel(header_array[x_index])
    ax.set_ylabel(y_label)
    if legend_: ax.legend(loc=legend_loc)
    ax.xaxis.set_ticks(np.arange(0, 24, 3))
    #
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)

    if add_line is not None:
        ax.plot(add_line[0], add_line[1], color='black', label=add_line[2], lw=lw_)
    #
    plt.show()

    if return_stats:
        return stats_list_x, stats_list_y
def plot_diurnal_multi_wind_direction(header_array, time_array_list, wd_ws_list_list,
                       bin_size=1, min_bin_population=10, legend_= True, y_label='', x_label='',legend_loc='best',
                       custom_y_range_tuple=None, custom_x_range_tuple=None, lw_=0, size_=5):
    color_list = default_cm(np.linspace(0,1,len(time_array_list)))
    # start figure
    fig, ax = plt.subplots()#figsize=(16, 10))
    for c_ in range(len(time_array_list)):
        color_ = color_list[c_]
        x_val_original = time_array_list[c_]
        wd_val_original = wd_ws_list_list[c_][0]
        ws_val_original = wd_ws_list_list[c_][1]

        # # get coincidences only
        # wd_val,ws_val = coincidence(wd_val_original,ws_val_original)

        North_, East_ = polar_to_cart(wd_val_original, ws_val_original)
        M_ = np.column_stack((North_,East_))

        Index_mean, Values_mean = mean_discrete(x_val_original, M_, bin_size, 0, min_data=min_bin_population)

        WD_mean, WS_mean = cart_to_polar(Values_mean[:,0], Values_mean[:,1])

        # add means series
        ax.scatter(Index_mean, WD_mean, s = size_, c=color_, label=header_array[c_], lw = lw_)

    # axes labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_ticks(np.arange(0, 361, 45))
    if legend_: ax.legend(loc=legend_loc)
    ax.xaxis.set_ticks(np.arange(0, 24, 3))
    #
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)
    #
    plt.show()
def fit_test_1(values_x, values_y, fit_func, x_label=None, y_label=None, bin_size=1,min_bin_population=10):
    x_val_original = values_x
    y_val_original = values_y

    # get coincidences only
    x_val,y_val = coincidence(x_val_original,y_val_original)

    # start figure
    fig, ax = plt.subplots()#figsize=(16, 10))

    # combine x and y in matrix
    M_ = np.column_stack((x_val,y_val))
    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(x_val.shape[0]-1):
        if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
            if x_val[x+1] < x_val[x]:
                always_ascending = 0
    if always_ascending == 0:
        M_sorted = M_[M_[:,0].argsort()] # sort by first column
        M_ = M_sorted
    # convert data to list of bins
    y_binned = []
    x_binned = []
    start_bin_edge = np.nanmin(x_val)
    last_row = 0
    last_row_temp = last_row
    while start_bin_edge <= np.nanmax(x_val):
        y_val_list = []
        for row_ in range(last_row, M_.shape[0]):
            if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                if M_[row_, 1] == M_[row_, 1]:
                    y_val_list.append(M_[row_, 1])
                    last_row_temp = row_
            if M_[row_, 0] >= start_bin_edge + bin_size:
                last_row_temp = row_
                break
        x_binned.append(start_bin_edge)
        if len(y_val_list) >= min_bin_population:
            y_binned.append(y_val_list)
        else:
            y_binned.append([])
        start_bin_edge += bin_size
        last_row = last_row_temp
    # if bin_size >= 1:
    #     x_binned_int = np.array(x_binned, dtype=int)
    # else:
    #     x_binned_int = x_binned

    # get mean only list
    y_means = []
    for y_bin in y_binned:
        y_means.append(np.mean(y_bin))

    x_,y_= coincidence(x_binned,y_means)

    # add means series
    ax.plot(x_, y_, 'rs')

    # axes labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    params = curve_fit(fit_func, x_, y_)
    print(params[0])

    fit_line_x = np.arange(0,int(np.max(x_))+1,.1)
    plotting_par_list = [fit_line_x]
    for fit_par in params[0]:
        plotting_par_list.append(fit_par)
    funt_par = tuple(plotting_par_list)
    fit_line_y = fit_func(*funt_par)
    # fit_line_y =  (a_ * (fit_line_x ** 3)) + (b_ * (fit_line_x ** 2)) + (c_ * fit_line_x) + d_

    ax.plot(fit_line_x,fit_line_y,'k')
    # ax.yaxis.set_ticks(np.arange(0, 2800, 200))

    for i in range(len(x_)):
        print(x_[i],y_[i])

    print('-' * 20)

    #
    plt.show()
def plot_diurnal_multi_cumulative(values_array, header_array, x_index, y_index_ordered_list, alpha_=.5,add_line=None,
                                  bin_size=1, min_bin_population=10, legend_=True, y_label='',legend_loc='best',
                                  custom_color_list=None, custom_y_range_tuple=None, custom_x_range_tuple = None):
    if custom_color_list is not None:
        color_list = custom_color_list
    else:
        color_list = default_cm(np.linspace(0,1,len(y_index_ordered_list)))

    # start figure
    fig, ax = plt.subplots()#figsize=(16, 10))
    c_, parameter_index = 0, y_index_ordered_list[0]
    color_ = color_list[c_]
    x_val_original = values_array[:,x_index]
    y_val_original = values_array[:,parameter_index]
    # get coincidences only
    x_val,y_val = coincidence(x_val_original,y_val_original)
    # combine x and y in matrix
    M_ = np.column_stack((x_val,y_val))
    # checking if always ascending to increase efficiency
    always_ascending = 1
    for x in range(x_val.shape[0]-1):
        if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
            if x_val[x+1] < x_val[x]:
                always_ascending = 0
    if always_ascending == 0:
        M_sorted = M_[M_[:,0].argsort()] # sort by first column
        M_ = M_sorted
    # convert data to list of bins
    y_binned = []
    x_binned = []
    start_bin_edge = np.nanmin(x_val)
    last_row = 0
    last_row_temp = last_row
    while start_bin_edge <= np.nanmax(x_val):
        y_val_list = []
        for row_ in range(last_row, M_.shape[0]):
            if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                if M_[row_, 1] == M_[row_, 1]:
                    y_val_list.append(M_[row_, 1])
                    last_row_temp = row_
            if M_[row_, 0] >= start_bin_edge + bin_size:
                last_row_temp = row_
                break
        x_binned.append(start_bin_edge)
        if len(y_val_list) >= min_bin_population:
            y_binned.append(y_val_list)
        else:
            y_binned.append([])
        start_bin_edge += bin_size
        last_row = last_row_temp
    # if bin_size >= 1:
    #     x_binned_int = np.array(x_binned, dtype=int)
    # else:
    #     x_binned_int = x_binned
    # get mean only list
    y_means = []
    for y_bin in y_binned:
        y_means.append(np.mean(y_bin))
    # add means series
    # ax.plot(x_, y_, color=color_, label=header_array[parameter_index])
    ax.fill_between(x_binned, y_means, color=color_, label=header_array[parameter_index])
    # ax.plot(x_binned, y_means, color=color_, label=header_array[parameter_index], lw=2)

    if len(y_index_ordered_list) > 1:
        for c_ in range(len(y_index_ordered_list[1:])):
            parameter_index = y_index_ordered_list[c_ + 1]
            color_ = color_list[c_ + 1]
            x_val_original = values_array[:,x_index]
            y_val_original = values_array[:,parameter_index]
            # get coincidences only
            x_val,y_val = coincidence(x_val_original,y_val_original)
            # combine x and y in matrix
            M_ = np.column_stack((x_val,y_val))
            # checking if always ascending to increase efficiency
            always_ascending = 1
            for x in range(x_val.shape[0]-1):
                if x_val[x]==x_val[x] and x_val[x+1]==x_val[x+1]:
                    if x_val[x+1] < x_val[x]:
                        always_ascending = 0
            if always_ascending == 0:
                M_sorted = M_[M_[:,0].argsort()] # sort by first column
                M_ = M_sorted
            # convert data to list of bins
            y_binned = []
            x_binned = []
            start_bin_edge = np.nanmin(x_val)
            last_row = 0
            last_row_temp = last_row
            while start_bin_edge <= np.nanmax(x_val):
                y_val_list = []
                for row_ in range(last_row, M_.shape[0]):
                    if start_bin_edge <= M_[row_, 0] < start_bin_edge + bin_size:
                        if M_[row_, 1] == M_[row_, 1]:
                            y_val_list.append(M_[row_, 1])
                            last_row_temp = row_
                    if M_[row_, 0] >= start_bin_edge + bin_size:
                        last_row_temp = row_
                        break
                x_binned.append(start_bin_edge)
                if len(y_val_list) >= min_bin_population:
                    y_binned.append(y_val_list)
                else:
                    y_binned.append([])
                start_bin_edge += bin_size
                last_row = last_row_temp
            # if bin_size >= 1:
            #     x_binned_int = np.array(x_binned, dtype=int)
            # else:
            #     x_binned_int = x_binned
            # get mean only list
            y_means_previous = y_means
            y_means = []
            for i_, y_bin in enumerate(y_binned):
                y_means.append(np.mean(y_bin)+y_means_previous[i_])
            # add means series
            # ax.plot(x_, y_, color=color_, label=header_array[parameter_index])
            ax.fill_between(x_binned, y_means, y_means_previous,
                            color=color_, label=header_array[parameter_index],alpha = alpha_)
            # ax.plot(x_binned, y_means, color=color_, label=header_array[parameter_index], lw=2)

    # axes labels
    ax.set_xlabel(header_array[x_index])
    ax.set_ylabel(y_label)
    if add_line is not None:
        ax.plot(add_line[0], add_line[1], color='black', label=add_line[2],lw=10)
    if legend_: ax.legend(loc=legend_loc)
    ax.xaxis.set_ticks(np.arange(0, 24, 3))
    #
    if custom_y_range_tuple is not None: ax.set_ylim(custom_y_range_tuple)
    if custom_x_range_tuple is not None: ax.set_xlim(custom_x_range_tuple)


    plt.show()


# polars
def plot_wind_rose(parameter_name,wd_,va_):
    # convert data to mean, 25pc, 75pc
    wd_off = np.array(wd_)
    for i,w in enumerate(wd_):
        if w > 360-11.25:
            wd_off [i] = w - 360 #offset wind such that north is correct
    # calculate statistical distribution per wind direction bin
    # wd_bin, ws_bin_mean, ws_bin_25, ws_bin_75
    table_ = np.column_stack((median_discrete(wd_off, va_, 22.5, 0, position_=.5)))
    # repeating last value to close lines
    table_ = np.row_stack((table_,table_[0,:]))

    # start figure
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': 'polar'})
    # ax = plt.subplot(projection='polar')
    wd_rad = np.radians(table_[:,0])
    # format chart
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta_angles = np.arange(0, 360, 45)
    theta_labels = ['N', 'N-E','E','S-E', 'S', 'S-W', 'W', 'N-W']
    ax.set_thetagrids(angles=theta_angles, labels=theta_labels)
    # add series
    ax.plot(wd_rad, table_[:,1], 'ko-', linewidth=3, label = 'Median')
    ax.plot(wd_rad, table_[:,2], 'b-', linewidth=3, label = '25 percentile')
    ax.plot(wd_rad, table_[:,3], 'r-', linewidth=3, label = '75 percentile')
    ax.legend(title=parameter_name, loc=(1,.75))

    plt.show()
def plot_scatter_polar(parameter_name,WD_,Y_,C_,file_name=None):
    # start figure
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax = plt.subplot(projection='polar')
    WD_rad = np.radians(WD_)
    # format chart
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta_angles = np.arange(0, 360, 45)
    theta_labels = ['N', 'N-E','E','S-E', 'S', 'S-W', 'W', 'N-W']
    ax.set_thetagrids(angles=theta_angles, labels=theta_labels)
    # add series
    ax.scatter(WD_rad, Y_, c = C_, s=5, lw = 0, label=parameter_name)
    # color_bar = fig.colorbar(im,fraction=0.046, pad=0.08)
    # if c_header != None: color_bar.ax.set_ylabel(c_header)
    # ax.legend(loc=(-0.1,.95))
    ax.set_ylim(0,10)

    if file_name is None:
        plt.show()
    else:
        fig.savefig(path_output + '/' + 'polar_scatter_' + file_name + '.png',transparent=True, bbox_inches='tight')

# fitting functions
def linear_1_slope(x,b):
    return x + b
def hcho_fitting_2(M, a, b, c, d, e, f):
    co = M[:,0]
    o3 = M[:,1]
    so2 = M[:,2]
    no = M[:,3]
    no2 = M[:,4]

    hcho_calc = a*co + b*o3 + c*so2 + d*no + e*no2 + f
    return hcho_calc
def hcho_fitting_1(M, a, b, c, d):
    co = M[:,0]
    o3 = M[:,1]
    so2 = M[:,2]

    hcho_calc = a*co + b*o3 + c*so2 + d
    return hcho_calc
def hcho_fitting_0(M, a, b, c, d):
    s1 = M[:,0]
    s2 = M[:,1]
    s3 = M[:,2]

    return a*s1 + b*s2 + c*s3 + d
def polynomial_function_3(x,a,b,c,d):
    return a*(x**3) + b*(x**2) + c*(x**1) + d
def polynomial_function_2(x,a,b,c):
    return a*(x**2) + b*(x**1) + c
def exponential_function(x,a,b):
    return a * e_constant**(b * x)
def exponential_with_background_function(x,a,b,c):
    return (a * e_constant**(b * x)) + c
def sigmoid_for_soiling(pm_, rh_, a_, b_):
    return pm_ / (1 + (e_constant**(a_ * (rh_ + b_))))
def sigmoid_for_soiling_mod_1(pm_, rh_, rh_slope, rh_inflexion, pm_slope, pm_inflexion):
    rh_stickiness_ratio =  pm_ / (1 + (e_constant ** (rh_slope * (rh_ + rh_inflexion))))

    residual_pm = pm_ - rh_stickiness_ratio

    pm_gravity_deposition_ratio =  residual_pm / (1 + (e_constant ** (pm_slope * (residual_pm + pm_inflexion))))

    return pm_gravity_deposition_ratio + rh_stickiness_ratio
def modified_sigmoid(rh_, pm_, a_, b_, c_, d_):
    or_ = pm_ / (1 + (e_constant**(a_ * (rh_ + b_))))
    mod_ = (or_*(1- c_)*(1-d_))+ d_

    # out_ = 100 * (mod_/pm_)

    return mod_
def modified_sigmoid_2(rh_, pm_, a_, b_, c_, d_):
    # or_ = pm_ / (1 + (e_constant**(a_ * (rh_ + b_))))
    # mod_ = (or_ * (pm_ - (pm_*c_)) / pm_) + (pm_ * c_)
    # return mod_

    sig_ratio = 1 / (1 + (e_constant**(a_ * (rh_ + b_))))
    min_scale = pm_ * c_
    max_scale = ((1-d_-c_)*pm_)/pm_

    return pm_ * sig_ratio * max_scale + min_scale
def modified_sigmoid_2_for_fitting(rh_, pm_, a_, b_, c_):
    or_ = pm_ / (1 + (e_constant**(a_ * (rh_ + b_))))

    mod_ = (or_ * (pm_ - (pm_*c_)) / pm_) + (pm_ * c_)

    return mod_
def gaussian_func(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def DSD_gamma_dist_1(D, N_o_star, U_o, D_o):
    N_D = N_o_star * \
          ((math.gamma(4) * ((3.67 + U_o) ** (4 + U_o))) / ((3.67 ** 4) * math.gamma(4 + U_o))) * \
          ((D / D_o) ** U_o) * \
          np.exp(-(3.67 + U_o) * (D / D_o))
    return N_D
def SR_Ze_func(Ze_,a,b):
    SR_ = ((Ze_/a))**(1/b)
    return SR_

def Ze_SR_func(SR_,a,b):
    Ze_ = a * (SR_**b)
    return Ze_


p = p_

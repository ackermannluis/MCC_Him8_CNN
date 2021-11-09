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
from  U_Analysis_nci import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import CSVLogger
from mpl_toolkits.basemap import Basemap
import warnings
from scipy import ndimage
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import platform
from numpy.lib.stride_tricks import as_strided


# <editor-fold desc="tools, functions, and other background commands">
warnings.filterwarnings("ignore")
plt.style.use('classic')

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

path_program = os.path.dirname(os.path.realpath(sys.argv[0])) + '\\'

colorbar_tick_labels_list_MCC = ['N/A', 'Closed MCC', 'Open MCC']
listed_cm_colors_list_MCC = ['none', 'red', 'blue']
listed_cm_MCC = ListedColormap(listed_cm_colors_list_MCC, 'indexed')

default_cmap = cm.jet

colorbar_tick_labels_list_MCC_2 = ['Unknown', 'N/A', 'Closed MCC', 'Open MCC']
listed_cm_colors_list_MCC_2 = ['green', 'none', 'red', 'blue']
listed_cm_MCC_2 = ListedColormap(listed_cm_colors_list_MCC_2, 'indexed')




# misc
class Object_create(object):
    pass
def bell_alarm():
    if platform.platform()[0] == "W":
        import winsound
        winsound.MessageBeep()
    else:
        os.system('play -nq -t alsa synth {} sine {}'.format(1, 440))
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
def onclick_x_y(event):
    print(event.xdata, ',', event.ydata)



# def onclick_x_y_coeff_key(event):
#     global coeffs_dict, last_event_dict, figure_line_list
#
#     if event.key == 'backspace':
#         lines_name = last_event_dict['lines_name']
#         current_coef_list = list(coeffs_dict[lines_name])
#         print('deleted\t', lines_name, current_coef_list[-1])
#         del current_coef_list[-1]
#         coeffs_dict[lines_name] = current_coef_list
#
#         # delete line from figure
#         figure_line_list[-1].pop(0).remove()
#         del figure_line_list[-1]
# def onclick_x_y_coeff_press(event):
#     global coeffs_dict, last_event_dict
#     last_event_dict['x_1'] = event.xdata
#     last_event_dict['y_1'] = event.ydata
# def onclick_x_y_coeff_release(event):
#     global coeffs_dict, last_event_dict, figure_line_list
#
#     button_type = event.button
#     if str(button_type) == 'MouseButton.LEFT':
#         top_bottom_var = 0
#     elif str(button_type) == 'MouseButton.RIGHT':
#         top_bottom_var = 1
#     else:
#         print(str(button_type))
#         print('canceling')
#         return
#
#     x_header = str(event.inaxes.xaxis.get_label()).split("'")[-2]
#     y_header = str(event.inaxes.yaxis.get_label()).split("'")[-2]
#     mcc_type = fig.texts[0].get_text()
#
#     last_event_dict['x_2'] = event.xdata
#     last_event_dict['y_2'] = event.ydata
#
#     last_event_dict['x_header'] = x_header
#     last_event_dict['y_header'] = y_header
#
#     # arrange x ascending
#     arr_ = array_2D_sort_ascending_by_column(np.column_stack((
#         np.array([last_event_dict['x_1'], last_event_dict['x_2']]),
#         np.array([last_event_dict['y_1'], last_event_dict['y_2']]))))
#
#     coeffs = np.polyfit(arr_[:,0], arr_[:,1], 1)
#     intercept = coeffs[-1]
#     slope = coeffs[-2]
#
#     lines_name = 'lines_' + mcc_type +'_'+ x_header +'_'+ y_header # lines_open__mOT_vOT
#     last_event_dict['lines_name'] = lines_name
#
#     if lines_name in coeffs_dict.keys():
#         current_coef_list = list(coeffs_dict[lines_name])
#         current_coef_list.append((slope, intercept, top_bottom_var))
#         coeffs_dict[lines_name] = current_coef_list
#     else:
#         coeffs_dict[lines_name] = [(slope, intercept, top_bottom_var)]
#
#     print(lines_name, coeffs_dict[lines_name][-1])
#
#
#     # draw line
#     x_1,x_2,y_1,y_2 = get_chart_range(event.inaxes)
#     x_ = np.array([x_1,x_2])
#     y_ = line_function(x_, slope, intercept)
#     if top_bottom_var == 0:
#         color_ = 'k'
#     else:
#         color_ = 'b'
#     figure_line_list.append(event.inaxes.plot(x_, y_, color_))
#     event.inaxes.set_xlim((x_1,x_2))
#     event.inaxes.set_ylim((y_1,y_2))



# time transforms


# array transforms
def sliding_window_view(arr, window_shape, steps):
    """ Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
        Notes
        -----
        In general, given
          `out` = sliding_window_view(arr,
                                      window_shape=[Wx, (...), Wz],
                                      steps=[Sx, (...), Sz])
           out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]
         Examples
         --------
         >>> import numpy as np
         >>> x = np.arange(9).reshape(3,3)
         >>> x
         array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
         >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
         >>> y
         array([[[[0, 1],
                  [3, 4]],
                 [[1, 2],
                  [4, 5]]],
                [[[3, 4],
                  [6, 7]],
                 [[4, 5],
                  [7, 8]]]])
        >>> np.shares_memory(x, y)
         True
        # Performing a neural net style 2D conv (correlation)
        # placing a 4x4 filter with stride-1
        >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
        >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
        >>> windowed_data = sliding_window_view(data,
        ...                                     window_shape=(4, 4),
        ...                                     steps=(1, 1))
        >>> conv_out = np.tensordot(filters,
        ...                         windowed_data,
        ...                         axes=[[1,2,3], [3,4,5]])
        # (F, H', W', N) -> (N, F, H', W')
        >>> conv_out = conv_out.transpose([3,0,1,2])
         """
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)
def strides_2d(a, r, linear=True):
    ax = np.zeros(shape=(a.shape[0] + 2 * r[0], a.shape[1] + 2 * r[1]))
    ax[:] = np.nan
    ax[r[0]:ax.shape[0] - r[0], r[1]:ax.shape[1] - r[1]] = a

    shape = a.shape + (1 + 2 * r[0], 1 + 2 * r[1])
    strides = ax.strides + ax.strides
    s = as_strided(ax, shape=shape, strides=strides)

    return s.reshape(a.shape + (shape[2] * shape[3],)) if linear else s
def windowed_sum(a, win, keep_dims=True):
    table = np.nancumsum(np.nancumsum(a, axis=0), axis=1)
    win_sum = np.empty(tuple(np.subtract(a.shape, win - 1)))
    win_sum[0, 0] = table[win - 1, win - 1]
    win_sum[0, 1:] = table[win - 1, win:] - table[win - 1, :-win]
    win_sum[1:, 0] = table[win:, win - 1] - table[:-win, win - 1]
    win_sum[1:, 1:] = (table[win:, win:] + table[:-win, :-win] -
                       table[win:, :-win] - table[:-win, win:])
    if keep_dims:
        if (win % 2) == 0:
            print('error, cannot keep size if moving window size is even')
            return None
        arr_out = np.zeros((a.shape)) * np.nan
        arr_out[int(win/2):int(win/2) + win_sum.shape[0], int(win/2):int(win/2) + win_sum.shape[1]] = win_sum
        return arr_out

    else:
        return win_sum
def windowed_var(a, win):
    win_a = windowed_sum(a, win)
    win_a2 = windowed_sum(a * a, win)
    return (win_a2 - win_a * win_a / win / win) / win / win
def windowed_mean(a, win):
    return windowed_sum(a, win)/(win**2)

# saving and plotting
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
def plot_arr_over_map(arr_, lat_arr, lon_arr, resolution_='i', format_='%.2f', cbar_label='', cmap_=default_cmap,
                      map_pad=0, min_lat=None,max_lat=None,min_lon=None,max_lon=None, vmin_=None,vmax_=None,
                      save_fig=False, figure_filename='', projection_='merc', show_grid = False, grid_step=5,
                      coast_color='black', title_str = '', colorbar_tick_labels_list=None, return_traj=False,
                      figsize_= (10, 6), parallels_=None, meridians_=None, grid_line_width=1,font_size=14,
                      lcc_args=(-10, -30, -17.5, 140.7)):
    fig, ax = plt.subplots(figsize=figsize_)

    if min_lat is None: min_lat = np.nanmin(lat_arr)
    if max_lat is None: max_lat = np.nanmax(lat_arr)
    if min_lon is None: min_lon = np.nanmin(lon_arr)
    if max_lon is None: max_lon = np.nanmax(lon_arr)

    if vmin_ is None: vmin_ = np.nanmin(arr_)
    if vmax_ is None: vmax_ = np.nanmax(arr_)

    llcrnrlat_ = min_lat - ((max_lat - min_lat) * map_pad)
    urcrnrlat_ = max_lat + ((max_lat - min_lat) * map_pad)
    llcrnrlon_ = min_lon - ((max_lon - min_lon) * map_pad)
    urcrnrlon_ = max_lon + ((max_lon - min_lon) * map_pad)

    if llcrnrlat_ < -90 : llcrnrlat_ = -89
    if urcrnrlat_ > 90: urcrnrlat_ = 89
    if llcrnrlon_ < -180: llcrnrlon_ = -179
    if urcrnrlon_ > 180: urcrnrlon_ = 179


    if projection_ == 'merc':
        m = Basemap(projection='merc',
                    llcrnrlat = llcrnrlat_, urcrnrlat = urcrnrlat_,
                    llcrnrlon = llcrnrlon_, urcrnrlon = urcrnrlon_,
                    resolution=resolution_)
    elif projection_ == 'lcc':
        m = Basemap(projection='lcc',
                    llcrnrlat = llcrnrlat_, urcrnrlat = urcrnrlat_,
                    llcrnrlon = llcrnrlon_, urcrnrlon = urcrnrlon_,
                    resolution=resolution_,
                    lat_1=lcc_args[0], lat_2=lcc_args[1], lat_0=-lcc_args[2], lon_0=lcc_args[3])
    elif projection_ == 'geos':
        m = Basemap(projection='geos',
                    rsphere=(6378137.00, 6356752.3142),
                    resolution=resolution_,
                    llcrnrlat = llcrnrlat_, urcrnrlat = urcrnrlat_,
                    llcrnrlon = llcrnrlon_, urcrnrlon = urcrnrlon_,
                    lon_0=140.7,
                    satellite_height=35785831)
    else:
        m = Basemap(projection=projection_,
                    llcrnrlat = llcrnrlat_, urcrnrlat = urcrnrlat_,
                    llcrnrlon = llcrnrlon_, urcrnrlon = urcrnrlon_,
                    resolution=resolution_,
                    lat_1=-10., lat_2=-30, lat_0=-17.5, lon_0=140.7)


    m.drawcoastlines(color=coast_color)
    m.drawcountries()

    if show_grid:
        if parallels_ is None:
            parallels_ = np.arange(min_lat, max_lat, grid_step)
        if meridians_ is None:
            meridians_ = np.arange(min_lon, max_lon, grid_step)

        m.drawparallels(parallels_, labels=[True, False, False, False],  linewidth=grid_line_width)
        m.drawmeridians(meridians_, labels=[False, False, False, True],  linewidth=grid_line_width)

    if len(lon_arr.shape) == 1:
        array_x_reshaped = np.zeros((arr_.shape[0], arr_.shape[1]), dtype=float)
        for r_ in range(arr_.shape[0]):
            array_x_reshaped[r_, :] = lon_arr
    else:
        array_x_reshaped = lon_arr

    if len(lat_arr.shape) == 1:
        array_y_reshaped = np.zeros((arr_.shape[0], arr_.shape[1]), dtype=float)
        for c_ in range(arr_.shape[1]):
            array_y_reshaped[:, c_] = lat_arr
    else:
        array_y_reshaped = lat_arr

    lat_arr = array_y_reshaped
    lon_arr = array_x_reshaped
    x, y = m(lon_arr, lat_arr)

    trajs_ = ax.pcolormesh(x, y, arr_, cmap=cmap_, vmin=vmin_, vmax=vmax_)

    color_bar = m.colorbar(trajs_, pad="5%", format=format_)
    color_bar.ax.set_ylabel(cbar_label)
    color_bar.ax.tick_params(labelsize=font_size)

    if colorbar_tick_labels_list is not None:
        ticks_ = np.linspace(0.5, len(colorbar_tick_labels_list) - 0.5, len(colorbar_tick_labels_list))
        color_bar.set_ticks(ticks_)
        color_bar.set_ticklabels(colorbar_tick_labels_list)

    if title_str != '':
        ax.set_title(title_str)

    ax.format_coord = lambda x_fig, y_fig: 'x=%g, y=%g, v=%g' % (
        lon_arr.flatten()[np.argmin(np.abs(x - x_fig))],
        lat_arr.flatten()[np.argmin(np.abs(y - y_fig))],
        arr_.flatten()[int(np.argmin(np.abs(x - x_fig) ** 2 + np.abs(y - y_fig) ** 2))])

    if save_fig or figure_filename != '':
        if figure_filename == '':
            name_ = str(calendar.timegm(time.gmtime()))[:-2]
            fig.savefig(path_output + 'image_' + name_ + '.png', transparent=True, bbox_inches='tight')
        else:
            fig.savefig(figure_filename, transparent=True, bbox_inches='tight')

        plt.close(fig)
    else:
        plt.show()

    if return_traj:
        return fig, ax, m, trajs_
    else:
        return fig, ax, m
def p_arr(A_, cmap_=default_cmap, extend_x1_x2_y1_y2 =(0,1), figsize_= (10, 6), aspect_='auto', rot_=0, title_str = '',
          vmin_=None, vmax_=None, cbar_label = '', x_as_time=False, time_format_='%H:%M %d%b%y', save_fig=False,
          figure_filename='', x_header='',y_header='', x_ticks_tuple=None, y_ticks_tuple=None, fig_ax=None,
          origin_='upper', colorbar_tick_labels_list=None, tick_label_format='plain', tick_offset=False,
          show_cbar=True):
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

    if show_cbar:
        color_bar = fig.colorbar(img_)
        color_bar.ax.set_ylabel(cbar_label)
    else:
        color_bar = None

    if show_cbar and colorbar_tick_labels_list is not None:
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
def get_chart_range(ax):
    x_1 = ax.axis()[0]
    x_2 = ax.axis()[1]
    y_1 = ax.axis()[2]
    y_2 = ax.axis()[3]
    return x_1,x_2,y_1,y_2

# </editor-fold>


p = p_
################################################################ user defined variables #################
path_input = 'C:\\_input\\'        # where the himawari-8 NC files and line filters are located         #
path_output = 'C:\\_output\\'      # where the filtered arrays will be saved                            #
path_data = 'C:\\_input\\DIC\\'      # where the filtered arrays will be saved                            #

# file_list = list_files_recursive(path_input, 'NC_')                                                     #
# MCC_index_OPEN__dict = np.load(path_data + 'MCC_index_OPEN_dict.npy', allow_pickle=True).item()         #
# MCC_index_CLOSE_dict = np.load(path_data + 'MCC_index_CLOSE_dict.npy', allow_pickle=True).item()        #
# MCC_index_NONE__dict = np.load(path_data + 'MCC_index_NONE_dict.npy', allow_pickle=True).item()         #
# keys_ = sorted(MCC_index_OPEN__dict)                                                                    #
# cmap_ = plt.cm.Greys_r                                                                                  #
# vmin_ = .5                                                                                              #
# vmax_ = 6                                                                                               #
################################################################ user defined variables #################


################################################################################################################


# Training with 16 by 16 ch11 with solar and satellite angles (SOA, SOZ, SAA, SAZ)


# <editor-fold desc="create selected numpy arrays from nc files, and store to disk">
file_list = list_files_recursive('C:\\_input\\MCC_new_training_20210630\\', 'NC_H08_')
selected_numpy_path_output = 'D:\\MCC\\Data_selection_Francisco\\update_20210703\\'

variables_to_save_nc_names = ['SOA', 'SOZ', 'tbb_10', 'tbb_11', 'tbb_13']
variables_to_save_np_names = ['SOA', 'SOZ', 'ch10', 'ch11', 'ch13']


for i_, filename_ in enumerate(file_list):
    p_progress_bar(i_, len(file_list))

    time_stamp_str = filename_[-37:-24]

    file_nc = nc.Dataset(filename_)

    for var_name_index in range(len(variables_to_save_nc_names)):
        filename_output = selected_numpy_path_output + 'NC_H08_' + time_stamp_str + '_' + \
                          variables_to_save_np_names[var_name_index] + '.npy'

        np.save(filename_output,file_nc.variables[variables_to_save_nc_names[var_name_index]][:].filled(np.nan))

    file_nc.close()
# </editor-fold>

# <editor-fold desc="update the training dictionaries 2021-04-01">
path_dicts_update = 'D:\\MCC\\Data_selection_Francisco\\update_20210703\\'
path_previous_dicts =  'D:\\MCC\\Data_selection_Francisco\\'
old_compiled_training_dictionary = path_previous_dicts + 'MCC_training_dict_20210401_V1.npy'
new_training_dict = np.load(old_compiled_training_dictionary, allow_pickle=True).item()

file_list_open = list_files_recursive(path_dicts_update, 'MCC_index_OPEN_dict')
file_list_close = list_files_recursive(path_dicts_update, 'MCC_index_CLOSE_dict')
file_list_none = list_files_recursive(path_dicts_update, 'MCC_index_NONE_dict')

for filename_ in file_list_open:
    MCC_index = np.load(filename_, allow_pickle=True).item()
    keys_ = sorted(MCC_index.keys())
    for key_ in keys_:
        date_time_str = key_[-37:-24]
        new_training_dict['OPEN_'][date_time_str] = MCC_index[key_]

for filename_ in file_list_close:
    MCC_index = np.load(filename_, allow_pickle=True).item()
    keys_ = sorted(MCC_index.keys())
    for key_ in keys_:
        date_time_str = key_[-37:-24]
        new_training_dict['CLOSE'][date_time_str] = MCC_index[key_]

for filename_ in file_list_none:
    MCC_index = np.load(filename_, allow_pickle=True).item()
    keys_ = sorted(MCC_index.keys())
    for key_ in keys_:
        date_time_str = key_[-37:-24]
        new_training_dict['NONE_'][date_time_str] = MCC_index[key_]

np.save(path_previous_dicts + 'MCC_training_dict_20210703_V1', new_training_dict)
# </editor-fold>

# <editor-fold desc="create training data with 16 by 16 ch11 with solar and satellite angles (SOA, SOZ, SAA, SAZ)">
arrays_path = 'D:\\MCC\\Data_selection_Francisco\\'
training_dict = np.load(arrays_path + 'MCC_training_dict_20210703_V1.npy', allow_pickle=True).item()
window_full = 16
window_cut= int(window_full/2)

date_time_str_open__list = sorted(training_dict['OPEN_'].keys())
date_time_str_close_list = sorted(training_dict['CLOSE'].keys())
date_time_str_none__list = sorted(training_dict['NONE_'].keys())

SAA_arr = np.load(arrays_path + 'SAA_arr.npy')
SAZ_arr = np.load(arrays_path + 'SAZ_arr.npy')

channel_number_str = '11'
CH_array_list_OPEN_ = []
CH_array_list_CLOSE = []
CH_array_list_NONE_ = []

AN_array_list_OPEN_ = []
AN_array_list_CLOSE = []
AN_array_list_NONE_ = []


# OPEN
for i_, date_time_str in enumerate(date_time_str_open__list):
    p_progress_bar(i_, len(date_time_str_open__list))

    filename_ch_ = list_files_recursive(arrays_path, date_time_str + '_ch' + channel_number_str)[0]
    filename_SOA = list_files_recursive(arrays_path, date_time_str + '_SOA')[0]
    filename_SOZ = list_files_recursive(arrays_path, date_time_str + '_SOZ')[0]

    CH_arr_sliced = sliding_window_view(np.load(filename_ch_), (window_full,window_full), (1,1))
    SOA_arr = np.load(filename_SOA)
    SOZ_arr = np.load(filename_SOZ)

    for index_ in training_dict['OPEN_'][date_time_str]:
        if np.min(index_) - window_cut >= 0 and np.max(index_) - window_cut < CH_arr_sliced.shape[0]:
            CH_array_list_OPEN_.append(CH_arr_sliced[index_[1] - window_cut, index_[0] - window_cut, :,:])

            AN_array_list_OPEN_.append(np.array([
                SAA_arr[index_[1], index_[0]],
                SAZ_arr[index_[1], index_[0]],
                SOA_arr[index_[1], index_[0]],
                SOZ_arr[index_[1], index_[0]]
            ]))

# CLOSE
for i_, date_time_str in enumerate(date_time_str_close_list):
    p_progress_bar(i_, len(date_time_str_close_list))

    filename_ch_ = list_files_recursive(arrays_path, date_time_str + '_ch' + channel_number_str)[0]
    filename_SOA = list_files_recursive(arrays_path, date_time_str + '_SOA')[0]
    filename_SOZ = list_files_recursive(arrays_path, date_time_str + '_SOZ')[0]

    CH_arr_sliced = sliding_window_view(np.load(filename_ch_), (window_full,window_full), (1,1))
    SOA_arr = np.load(filename_SOA)
    SOZ_arr = np.load(filename_SOZ)

    for index_ in training_dict['CLOSE'][date_time_str]:
        if np.min(index_) - window_cut >= 0 and np.max(index_) - window_cut < CH_arr_sliced.shape[0]:
            CH_array_list_CLOSE.append(CH_arr_sliced[index_[1] - window_cut, index_[0] - window_cut, :,:])

            AN_array_list_CLOSE.append(np.array([
                SAA_arr[index_[1], index_[0]],
                SAZ_arr[index_[1], index_[0]],
                SOA_arr[index_[1], index_[0]],
                SOZ_arr[index_[1], index_[0]]
            ]))

# NONE
for i_, date_time_str in enumerate(date_time_str_none__list):
    p_progress_bar(i_, len(date_time_str_none__list))

    filename_ch_ = list_files_recursive(arrays_path, date_time_str + '_ch' + channel_number_str)[0]
    filename_SOA = list_files_recursive(arrays_path, date_time_str + '_SOA')[0]
    filename_SOZ = list_files_recursive(arrays_path, date_time_str + '_SOZ')[0]

    CH_arr_sliced = sliding_window_view(np.load(filename_ch_), (window_full,window_full), (1,1))
    SOA_arr = np.load(filename_SOA)
    SOZ_arr = np.load(filename_SOZ)

    for index_ in training_dict['NONE_'][date_time_str]:
        if np.min(index_) - window_cut >= 0 and np.max(index_) - window_cut < CH_arr_sliced.shape[0]:
            CH_array_list_NONE_.append(CH_arr_sliced[index_[1] - window_cut, index_[0] - window_cut, :,:])

            AN_array_list_NONE_.append(np.array([
                SAA_arr[index_[1], index_[0]],
                SAZ_arr[index_[1], index_[0]],
                SOA_arr[index_[1], index_[0]],
                SOZ_arr[index_[1], index_[0]]
            ]))


CH_array_3D_OPEN_ = np.zeros((len(CH_array_list_OPEN_), window_full, window_full, 1), dtype=float)
CH_array_3D_CLOSE = np.zeros((len(CH_array_list_CLOSE), window_full, window_full, 1), dtype=float)
CH_array_3D_NONE_ = np.zeros((len(CH_array_list_NONE_), window_full, window_full, 1), dtype=float)

AN_array_2D_OPEN_ = np.zeros((len(CH_array_list_OPEN_), 4), dtype=float)
AN_array_2D_CLOSE = np.zeros((len(CH_array_list_CLOSE), 4), dtype=float)
AN_array_2D_NONE_ = np.zeros((len(CH_array_list_NONE_), 4), dtype=float)



for i_open_ in range(len(CH_array_list_OPEN_)):
    CH_array_3D_OPEN_[i_open_, :, :, 0] = CH_array_list_OPEN_[i_open_]
    AN_array_2D_OPEN_[i_open_, :]       = AN_array_list_OPEN_[i_open_]
for i_close in range(len(CH_array_list_CLOSE)):
    CH_array_3D_CLOSE[i_close, :, :, 0] = CH_array_list_CLOSE[i_close]
    AN_array_2D_CLOSE[i_close, :]       = AN_array_list_CLOSE[i_close]
for i_none_ in range(len(CH_array_list_NONE_)):
    CH_array_3D_NONE_[i_none_, :, :, 0] = CH_array_list_NONE_[i_none_]
    AN_array_2D_NONE_[i_none_, :]       = AN_array_list_NONE_[i_none_]


output_dict = {}
output_dict['CH_array_3D_OPEN_'] = CH_array_3D_OPEN_
output_dict['CH_array_3D_CLOSE'] = CH_array_3D_CLOSE
output_dict['CH_array_3D_NONE_'] = CH_array_3D_NONE_

output_dict['AN_array_2D_OPEN_'] = AN_array_2D_OPEN_
output_dict['AN_array_2D_CLOSE'] = AN_array_2D_CLOSE
output_dict['AN_array_2D_NONE_'] = AN_array_2D_NONE_


np.save(arrays_path + 'training_dataset_16win_ch11_angles_20210703_V1', output_dict)
# </editor-fold>


# <editor-fold desc="prepare data for training, i.e. crop, shuffle, normalize, cap, and stack">
normalization_ch11_offset   = -200
normalization_ch11_scale    = 130
normalization_SAA_offset    = 180
normalization_SAA_scale     = 360
normalization_SAZ_offset    = 0
normalization_SAZ_scale     = 90
normalization_SOA_offset    = 180
normalization_SOA_scale     = 360
normalization_SOZ_offset    = 0
normalization_SOZ_scale     = 180



MCC_training_path = 'D:\\MCC\\Data_selection_Francisco\\'
training_data_filename = MCC_training_path + 'training_dataset_16win_ch11_angles_20210703_V1.npy'

# load training data
dict_16win_ch = np.load(training_data_filename, allow_pickle=True).item()
p(dict_16win_ch)


CH_array_3D_OPEN_ = dict_16win_ch['CH_array_3D_OPEN_']
CH_array_3D_CLOSE = dict_16win_ch['CH_array_3D_CLOSE']
CH_array_3D_NONE_ = dict_16win_ch['CH_array_3D_NONE_']
AN_array_2D_OPEN_ = dict_16win_ch['AN_array_2D_OPEN_']
AN_array_2D_CLOSE = dict_16win_ch['AN_array_2D_CLOSE']
AN_array_2D_NONE_ = dict_16win_ch['AN_array_2D_NONE_']


# shuffle
seed = 777
CH_array_3D_OPEN__shuffle = np.array(CH_array_3D_OPEN_)
CH_array_3D_CLOSE_shuffle = np.array(CH_array_3D_CLOSE)
CH_array_3D_NONE__shuffle = np.array(CH_array_3D_NONE_)
AN_array_2D_OPEN__shuffle = np.array(AN_array_2D_OPEN_)
AN_array_2D_CLOSE_shuffle = np.array(AN_array_2D_CLOSE)
AN_array_2D_NONE__shuffle = np.array(AN_array_2D_NONE_)

np.random.seed(seed)
np.random.shuffle(CH_array_3D_OPEN__shuffle)
np.random.seed(seed)
np.random.shuffle(CH_array_3D_CLOSE_shuffle)
np.random.seed(seed)
np.random.shuffle(CH_array_3D_NONE__shuffle)
np.random.seed(seed)
np.random.shuffle(AN_array_2D_OPEN__shuffle)
np.random.seed(seed)
np.random.shuffle(AN_array_2D_CLOSE_shuffle)
np.random.seed(seed)
np.random.shuffle(AN_array_2D_NONE__shuffle)



# crop data so all classes are equally represented
maximum_sample_unbalance = 20
rows_min = np.min([
    CH_array_3D_OPEN_.shape[0],
    CH_array_3D_CLOSE.shape[0],
    CH_array_3D_NONE_.shape[0]
])

if CH_array_3D_OPEN_.shape[0] > rows_min * maximum_sample_unbalance:
    rows_crop_OPEN_ = rows_min * maximum_sample_unbalance
else:
    rows_crop_OPEN_ = CH_array_3D_OPEN_.shape[0]

if CH_array_3D_CLOSE.shape[0] > rows_min * maximum_sample_unbalance:
    rows_crop_CLOSE = rows_min * maximum_sample_unbalance
else:
    rows_crop_CLOSE = CH_array_3D_CLOSE.shape[0]

if CH_array_3D_NONE_.shape[0] > rows_min * maximum_sample_unbalance:
    rows_crop_NONE_ = rows_min * maximum_sample_unbalance
else:
    rows_crop_NONE_ = CH_array_3D_NONE_.shape[0]

CH_array_3D_OPEN__shuffle_cropped = CH_array_3D_OPEN__shuffle[:rows_crop_OPEN_]
AN_array_2D_OPEN__shuffle_cropped = AN_array_2D_OPEN__shuffle[:rows_crop_OPEN_]

CH_array_3D_CLOSE_shuffle_cropped = CH_array_3D_CLOSE_shuffle[:rows_crop_CLOSE]
AN_array_2D_CLOSE_shuffle_cropped = AN_array_2D_CLOSE_shuffle[:rows_crop_CLOSE]

CH_array_3D_NONE__shuffle_cropped = CH_array_3D_NONE__shuffle[:rows_crop_NONE_]
AN_array_2D_NONE__shuffle_cropped = AN_array_2D_NONE__shuffle[:rows_crop_NONE_]


# stack classes
rows_ = rows_crop_OPEN_ + rows_crop_CLOSE + rows_crop_NONE_

img_array = np.zeros((rows_, 16,16, 1), dtype=float)
img_array[:rows_crop_OPEN_, :, :, 0] = CH_array_3D_OPEN__shuffle_cropped[:,:,:,0]
img_array[rows_crop_OPEN_:rows_crop_OPEN_ + rows_crop_CLOSE, :, :, 0] = CH_array_3D_CLOSE_shuffle_cropped[:,:,:,0]
img_array[rows_crop_OPEN_ + rows_crop_CLOSE:, :, :, 0] = CH_array_3D_NONE__shuffle_cropped[:,:,:,0]

ang_array = np.zeros((rows_, 4), dtype=float)
ang_array[:rows_crop_OPEN_, :] = AN_array_2D_OPEN__shuffle_cropped[:,:]
ang_array[rows_crop_OPEN_:rows_crop_OPEN_ + rows_crop_CLOSE, :] = AN_array_2D_CLOSE_shuffle_cropped[:,:]
ang_array[rows_crop_OPEN_ + rows_crop_CLOSE:, :] = AN_array_2D_NONE__shuffle_cropped[:,:]


# normalyze
img_array = img_array + normalization_ch11_offset
img_array = img_array / normalization_ch11_scale

ang_array[:,0] = ang_array[:,0] + normalization_SAA_offset
ang_array[:,0] = ang_array[:,0] / normalization_SAA_scale

ang_array[:,1] = ang_array[:,1] + normalization_SAZ_offset
ang_array[:,1] = ang_array[:,1] / normalization_SAZ_scale

ang_array[:,2] = ang_array[:,2] + normalization_SOA_offset
ang_array[:,2] = ang_array[:,2] / normalization_SOA_scale

ang_array[:,3] = ang_array[:,3] + normalization_SOZ_offset
ang_array[:,3] = ang_array[:,3] / normalization_SOZ_scale


# delete invalids
img_array[np.isnan(img_array)] = 0
ang_array[np.isnan(ang_array)] = 0



# cap
img_array[img_array > 1] = 1
img_array[img_array < 0] = 0
ang_array[ang_array > 1] = 1
ang_array[ang_array < 0] = 0


# labels
cls_array = np.zeros(rows_, dtype=int)
cls_array[:rows_crop_OPEN_] = 2
cls_array[rows_crop_OPEN_:rows_crop_OPEN_ + rows_crop_CLOSE] = 1
cls_array[rows_crop_OPEN_ + rows_crop_CLOSE:] = 0


# shuffle
seed = 367
img_array_shuf = np.array(img_array)
ang_array_shuf = np.array(ang_array)
cls_array_shuf = np.array(cls_array)

np.random.seed(seed)
np.random.shuffle(img_array_shuf)
np.random.seed(seed)
np.random.shuffle(ang_array_shuf)
np.random.seed(seed)
np.random.shuffle(cls_array_shuf)



# save training ready shuffled, cropped data
training_ready_data_dict = {
    'img_array_shuf' : img_array_shuf,
    'ang_array_shuf' : ang_array_shuf,
    'cls_array_shuf' : cls_array_shuf
}
np.save(MCC_training_path + 'training_ready_data_dict_img_array_shuf_20210703_V1', img_array_shuf)
np.save(MCC_training_path + 'training_ready_data_dict_ang_array_shuf_20210703_V1', ang_array_shuf)
np.save(MCC_training_path + 'training_ready_data_dict_cls_array_shuf_20210703_V1', cls_array_shuf)

# np.save(MCC_training_path + 'training_ready_data_dict_16win_ch11_angles_20210224_V1', training_ready_data_dict)
# </editor-fold>


# <editor-fold desc="train model">
# load training data
MCC_training_path = 'D:\\MCC\\Data_selection_Francisco\\'
MCC_model_path = 'D:\\MCC\\'
trained_model_output = MCC_model_path + 'model_CH11_angles_20210703_V1\\'
csv_logger = CSVLogger(trained_model_output + "model_history_log.csv", append=True)
checkpoint_logger = tf.keras.callbacks.ModelCheckpoint(
                        trained_model_output, monitor='val_loss', verbose=0, save_best_only=False,
                        save_weights_only=True, mode='auto', save_freq='epoch',
                        options=None
                        )

# training_ready_data_filename = MCC_training_path + 'training_ready_data_dict_16win_ch11_angles_20210223_V1.npy'
# training_ready_data_dict = np.load(training_ready_data_filename, allow_pickle=True).item()
# img_array_shuf = training_ready_data_dict['img_array_shuf']
# ang_array_shuf = training_ready_data_dict['ang_array_shuf']
# cls_array_shuf = training_ready_data_dict['cls_array_shuf']

img_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_img_array_shuf_20210703_V1.npy')
ang_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_ang_array_shuf_20210703_V1.npy')
cls_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_cls_array_shuf_20210703_V1.npy')


validate_perc = .2
rows_ = img_array_shuf.shape[0]

val_rows = int(rows_ * validate_perc)
train_rows = rows_ - val_rows


# <editor-fold desc="create model structure (NEW more complex)">
# define two sets of inputs
input_1 = layers.Input(shape=(16, 16, 1))
input_2 = layers.Input(shape=(4,))

# images branch
x_1 = layers.Conv2D(28, (4, 4), activation='relu')(input_1)
x_1 = layers.MaxPooling2D((2, 2))(x_1)
x_1 = layers.Conv2D(32, (3, 3), activation='relu')(x_1)
x_1 = layers.MaxPooling2D((2, 2))(x_1)
x_1 = layers.Conv2D(32, (2, 2), activation='relu')(x_1)
x_1 = layers.Flatten()(x_1)
x_1 = models.Model(inputs=input_1, outputs=x_1)

# angles branch
x_2 = layers.Dense(16, activation="relu")(input_2)
x_2 = layers.Dense(6, activation="relu")(x_2)
x_2 = models.Model(inputs=input_2, outputs=x_2)

# combine the output of the two branches
combined = layers.concatenate([x_1.output, x_2.output])
combined = layers.Dense(32, activation='relu')(combined)
combined = layers.Dense(3)(combined)

# finalize model
NN_model = models.Model(inputs=[x_1.input, x_2.input], outputs=combined)

NN_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

NN_model.summary()
# </editor-fold>


#
# latest = tf.train.latest_checkpoint(MCC_model_path + 'model_CH11_angles_20210224_V1\\')
# NN_model.load_weights(latest)


t0 = time.time()
history = NN_model.fit([img_array_shuf[:train_rows], ang_array_shuf[:train_rows]],
                       cls_array_shuf[:train_rows],
                       epochs=50,
                       validation_data=([img_array_shuf[train_rows:], ang_array_shuf[train_rows:]],
                                        cls_array_shuf[train_rows:]),
                       callbacks=[csv_logger, checkpoint_logger],
                       )
print('model trained in', int((time.time() - t0)/60), 'minutes')
NN_model.save_weights(trained_model_output + 'MCC_NN_model_weights_1')



# print('-'*20)
# class_names = np.arange(3)
# print('ANS\t\tMOD\t\tConf%')
# for i_ in range(10):
#     predictions = NN_model.predict([np.array(img_array_shuf[i_,:,:,0]).reshape((1,16,16,1)),
#                                     np.array(ang_array_shuf[i_]).reshape((1,4)) ])
#     score = tf.nn.softmax(predictions[0,:])
#     print('{}\t\t{}\t\t{:.2f}'.format(cls_array_shuf[i_],class_names[np.argmax(score)],100 * np.max(score)))


# </editor-fold>


# <editor-fold desc="Training performance statistics">
sess = tf.Session()
MCC_training_path = 'D:\\MCC\\Data_selection_Francisco\\'
MCC_model_path = 'D:\\MCC\\'
model_name = '20210703_V1'

trained_model_output = MCC_model_path + 'model_CH11_angles_' + model_name + '\\'


# <editor-fold desc="training_data_monthly_daily_distribution">
# total number of training time stamps
training_time_stamps_list = list_files_recursive(MCC_training_path, '_ch11.npy')
list_of_times_file = open(path_output + 'training_time_stamps_list.csv', 'w')
for line_ in training_time_stamps_list:
    list_of_times_file.write(line_ + '\n')
list_of_times_file.close()

croped_list = []
for line_ in training_time_stamps_list:
    croped_list.append(line_[-22:-9])
list_of_times_file_unique = open(path_output + 'training_time_stamps_list_unique.csv', 'w')
for line_ in sorted(set(croped_list)):
    list_of_times_file_unique.write(line_ + '\n')
list_of_times_file_unique.close()


month_list = []
for line_ in sorted(set(croped_list)):
    month_list.append(time_seconds_to_str(time_str_to_seconds(line_, '%Y%m%d_%H%M'), '%m'))
month_arr = np.array(month_list, dtype=float)


hour_list = []
for line_ in sorted(set(croped_list)):
    hour_list.append(time_seconds_to_str(time_str_to_seconds(line_, '%Y%m%d_%H%M'), '%H'))
hour_list = np.array(hour_list, dtype=float)

fig, (ax_list) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6,6),)
p_hist(month_arr, x_bins=np.arange(1,13,1), x_header='month',y_header='sample count', fig_ax=(fig,ax_list[0]))
p_hist(hour_list, x_bins=np.arange(1,24,1), x_header='hour',y_header='sample count', fig_ax=(fig,ax_list[1]))


plt.tight_layout()
fig.savefig(path_output+'training_monthly_daily_distribution.png', transparent=True, bbox_inches='tight')
plt.close(fig)
# </editor-fold>


# <editor-fold desc="Confusion Matrix">
# load training data
img_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_img_array_shuf_' + model_name + '.npy')
ang_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_ang_array_shuf_' + model_name + '.npy')
cls_array_shuf = np.load(MCC_training_path + 'training_ready_data_dict_cls_array_shuf_' + model_name + '.npy')


validate_perc = .2
rows_ = img_array_shuf.shape[0]

val_rows = int(rows_ * validate_perc)
train_rows = rows_ - val_rows


training_sample_total_none_ = np.sum([cls_array_shuf[train_rows:]==0])
training_sample_total_close = np.sum([cls_array_shuf[train_rows:]==1])
training_sample_total_open_ = np.sum([cls_array_shuf[train_rows:]==2])


# <editor-fold desc="create model structure (NEW more complex)">
# define two sets of inputs
input_1 = layers.Input(shape=(16, 16, 1))
input_2 = layers.Input(shape=(4,))

# images branch
x_1 = layers.Conv2D(28, (4, 4), activation='relu')(input_1)
x_1 = layers.MaxPooling2D((2, 2))(x_1)
x_1 = layers.Conv2D(32, (3, 3), activation='relu')(x_1)
x_1 = layers.MaxPooling2D((2, 2))(x_1)
x_1 = layers.Conv2D(32, (2, 2), activation='relu')(x_1)
x_1 = layers.Flatten()(x_1)
x_1 = models.Model(inputs=input_1, outputs=x_1)

# angles branch
x_2 = layers.Dense(16, activation="relu")(input_2)
x_2 = layers.Dense(6, activation="relu")(x_2)
x_2 = models.Model(inputs=input_2, outputs=x_2)

# combine the output of the two branches
combined = layers.concatenate([x_1.output, x_2.output])
combined = layers.Dense(32, activation='relu')(combined)
combined = layers.Dense(3)(combined)

# finalize model
NN_model = models.Model(inputs=[x_1.input, x_2.input], outputs=combined)

NN_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

NN_model.summary()
# </editor-fold>

# load latest save
latest = tf.train.latest_checkpoint(MCC_model_path + 'model_CH11_angles_' + model_name + '\\')
NN_model.load_weights(latest)


# apply model
predictions = NN_model.predict([img_array_shuf[train_rows:], ang_array_shuf[train_rows:]])
MCC_predictions = np.argmax(predictions, axis=1)


# confusion matrix
confusion_matrix = tf.confusion_matrix(labels=cls_array_shuf[train_rows:], predictions=MCC_predictions, num_classes=3)
confusion_matrix_arr = np.array(confusion_matrix.eval(session=sess), dtype=float)

# normalize matrix
confusion_matrix_arr[0,:] = confusion_matrix_arr[0,:] / training_sample_total_none_
confusion_matrix_arr[1,:] = confusion_matrix_arr[1,:] / training_sample_total_close
confusion_matrix_arr[2,:] = confusion_matrix_arr[2,:] / training_sample_total_open_

print(np.round(confusion_matrix_arr, 2))
# </editor-fold>


# <editor-fold desc="training accuracy and loss plots">
training_stats_arr = numpy_load_txt(trained_model_output + "model_history_log.csv", skip_head=1)

fig, (ax_list) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(6,6),
                              gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

y_range_acc = (.9,1)
y_range_los = (0.015,.249)
x_range_acc = (0,49)
x_range_los = (0,49)


o_ = p_plot(training_stats_arr[:,0],
            training_stats_arr[:,1],
            fig_ax=(fig,ax_list[0,0]),
            title_str='Training',
            y_header='Accuracy',
            c_='r',
            add_line=True,
            custom_x_range_tuple=x_range_acc,
            custom_y_range_tuple=y_range_acc,
            )
o_ = p_plot(training_stats_arr[:,0],
            training_stats_arr[:,3],
            fig_ax=(fig,ax_list[0,1]),
            title_str='Validation',
            c_='r',
            add_line=True,
            custom_x_range_tuple=x_range_acc,
            custom_y_range_tuple=y_range_acc,
            )

o_ = p_plot(training_stats_arr[:,0],
            training_stats_arr[:,2],
            fig_ax=(fig,ax_list[1,0]),
            x_header='Epoch',
            y_header='Loss',
            c_='r',
            add_line=True,
            custom_x_range_tuple=x_range_los,
            custom_y_range_tuple=y_range_los,
            )
o_ = p_plot(training_stats_arr[:,0],
            training_stats_arr[:,4],
            fig_ax=(fig,ax_list[1,1]),
            x_header='Epoch',
            # title_str='Accuracy',
            c_='r',
            add_line=True,
            custom_x_range_tuple=x_range_los,
            custom_y_range_tuple=y_range_los,
            )

plt.setp(ax_list[0,0].get_xticklabels(), visible=False)
plt.setp(ax_list[0,1].get_xticklabels(), visible=False)

plt.setp(ax_list[0,1].get_yticklabels(), visible=False)
plt.setp(ax_list[1,1].get_yticklabels(), visible=False)

fig.subplots_adjust(hspace=0.01, wspace=0.01)

fig.savefig(path_output+'MCC_NN_training_accuracy_loss' + model_name + '.png', transparent=False, bbox_inches='tight')

# </editor-fold>


# </editor-fold>



# <editor-fold desc="make PBS job files for each month">


# <editor-fold desc="create individual run files">
# read sample file
script_name = 'MCC_type_HIM8_V_NN_13_apply_model_gadi_northern.py'
filename_date = 'run_MCC_20210810_'

file_original = open('run_MCC_sample_file.sh')
file_org_lines = file_original.readlines()
file_original.close()
p(file_org_lines)

year_list = [2016,2017,2018]
month_list = np.arange(1,13)

year_month_list_str = []
for year_ in year_list:
    for month_ in month_list:
        year_month_list_str.append(str(year_)+str(month_).zfill(2))
p(year_month_list_str)


for i_ in range(len(year_month_list_str)):
    filename_ = path_output + filename_date + str(i_+1).zfill(2) + '.sh'
    file_output = open(filename_, 'w', newline='')
    for ii_, line_ in enumerate(file_org_lines):

        if ii_ == 17:
            file_output.write('YEAR_MONTH="' + year_month_list_str[i_] + '"\n')
        elif ii_ == 16:
            file_output.write('PYTHON_SCRIPT_NAME="' + script_name + '"\n')
        else:
            file_output.write(file_org_lines[ii_])
    file_output.close()
# </editor-fold>



# <editor-fold desc="create multi run file">
file_output_multi = open(path_output + 'run_MCC_multi.sh', 'w', newline='')
file_output_multi.write('#!/bin/bash\n\n')
for i_ in range(len(year_month_list_str)):
    file_output_multi.write('qsub ' + filename_date + str(i_+1).zfill(2) + '.sh\n')
file_output_multi.close()
# </editor-fold>


# </editor-fold>









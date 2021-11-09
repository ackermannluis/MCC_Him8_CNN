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
import warnings
from numpy.lib.stride_tricks import as_strided
import zipfile
from multiprocessing import Pool
import gc

# <editor-fold desc="tools, functions, and other background commands">
path_log = '/g/data/k10/la6753/job_logs/'

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



colorbar_tick_labels_list_MCC = ['N/A', 'Closed MCC', 'Open MCC']
listed_cm_colors_list_MCC = ['none', 'red', 'blue']
listed_cm_MCC = ListedColormap(listed_cm_colors_list_MCC, 'indexed')

default_cmap = cm.jet



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

path_program = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'


def log_msg(text_, process_id):
    log_file = open(path_log + process_id + '.log', 'a')
    log_file.write(time_seconds_to_str(time.time(), time_format_mod) + '\t' + text_ + '\n')
    log_file.close()
def read_zip_png_to_array(zip_filename, element_name):
    # read image into array
    with zipfile.ZipFile(zip_filename) as archive:
        with archive.open(element_name) as file:
            img_arr = np.array(PIL_Image.open(file))

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




p = p_
################################################################ user defined variables #################
path_output = '/scratch/k10/la6753/tmp/MCC_output/'
path_data = '/scratch/k10/la6753/tmp/MCC_H08_data/'
path_model = '/g/data/k10/la6753/MCC/model_CH11_angles_20210401_V1/'
row_start = 1600
row_stop_ = 2401
show_child_prints = False
################################################################ user defined variables #################
process_id = str(os.getpid())

data_zipfile_name = list_files_recursive(path_program, 'NC_H08_')[0]
year_month_str = data_zipfile_name[-10:-4]

window_size = 16
normalization_ch11_offset = -200  # CH11
normalization_ch11_scale = 130  # CH11

normalization_SAA_offset = 180
normalization_SAA_scale = 360
normalization_SAZ_offset = 0
normalization_SAZ_scale = 90
normalization_SOA_offset = 180
normalization_SOA_scale = 360
normalization_SOZ_offset = 0
normalization_SOZ_scale = 180

###################################################################################################################

arr_SAA = np.load(path_data + 'SAA_arr.npy')[row_start:row_stop_, :]
arr_SAZ = np.load(path_data + 'SAZ_arr.npy')[row_start:row_stop_, :]
arr_SAA = arr_SAA + normalization_SAA_offset
arr_SAA = arr_SAA / normalization_SAA_scale
arr_SAZ = arr_SAZ + normalization_SAZ_offset
arr_SAZ = arr_SAZ / normalization_SAZ_scale
arr_SAA[arr_SAA > 1] = 1
arr_SAA[arr_SAA < 0] = 0
arr_SAZ[arr_SAZ > 1] = 1
arr_SAZ[arr_SAZ < 0] = 0


def MCC_noise_filter(MCC_raw_array,
                     filter_window_size_open=41, filter_window_size_close=41,
                     filter_min_average_open=0.4, filter_min_average_close=0.4,
                     filter_max_average_open=0.6, filter_max_average_close=0.6,
                     ):

    # create binaries
    MCC_OPEN_bin = np.array(MCC_raw_array)
    MCC_CLOS_bin = np.array(MCC_raw_array)
    MCC_OPEN_bin[MCC_OPEN_bin != 2] = 0
    MCC_OPEN_bin = MCC_OPEN_bin/MCC_OPEN_bin
    MCC_CLOS_bin[MCC_CLOS_bin != 1] = 0

    # get window mean
    MCC_open_bin_mean = windowed_mean(MCC_OPEN_bin, filter_window_size_open)
    MCC_clos_bin_mean = windowed_mean(MCC_CLOS_bin, filter_window_size_close)


    # filter
    MCC_CLOS_bin[np.where(MCC_clos_bin_mean > filter_max_average_close)] = 1
    MCC_CLOS_bin[np.where(MCC_clos_bin_mean < filter_min_average_close)] = 0

    MCC_OPEN_bin[np.where(MCC_open_bin_mean > filter_max_average_open)] = 1
    MCC_OPEN_bin[np.where(MCC_open_bin_mean < filter_min_average_open)] = 0

    # combine binaries into one output array
    MCC_clean_array = MCC_CLOS_bin + (MCC_OPEN_bin * 2)
    MCC_clean_array[np.where(MCC_clean_array > 2)] = 0

    return MCC_clean_array


def MCC_filter_NN_CH11_angles_2(datetime_str):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    latest = tf.train.latest_checkpoint(path_model)

    # <editor-fold desc="create model structure">
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
    # </editor-fold>

    # load weights
    NN_model.load_weights(latest)

    try:
        max_ch13 = 288.15
        min_ch13 = 253.15

        if show_child_prints: log_msg('starting filtering', datetime_str)

        # get file data
        arr_CH11 = read_png_to_array(list_files_recursive(path_program,
                                                          'NC_H08_' + datetime_str + '_ch11.png')[0])[row_start:row_stop_, :]
        arr_CH13 = read_png_to_array(list_files_recursive(path_program,
                                                          'NC_H08_' + datetime_str + '_ch13.png')[0])[row_start:row_stop_, :]

        arr_SOA  = read_png_to_array(list_files_recursive(path_program,
                                                          'NC_H08_' + datetime_str + '_SOA.png')[0])[row_start:row_stop_, :]
        arr_SOZ  = read_png_to_array(list_files_recursive(path_program,
                                                          'NC_H08_' + datetime_str + '_SOZ.png')[0])[row_start:row_stop_, :]
        if show_child_prints: log_msg('loaded data into arrays', datetime_str)

        # normalyze
        arr_CH11 = arr_CH11 + normalization_ch11_offset
        arr_CH11 = arr_CH11 / normalization_ch11_scale

        # cap
        arr_CH11[arr_CH11 > 1] = 1
        arr_CH11[arr_CH11 < 0] = 0

        # delete invalids
        arr_CH11[np.isnan(arr_CH11)] = 0

        # normalyze
        arr_SOA = arr_SOA + normalization_SOA_offset
        arr_SOA = arr_SOA / normalization_SOA_scale
        arr_SOZ = arr_SOZ + normalization_SOZ_offset
        arr_SOZ = arr_SOZ / normalization_SOZ_scale

        # cap
        arr_SOA[arr_SOA > 1] = 1
        arr_SOA[arr_SOA < 0] = 0
        arr_SOZ[arr_SOZ > 1] = 1
        arr_SOZ[arr_SOZ < 0] = 0

        # delete invalids
        arr_SAA[np.isnan(arr_SAA)] = 0
        arr_SAZ[np.isnan(arr_SAZ)] = 0
        arr_SOA[np.isnan(arr_SOA)] = 0
        arr_SOZ[np.isnan(arr_SOZ)] = 0

        if show_child_prints: log_msg('normalized arrays', datetime_str)



        # slice into moving window
        arr_CH11_sliced = sliding_window_view(arr_CH11, (window_size, window_size), (1, 1))
        if show_child_prints: log_msg('sliced arrays', datetime_str)


        # reshape for optimal model application
        arr_CH_sliced_reshaped = arr_CH11_sliced.reshape(arr_CH11_sliced.shape[0] * arr_CH11_sliced.shape[1],
                                                       window_size, window_size, 1)
        # arr_CH_sliced_reshaped = np.empty((arr_CH11_sliced.shape[0] * arr_CH11_sliced.shape[1],
        #                                                window_size, window_size, 2), dtype=float)
        # arr_CH_sliced_reshaped[:,:,:,0] = arr_CH11_sliced_reshaped[:,:,:,0]
        # arr_CH_sliced_reshaped[:,:,:,1] = arr_CH10_sliced_reshaped[:,:,:,0]

        arr_angles_reshaped = np.zeros((arr_CH11_sliced.shape[0] * arr_CH11_sliced.shape[1], 4))
        arr_angles_reshaped[:, 0] = arr_SAA[int(window_size/2):-int(window_size/2)+1, int(window_size/2):-int(window_size/2)+1].flatten()
        arr_angles_reshaped[:, 1] = arr_SAZ[int(window_size/2):-int(window_size/2)+1, int(window_size/2):-int(window_size/2)+1].flatten()
        arr_angles_reshaped[:, 2] = arr_SOA[int(window_size/2):-int(window_size/2)+1, int(window_size/2):-int(window_size/2)+1].flatten()
        arr_angles_reshaped[:, 3] = arr_SOZ[int(window_size/2):-int(window_size/2)+1, int(window_size/2):-int(window_size/2)+1].flatten()
        if show_child_prints: log_msg('reshaped arrays', datetime_str)


        # apply model
        predictions = NN_model.predict([arr_CH_sliced_reshaped, arr_angles_reshaped])
        if show_child_prints: log_msg('applied model', datetime_str)

        # convert to MCC
        score = tf.nn.softmax(predictions)
        MRR_predicts = np.argmax(score, axis=1)
        MRR_predicts_reshaped = MRR_predicts.reshape(arr_CH11_sliced.shape[0], arr_CH11_sliced.shape[1])
        if show_child_prints: log_msg('reshaped output', datetime_str)


        # create output arrays
        window_half = int((window_size)/2)
        output_array = arr_CH11 * 0
        output_array[window_half-1:-window_half,window_half-1:-window_half] = MRR_predicts_reshaped
        store_array_to_png(output_array, path_program + 'MCC_raw_' + datetime_str + '.png')
        if show_child_prints: log_msg('saved', datetime_str)

        # apply ch13 thresholds
        output_array[np.where(arr_CH13 > max_ch13)] = 0
        output_array[np.where(arr_CH13 < min_ch13)] = 0

        # clean noise
        output_array_clean = MCC_noise_filter(output_array,
                                              filter_window_size_open=41, filter_window_size_close=41)
        store_array_to_png(output_array_clean, path_program + 'MCC_clean_' + datetime_str + '.png')
        if show_child_prints: log_msg('saved clean', datetime_str)

        tf.keras.backend.clear_session()
        del predictions
        gc.collect()

        return 0
    except BaseException as error_msg:
        exc_type, exc_obj, tb = sys.exc_info()
        log_msg('Error in line: ' + str(tb.tb_lineno) + '\n' + 'Error while saving data \n' + str(error_msg),
                datetime_str)
        return 1

if __name__ == '__main__':
    log_msg('main process started', year_month_str)


    # extract files from zip archive
    with zipfile.ZipFile(data_zipfile_name, 'r') as zip_ref:
        zip_ref.extractall(path_program)
    log_msg('extracted all files to working path', year_month_str)

    # count processors and define pool
    processes_ = 30
    Pool_ = Pool(processes_)
    log_msg('started children processes', year_month_str)

    # get datetime_str_list
    file_list = list_files_recursive(path_program, '.png')
    datetime_str_list = []
    for i_ in range(len(file_list)):
        datetime_str_list.append(file_list[i_].split('/')[-1][7:20])
    datetime_str_list_unique = sorted(set(datetime_str_list))
    log_msg('create list of timestamps to be processed', year_month_str)

    # start filtering
    multip_out = Pool_.map(MCC_filter_NN_CH11_angles_2, datetime_str_list_unique)
    log_msg('processed all files, errors while processing month = ' + str(np.sum(multip_out)), year_month_str)


    # get file list
    file_list_raw = list_files_recursive(path_program, filter_str='MCC_raw_')
    # start the zip file
    filename_zip = path_output + 'MCC_raw_' + year_month_str + '.zip'
    zf = zipfile.ZipFile(filename_zip, mode='w')
    zf.close()
    # zip pngs and delete source
    for filename_ in file_list_raw:
        zf = zipfile.ZipFile(filename_zip, mode='a')
        zf.write(filename_)
        zf.close()
    log_msg('compressed raw files!', year_month_str)

    # get file list
    file_list_clean = list_files_recursive(path_program, filter_str='MCC_clean_')
    # start the zip file
    filename_zip = path_output + 'MCC_clean_' + year_month_str + '.zip'
    zf = zipfile.ZipFile(filename_zip, mode='w')
    zf.close()
    # zip pngs and delete source
    for filename_ in file_list_clean:
        zf = zipfile.ZipFile(filename_zip, mode='a')
        zf.write(filename_)
        zf.close()
    log_msg('compressed clean files!', year_month_str)


    # report possible errors
    for i_ in range(len(datetime_str_list_unique)):
        log_msg(datetime_str_list_unique[i_] + '\t' + str(multip_out[i_]), year_month_str)









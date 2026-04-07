"""
This file contains the 'topologies' class that handles the topology files fed to SCALE_Sim tool.
"""

import math


class topologies(object):
    """
    Class which contains the methods to preprocess the data from topology file (.csv format) before
    doing compute simulation.

    Internal topology entry layout:
    [name, ifmap_h, ifmap_w, filt_h, filt_w, channels, num_filters,
     stride_h, stride_w, batch, sparsity_n, sparsity_m]
    """

    LAYER_NAME_IDX = 0
    IFMAP_H_IDX = 1
    IFMAP_W_IDX = 2
    FILT_H_IDX = 3
    FILT_W_IDX = 4
    NUM_CH_IDX = 5
    NUM_FILT_IDX = 6
    STRIDE_H_IDX = 7
    STRIDE_W_IDX = 8
    BATCH_IDX = 9
    SPARSITY_N_IDX = 10
    SPARSITY_M_IDX = 11

    INTERNAL_ENTRY_LEN = 12
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_SPARSITY_N = 1
    DEFAULT_SPARSITY_M = 1

    CONV_HEADER_ALIASES = {
        'layer_name': ('layer name', 'layer'),
        'ifmap_h': ('ifmap height',),
        'ifmap_w': ('ifmap width',),
        'filt_h': ('filter height',),
        'filt_w': ('filter width',),
        'channels': ('channels',),
        'num_filters': ('num filter', 'num filters'),
        'stride': ('strides', 'stride'),
        'stride_h': ('stride height', 'stride h'),
        'stride_w': ('stride width', 'stride w'),
        'batch': ('batch', 'batch size'),
        'sparsity': ('sparsity', 'sparsity ratio'),
    }

    GEMM_HEADER_ALIASES = {
        'layer_name': ('layer name', 'layer'),
        'm': ('m',),
        'n': ('n',),
        'k': ('k',),
        'batch': ('batch', 'batch size'),
        'sparsity': ('sparsity', 'sparsity ratio'),
    }

    #
    def __init__(self):
        """
        __init__ method
        """
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_arrays = []
        self.spatio_temp_dim_arrays = []
        self.layers_calculated_hyperparams = []
        self.num_layers = 0
        self.topo_load_flag = False
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False
        self.df = ""
        self.current_toponame = ""
        self.layer_name = ""
        self.mnk_inputs = False

    # reset topology parameters
    def reset(self):
        """
        Method to reset the topology parameters.
        """
        print("All data reset")
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_load_flag = False
        self.topo_arrays = []
        self.num_layers = 0
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False
        self.spatio_temp_dim_arrays = []
        self.layers_calculated_hyperparams = []
        self.df = ""
        self.current_toponame = ""
        self.layer_name = ""
        self.mnk_inputs = False

    #
    def load_layer_params_from_list(self, layer_name, elems_list=None):
        """
        Method to load layer parameters from the given layer name and element list.
        """
        if elems_list is None:
            elems_list = []

        self.topo_file_name = ''
        self.current_toponame = ''
        self.layer_name = layer_name
        self.mnk_inputs = False
        self.append_topo_arrays(layer_name, elems_list)

        self.num_layers += 1
        self.topo_load_flag = True
        self._validate_global_batch_consistency()

    #
    def load_arrays(self, topofile='', mnk_inputs=False):
        """
        Method to read the topology file and collect names and dimensions of all the workload
        layers.
        """
        if mnk_inputs:
            self.load_arrays_gemm(topofile)
        else:
            self.load_arrays_conv(topofile)

    #
    def load_arrays_gemm(self, topofile=''):
        """
        Method to read the GEMM topology file and collect names and dimensions of all the workload
        layers.
        """
        self._prepare_for_file_load(topofile=topofile, mnk_inputs=True)

        with open(topofile, 'r') as topofile_handle:
            header = None
            for row in topofile_handle:
                stripped_row = row.strip()
                if stripped_row == '':
                    continue

                fields = self._split_row(row)
                if header is None:
                    header = [self._normalize_header_text(elem) for elem in fields]
                    continue

                row_dict = self._build_row_dict(header, fields)
                layer_name = self._get_required_field(
                    row_dict, self.GEMM_HEADER_ALIASES['layer_name'], 'layer name'
                )
                m = self._get_required_positive_int(row_dict, self.GEMM_HEADER_ALIASES['m'], 'M')
                n = self._get_required_positive_int(row_dict, self.GEMM_HEADER_ALIASES['n'], 'N')
                k = self._get_required_positive_int(row_dict, self.GEMM_HEADER_ALIASES['k'], 'K')
                batch_size = self._get_optional_batch_size(row_dict)
                sparsity_n, sparsity_m = self._get_optional_sparsity_ratio(row_dict)

                entry = self._make_entry(
                    layer_name=layer_name,
                    ifmap_h=m,
                    ifmap_w=k,
                    filt_h=1,
                    filt_w=k,
                    num_ch=1,
                    num_filt=n,
                    stride_h=1,
                    stride_w=1,
                    batch_size=batch_size,
                    sparsity_n=sparsity_n,
                    sparsity_m=sparsity_m
                )
                self._append_entry(entry)

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = True
        self._validate_global_batch_consistency()

    # Load the topology data from the file
    def load_arrays_conv(self, topofile=''):
        """
        Method to read the CONV topology file and collect names and dimensions of all the workload
        layers.
        """
        self._prepare_for_file_load(topofile=topofile, mnk_inputs=False)

        with open(topofile, 'r') as topofile_handle:
            header = None
            for row in topofile_handle:
                stripped_row = row.strip()
                if stripped_row == '':
                    continue

                fields = self._split_row(row)
                if header is None:
                    header = [self._normalize_header_text(elem) for elem in fields]
                    continue

                row_dict = self._build_row_dict(header, fields)
                layer_name = self._get_required_field(
                    row_dict, self.CONV_HEADER_ALIASES['layer_name'], 'layer name'
                )
                ifmap_h = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['ifmap_h'], 'IFMAP height'
                )
                ifmap_w = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['ifmap_w'], 'IFMAP width'
                )
                filt_h = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['filt_h'], 'Filter height'
                )
                filt_w = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['filt_w'], 'Filter width'
                )
                num_ch = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['channels'], 'Channels'
                )
                num_filt = self._get_required_positive_int(
                    row_dict, self.CONV_HEADER_ALIASES['num_filters'], 'Num filter'
                )
                stride_h, stride_w = self._get_conv_strides(row_dict)
                batch_size = self._get_optional_batch_size(row_dict)
                sparsity_n, sparsity_m = self._get_optional_sparsity_ratio(row_dict)

                entry = self._make_entry(
                    layer_name=layer_name,
                    ifmap_h=ifmap_h,
                    ifmap_w=ifmap_w,
                    filt_h=filt_h,
                    filt_w=filt_w,
                    num_ch=num_ch,
                    num_filt=num_filt,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    batch_size=batch_size,
                    sparsity_n=sparsity_n,
                    sparsity_m=sparsity_m
                )

                if 'DP' in layer_name.strip():
                    for dp_layer in range(num_ch):
                        dp_entry = list(entry)
                        dp_entry[self.LAYER_NAME_IDX] = \
                            layer_name.strip() + "Channel_" + str(dp_layer)
                        dp_entry[self.NUM_CH_IDX] = 1
                        self._append_entry(dp_entry)
                else:
                    self._append_entry(entry)

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = True
        self._validate_global_batch_consistency()

    # Write the contents into a csv file
    def write_topo_file(self,
                      path="",
                      filename=""
                      ):
        """
        Method to write the workload data into a csv file.
        """
        if path == "":
            print("WARNING: topology_utils.write_topo_file: No path specified writing to the cwd")
            path = "./"

        if filename == "":
            print("ERROR: topology_utils.write_topo_file: No filename provided")
            return

        filename = path + "/" + filename

        if not self.topo_load_flag:
            print("ERROR: topology_utils.write_topo_file: No data loaded")
            return

        if self.mnk_inputs:
            header = [
                "Layer",
                "M",
                "N",
                "K",
                "Batch Size",
                "Sparsity"
            ]
        else:
            header = [
                "Layer name",
                "IFMAP height",
                "IFMAP width",
                "Filter height",
                "Filter width",
                "Channels",
                "Num filter",
                "Stride height",
                "Stride width",
                "Batch Size",
                "Sparsity"
            ]

        with open(filename, 'w') as topofile_handle:
            log = ",".join(header)
            log += ",\n"
            topofile_handle.write(log)

            for param_arr in self.topo_arrays:
                if self.mnk_inputs:
                    row = [
                        str(param_arr[self.LAYER_NAME_IDX]),
                        str(param_arr[self.IFMAP_H_IDX]),
                        str(param_arr[self.NUM_FILT_IDX]),
                        str(param_arr[self.IFMAP_W_IDX]),
                        str(param_arr[self.BATCH_IDX]),
                        self._format_sparsity_ratio(
                            param_arr[self.SPARSITY_N_IDX], param_arr[self.SPARSITY_M_IDX]
                        )
                    ]
                else:
                    row = [
                        str(param_arr[self.LAYER_NAME_IDX]),
                        str(param_arr[self.IFMAP_H_IDX]),
                        str(param_arr[self.IFMAP_W_IDX]),
                        str(param_arr[self.FILT_H_IDX]),
                        str(param_arr[self.FILT_W_IDX]),
                        str(param_arr[self.NUM_CH_IDX]),
                        str(param_arr[self.NUM_FILT_IDX]),
                        str(param_arr[self.STRIDE_H_IDX]),
                        str(param_arr[self.STRIDE_W_IDX]),
                        str(param_arr[self.BATCH_IDX]),
                        self._format_sparsity_ratio(
                            param_arr[self.SPARSITY_N_IDX], param_arr[self.SPARSITY_M_IDX]
                        )
                    ]
                log = ",".join(row)
                log += ",\n"
                topofile_handle.write(log)

    # LEGACY
    def append_topo_arrays(self, layer_name, elems):
        """
        Method to append the layer dimensions in int data type and layer name to the topo_arrays
        variable. This method also checks that the filter dimensions do not exceed the ifmap
        dimensions.
        """
        values = list(elems)
        if values and str(values[0]).strip() == str(layer_name).strip():
            values = values[1:]

        entry = self._normalize_external_entry(layer_name, values)
        self._append_entry(entry)

    # create network topology array
    def append_topo_entry_from_list(self, layer_entry_list=None):
        """
        Method to append the layer dimensions in int data type and layer name.
        """
        if layer_entry_list is None:
            layer_entry_list = []

        assert 7 < len(layer_entry_list) < 13, 'Incorrect number of parameters'

        layer_name = str(layer_entry_list[0]).strip()
        entry = self._normalize_external_entry(layer_name, layer_entry_list[1:])
        self.append_layer_entry(entry, toponame=self.current_topo_name)

    # add to the existing data from a list
    def append_layer_entry(self, entry, toponame=""):
        """
        Method to append data of a single layer to the array containing data of all the layers. This
        method also calls topo_calc_hyperparams method to calculate the hyperparameters.
        """
        assert len(entry) == self.INTERNAL_ENTRY_LEN, 'Incorrect number of parameters'

        if toponame != "":
            self.current_topo_name = toponame

        self._append_entry(entry)
        self.topo_load_flag = True
        self.topo_calc_hyperparams()
        self.num_layers = len(self.topo_arrays)
        self._validate_global_batch_consistency()

    # calculate hyper-parameters (ofmap dimensions, number of MACs, and window size of filter)
    def topo_calc_hyperparams(self, topofilename=""):
        """
        Method to calculate hyper-parameters (ofmap dimensions, number of MACs, and window size of
        filter) if topology array is loaded.
        """
        if not self.topo_load_flag:
            self.load_arrays(topofilename, mnk_inputs=self.mnk_inputs)
        self.layers_calculated_hyperparams = []
        for array in self.topo_arrays:
            ifmap_h = array[self.IFMAP_H_IDX]
            ifmap_w = array[self.IFMAP_W_IDX]
            filt_h = array[self.FILT_H_IDX]
            filt_w = array[self.FILT_W_IDX]
            num_ch = array[self.NUM_CH_IDX]
            num_filt = array[self.NUM_FILT_IDX]
            stride_h = array[self.STRIDE_H_IDX]
            stride_w = array[self.STRIDE_W_IDX]
            ofmap_h = int(math.ceil((ifmap_h - filt_h + stride_h) / stride_h))
            ofmap_w = int(math.ceil((ifmap_w - filt_w + stride_w) / stride_w))
            num_mac = ofmap_h * ofmap_w * filt_h * filt_w * num_ch * num_filt
            window_size = filt_h * filt_w * num_ch
            entry = [ofmap_h, ofmap_w, num_mac, window_size]
            self.layers_calculated_hyperparams.append(entry)
        self.topo_calc_hyper_param_flag = True

    #
    def calc_spatio_temporal_params(self, df='os', layer_id=0):
        """
        Method to calculate spatio-temporal parameters (S_r, S_c and T) based on the dataflow.
        (Refer the scalesim paper for more info)
        """
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(self.topo_file_name)

        s_row = -1
        s_col = -1
        t_time = -1
        num_filt = self.get_layer_num_filters(layer_id=layer_id)
        num_ofmap = self.get_layer_num_ofmap_px(layer_id=layer_id)
        num_ofmap = int(num_ofmap / num_filt)
        window_sz = self.get_layer_window_size(layer_id=layer_id)
        if df == 'os':
            s_row = num_ofmap
            s_col = num_filt
            t_time = window_sz
        elif df == 'ws':
            s_row = window_sz
            s_col = num_filt
            t_time = num_ofmap
        elif df == 'is':
            s_row = window_sz
            s_col = num_ofmap
            t_time = num_filt
        return s_row, s_col, t_time

    #
    def set_spatio_temporal_params(self):
        """
        Method to calculate spatio-temporal parameters (S_r, S_c and T) for all the layers
        """
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(self.topo_file_name)
        self.spatio_temp_dim_arrays = []
        for i in range(self.num_layers):
            this_layer_params_arr = []
            for df in ['os', 'ws', 'is']:
                sr, sc, tt = self.calc_spatio_temporal_params(df=df, layer_id=i)
                this_layer_params_arr.append([sr, sc, tt])
            self.spatio_temp_dim_arrays.append(this_layer_params_arr)
        self.topo_calc_spatiotemp_params_flag = True

    #
    def get_transformed_mnk_dimensions(self):
        """
        Method to get M, N and K parameters for all the layers. These are GEMM parameters in which
        an input matrix of dimensions MxN is multiplied to a filter matrix of dimensions NxK.
        """
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(self.topo_file_name)

        mnk_dims_arr = []
        for i in range(self.num_layers):
            if self.mnk_inputs:
                layer_params = self.get_layer_params(layer_id=i)
                m_dim = layer_params[self.IFMAP_H_IDX] * layer_params[self.BATCH_IDX]
                n_dim = layer_params[self.NUM_FILT_IDX]
                k_dim = layer_params[self.IFMAP_W_IDX]
            else:
                m_dim = self.get_layer_num_ofmap_px(layer_id=i)
                n_dim = self.get_layer_num_filters(layer_id=i)
                k_dim = self.get_layer_window_size(layer_id=i)

            mnk_dims_arr.append([m_dim, n_dim, k_dim])

        return mnk_dims_arr

    #
    def get_current_topo_name(self):
        """
        Method to get the name of the workload if available. If not, print an error message.
        """
        current_topo_name = ""
        if self.topo_load_flag:
            current_topo_name = self.current_topo_name
        else:
            print('Error: get_current_topo_name(): Topo file not read')
        return current_topo_name

    #
    def get_num_layers(self):
        """
        Method to get the number of layers of the workload if available. If not, print an error
        message.
        """
        if not self.topo_load_flag:
            print("ERROR: topologies.get_num_layers: No array loaded")
            return
        return self.num_layers

    #
    def get_layer_ifmap_dims(self, layer_id=0):
        """
        Method to get the ifmap dimensions of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ifmap_dims: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.IFMAP_H_IDX:self.IFMAP_W_IDX + 1]

    #
    def get_layer_filter_dims(self, layer_id=0):
        """
        Method to get the filter dimensions of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ifmap_dims: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.FILT_H_IDX:self.FILT_W_IDX + 1]

    #
    def get_layer_num_filters(self, layer_id=0):
        """
        Method to get the number of filters of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.NUM_FILT_IDX]

    #
    def get_layer_num_channels(self, layer_id=0):
        """
        Method to get the number of channels of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.NUM_CH_IDX]

    #
    def get_layer_strides(self, layer_id=0):
        """
        Method to get the strides of the layer if available. If not, print an error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_strides: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.STRIDE_H_IDX:self.STRIDE_W_IDX + 1]

    #
    def get_layer_batch_size(self, layer_id=0):
        """
        Method to get the batch size of the layer if available. If not, print an error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_batch_size: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.BATCH_IDX]

    #
    def get_layer_sparsity_ratio(self, layer_id=0):
        """
        Method to get the sparsity ratio of the layer if available. If not, print an error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_sparsity_ratio: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[self.SPARSITY_N_IDX:self.SPARSITY_M_IDX + 1]

    #
    def get_layer_window_size(self, layer_id=0):
        """
        Method to get the convolution window size of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        layer_calc_params = self.layers_calculated_hyperparams[layer_id]
        return layer_calc_params[3]

    #
    def get_layer_num_ofmap_px(self, layer_id=0):
        """
        Method to get the total number of OFMAP elements produced by the layer across the full
        batch. If not, print an error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        layer_calc_params = self.layers_calculated_hyperparams[layer_id]
        num_filters = self.get_layer_num_filters(layer_id)
        batch_size = self.get_layer_batch_size(layer_id)
        num_ofmap_px = layer_calc_params[0] * layer_calc_params[1] * num_filters * batch_size
        return num_ofmap_px

    #
    def get_layer_ofmap_dims(self, layer_id=0):
        """
        Method to get the ofmap dimensions of the layer if available. If not, print an error
        message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ofmap_dims: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        ofmap_dims = self.layers_calculated_hyperparams[layer_id][0:2]
        return ofmap_dims

    #
    def get_layer_params(self, layer_id=0):
        """
        Method to get the parameters of the layer if available. If not, print an error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_params: Invalid layer id")
            return
        layer_params = self.topo_arrays[layer_id]
        return layer_params

    #
    def get_layer_id_from_name(self, layer_name=""):
        """
        Method to get layer number from the given layer name if available. If not, print an error
        message.
        """
        if (not self.topo_load_flag) or layer_name == "":
            print("ERROR")
            return
        indx = -1
        for i in range(len(self.topo_arrays)):
            if layer_name == self.topo_arrays[i][self.LAYER_NAME_IDX]:
                indx = i
        if indx == -1:
            print("WARNING: Not found")
        return indx

    #
    def get_layer_name(self, layer_id=0):
        """
        Method to get the layer name from the given layer number if available. If not, print an
        error message.
        """
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_name: Invalid layer id")
            return

        name = self.topo_arrays[layer_id][self.LAYER_NAME_IDX]
        return str(name)

    #
    def get_layer_names(self):
        """
        Method to get the names of all the layers in the workload. If not, print an error message.
        """
        if not self.topo_load_flag:
            print("ERROR")
            return
        layer_names = []
        for entry in self.topo_arrays:
            layer_name = str(entry[self.LAYER_NAME_IDX])
            layer_names.append(layer_name)
        return layer_names

    #
    def get_layer_mac_ops(self, layer_id=0):
        """
        Method to get the total number of MAC operations of the layer across the full batch. If
        hyper-parameters are not calculated, calculate them first.
        """
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(topofilename=self.topo_file_name)
        layer_hyper_param = self.layers_calculated_hyperparams[layer_id]
        batch_size = self.get_layer_batch_size(layer_id)
        mac_ops = layer_hyper_param[2] * batch_size
        return mac_ops

    #
    def get_all_mac_ops(self):
        """
        Method to get the total number of mac operations of all the layer if hyper-parameters are
        calculated. If not, calculate the hyper-parameters first.
        """
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(topofilename=self.topo_file_name)
        total_mac = 0
        for layer in range(self.num_layers):
            total_mac += self.get_layer_mac_ops(layer)
        return total_mac

    # spatio-temporal dimensions specific to dataflow
    def get_spatiotemporal_dims(self, layer_id=0, df=''):
        """
        Method to get the spatio-temporal dimensions (S_r, S_c, T) of the layer if spatio-temporal
        parameters are calculated. If not, calculate the spatio-temporal parameters first. (refer to
        the scalesim paper for more info)
        """
        if df == '':
            df = self.df
        if not self.topo_calc_spatiotemp_params_flag:
            self.set_spatio_temporal_params()
        df_list = ['os', 'ws', 'is']
        df_idx = df_list.index(df)
        s_row = self.spatio_temp_dim_arrays[layer_id][df_idx][0]
        s_col = self.spatio_temp_dim_arrays[layer_id][df_idx][1]
        t_time = self.spatio_temp_dim_arrays[layer_id][df_idx][2]
        return s_row, s_col, t_time

    def _prepare_for_file_load(self, topofile='', mnk_inputs=False):
        self.topo_file_name = topofile.split('/')[-1]
        name_arr = self.topo_file_name.split('.')
        if len(name_arr) > 1:
            self.current_topo_name = self.topo_file_name.split('.')[-2]
        else:
            self.current_topo_name = self.topo_file_name

        self.mnk_inputs = mnk_inputs
        self.topo_arrays = []
        self.layers_calculated_hyperparams = []
        self.spatio_temp_dim_arrays = []
        self.num_layers = 0
        self.topo_load_flag = False
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False

    def _append_entry(self, entry):
        assert len(entry) == self.INTERNAL_ENTRY_LEN, 'Incorrect number of parameters'
        self._validate_entry(entry)
        self.topo_arrays.append(entry)
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False

    def _validate_entry(self, entry):
        assert entry[self.FILT_H_IDX] <= entry[self.IFMAP_H_IDX], \
            'Filter height cannot be larger than IFMAP height'
        assert entry[self.FILT_W_IDX] <= entry[self.IFMAP_W_IDX], \
            'Filter width cannot be larger than IFMAP width'

    def _normalize_external_entry(self, layer_name, values):
        if len(values) < 7:
            raise AssertionError('Incorrect number of parameters')

        mandatory = [self._coerce_positive_int(value, 'topology field') for value in values[:7]]
        ifmap_h, ifmap_w, filt_h, filt_w, num_ch, num_filt, stride_h = mandatory

        batch_size = self.DEFAULT_BATCH_SIZE
        sparsity_n = self.DEFAULT_SPARSITY_N
        sparsity_m = self.DEFAULT_SPARSITY_M
        stride_w = stride_h
        remaining = list(values[7:])

        if len(remaining) == 1:
            if self._is_ratio_value(remaining[0]):
                sparsity_n, sparsity_m = self._parse_sparsity_ratio(remaining[0])
            else:
                batch_size = self._coerce_batch_size(remaining[0])
        elif len(remaining) == 2:
            if self._is_ratio_value(remaining[1]):
                batch_size = self._coerce_batch_size(remaining[0])
                sparsity_n, sparsity_m = self._parse_sparsity_ratio(remaining[1])
            else:
                stride_w = self._coerce_positive_int(remaining[0], 'stride width')
                batch_size = self._coerce_batch_size(remaining[1])
        elif len(remaining) == 3:
            if self._is_ratio_value(remaining[2]):
                stride_w = self._coerce_positive_int(remaining[0], 'stride width')
                batch_size = self._coerce_batch_size(remaining[1])
                sparsity_n, sparsity_m = self._parse_sparsity_ratio(remaining[2])
            else:
                batch_size = self._coerce_batch_size(remaining[0])
                sparsity_n = self._coerce_positive_int(remaining[1], 'sparsity ratio N')
                sparsity_m = self._coerce_positive_int(remaining[2], 'sparsity ratio M')
        elif len(remaining) >= 4:
            stride_w = self._coerce_positive_int(remaining[0], 'stride width')
            batch_size = self._coerce_batch_size(remaining[1])
            if self._is_ratio_value(remaining[2]):
                sparsity_n, sparsity_m = self._parse_sparsity_ratio(remaining[2])
            else:
                sparsity_n = self._coerce_positive_int(remaining[2], 'sparsity ratio N')
                sparsity_m = self._coerce_positive_int(remaining[3], 'sparsity ratio M')

        return self._make_entry(
            layer_name=layer_name,
            ifmap_h=ifmap_h,
            ifmap_w=ifmap_w,
            filt_h=filt_h,
            filt_w=filt_w,
            num_ch=num_ch,
            num_filt=num_filt,
            stride_h=stride_h,
            stride_w=stride_w,
            batch_size=batch_size,
            sparsity_n=sparsity_n,
            sparsity_m=sparsity_m
        )

    def _make_entry(self,
                    layer_name,
                    ifmap_h,
                    ifmap_w,
                    filt_h,
                    filt_w,
                    num_ch,
                    num_filt,
                    stride_h,
                    stride_w,
                    batch_size,
                    sparsity_n,
                    sparsity_m):
        return [
            str(layer_name).strip(),
            int(ifmap_h),
            int(ifmap_w),
            int(filt_h),
            int(filt_w),
            int(num_ch),
            int(num_filt),
            int(stride_h),
            int(stride_w),
            int(batch_size),
            int(sparsity_n),
            int(sparsity_m)
        ]

    def _split_row(self, row):
        fields = row.rstrip('\n\r').split(',')
        if fields and fields[-1].strip() == '':
            fields = fields[:-1]
        return fields

    def _normalize_header_text(self, header_text):
        normalized = str(header_text).lstrip('\ufeff').strip().lower()
        normalized = " ".join(normalized.split())
        return normalized

    def _build_row_dict(self, header, fields):
        row_dict = {}
        for index, header_name in enumerate(header):
            if header_name == '':
                continue
            value = ''
            if index < len(fields):
                value = fields[index].strip()
            row_dict[header_name] = value
        return row_dict

    def _find_header_value(self, row_dict, aliases):
        for alias in aliases:
            if alias in row_dict:
                return row_dict[alias]
        return None

    def _get_required_field(self, row_dict, aliases, field_name):
        value = self._find_header_value(row_dict, aliases)
        if value is None or value.strip() == '':
            raise ValueError('Missing required topology field: {}'.format(field_name))
        return value.strip()

    def _get_required_positive_int(self, row_dict, aliases, field_name):
        value = self._get_required_field(row_dict, aliases, field_name)
        return self._coerce_positive_int(value, field_name)

    def _get_optional_batch_size(self, row_dict):
        value = self._find_header_value(row_dict, self.CONV_HEADER_ALIASES['batch'])
        if value is None or value.strip() == '':
            return self.DEFAULT_BATCH_SIZE
        return self._coerce_batch_size(value)

    def _get_optional_sparsity_ratio(self, row_dict):
        value = self._find_header_value(row_dict, self.CONV_HEADER_ALIASES['sparsity'])
        if value is None or value.strip() == '':
            return self.DEFAULT_SPARSITY_N, self.DEFAULT_SPARSITY_M
        return self._parse_sparsity_ratio(value)

    def _get_conv_strides(self, row_dict):
        stride_value = self._find_header_value(row_dict, self.CONV_HEADER_ALIASES['stride'])
        stride_h_value = self._find_header_value(row_dict, self.CONV_HEADER_ALIASES['stride_h'])
        stride_w_value = self._find_header_value(row_dict, self.CONV_HEADER_ALIASES['stride_w'])

        if stride_value is not None and stride_value.strip() != '':
            stride = self._coerce_positive_int(stride_value, 'Stride')
            return stride, stride

        stride_h = self._coerce_positive_int(stride_h_value, 'Stride height')
        if stride_w_value is None or stride_w_value.strip() == '':
            return stride_h, stride_h
        stride_w = self._coerce_positive_int(stride_w_value, 'Stride width')
        return stride_h, stride_w

    def _parse_sparsity_ratio(self, sparsity_value):
        parts = str(sparsity_value).strip().split(':')
        if len(parts) != 2:
            raise ValueError('Invalid sparsity ratio: {}'.format(sparsity_value))
        sparsity_n = self._coerce_positive_int(parts[0], 'sparsity ratio N')
        sparsity_m = self._coerce_positive_int(parts[1], 'sparsity ratio M')
        return sparsity_n, sparsity_m

    def _format_sparsity_ratio(self, sparsity_n, sparsity_m):
        return '{}:{}'.format(int(sparsity_n), int(sparsity_m))

    def _coerce_positive_int(self, value, field_name):
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ValueError('Invalid {} value: {}'.format(field_name, value)) from exc
        if parsed <= 0:
            raise ValueError('Invalid {} value: {}'.format(field_name, value))
        return parsed

    def _coerce_batch_size(self, value):
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ValueError('Invalid batch value: {}'.format(value)) from exc
        if parsed <= 0:
            raise ValueError('Invalid batch value: {}'.format(value))
        return parsed

    def _is_ratio_value(self, value):
        return ':' in str(value)

    def _validate_global_batch_consistency(self):
        batch_sizes = {entry[self.BATCH_IDX] for entry in self.topo_arrays}
        if len(batch_sizes) > 1:
            raise ValueError(
                'Mixed per-layer batch sizes are not supported in one topology: {}'.format(
                    sorted(batch_sizes)
                )
            )


if __name__ == '__main__':
    tp = topologies()

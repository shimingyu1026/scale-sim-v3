"""
Configuration file for ML Predictor module.
Contains all hyperparameters and configurable settings.
"""

# ============== Data Generation Config ==============
# DATA_GENERATION_CONFIG = {
#     # Number of samples to generate
#     "num_samples": 5000,
#     # Hardware config parameter ranges
#     "array_height_range": [64, 128, 256, 512],
#     "array_width_range": [64, 128, 256, 512],
#     "ifmap_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
#     "filter_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
#     "ofmap_sram_sz_kb_range": [256, 512, 1024, 2048],
#     "dataflow_options": ["os", "ws", "is"],
#     "bandwidth_range": [5, 10, 20, 50, 100],
#     # Convolution layer parameter ranges
#     "ifmap_height_range": [7, 13, 14, 27, 28, 56, 112, 224],
#     "ifmap_width_range": [7, 13, 14, 27, 28, 56, 112, 224],
#     "filter_height_range": [1, 3, 5, 7, 11],
#     "filter_width_range": [1, 3, 5, 7, 11],
#     "channels_range": [3, 16, 32, 64, 96, 128, 256, 384, 512],
#     "num_filter_range": [16, 32, 64, 96, 128, 256, 384, 512],
#     "strides_range": [1, 2, 4],
#     # Output paths
#     "output_dir": "./data/raw",
#     "processed_dir": "./data/processed",
# }

DATA_GENERATION_CONFIG = {
    # Number of samples to generate
    "num_samples": 5000,
    # Hardware config parameter ranges
    "array_height_range": [64, 128, 256, 512],
    "array_width_range": [64, 128, 256, 512],
    "ifmap_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
    "filter_sram_sz_kb_range": [256, 512, 1024, 2048, 4096, 6144],
    "ofmap_sram_sz_kb_range": [256, 512, 1024, 2048],
    "dataflow_options": ["os", "ws", "is"],
    "bandwidth_range": [5, 10, 20, 50, 100],
    # Convolution layer parameter ranges
    "ifmap_height_range": [7, 13, 14, 27, 28],
    "ifmap_width_range": [7, 13, 14, 27, 28],
    "filter_height_range": [1, 3, 5, 7],
    "filter_width_range": [1, 3, 5, 7],
    "channels_range": [3, 16, 32],
    "num_filter_range": [16, 32],
    "strides_range": [1, 2],
    # Output paths
    "output_dir": "./data/raw",
    "processed_dir": "./data/processed",
}


# ============== Model Config ==============
MODEL_CONFIG = {
    # Model architecture
    "hidden_dims": [128, 256, 128, 64],
    "dropout_rate": 0.2,
    "activation": "relu",
    # Training parameters
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 10,
    "train_val_test_split": [0.7, 0.15, 0.15],
    # Model save path
    "model_save_path": "./models/scalesim_predictor.pt",
    "scaler_save_path": "./models/feature_scaler.pkl",
}

# ============== Feature Config ==============
# Input features for the model
INPUT_FEATURES = [
    # Hardware config features
    "array_height",
    "array_width",
    "ifmap_sram_sz_kb",
    "filter_sram_sz_kb",
    "ofmap_sram_sz_kb",
    "dataflow_os",  # One-hot encoded
    "dataflow_ws",  # One-hot encoded
    "dataflow_is",  # One-hot encoded
    "bandwidth",
    # Convolution layer features
    "ifmap_height",
    "ifmap_width",
    "filter_height",
    "filter_width",
    "channels",
    "num_filter",
    "strides",
    # Derived features
    "total_macs",  # Computed: output_size * filter_size * channels * num_filter
    "ifmap_size",  # Computed: ifmap_height * ifmap_width * channels
    "filter_size",  # Computed: filter_height * filter_width * channels * num_filter
    "ofmap_size",  # Computed: ofmap_height * ofmap_width * num_filter
    "compute_intensity",  # Computed: total_macs / (ifmap_size + filter_size + ofmap_size)
]

# Output targets (from COMPUTE_REPORT)
OUTPUT_TARGETS = [
    "total_cycles_with_prefetch",
    "total_cycles",
    "stall_cycles",
    "overall_util_percent",
    "mapping_efficiency_percent",
    "compute_util_percent",
]

# ============== Paths Config ==============
PATHS = {
    "base_dir": "./",
    "topology_temp_dir": "./temp/topologies",
    "config_temp_dir": "./temp/configs",
    "results_temp_dir": "./temp/results",
}

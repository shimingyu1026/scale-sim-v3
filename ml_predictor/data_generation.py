"""
Data generation module for SCALE-Sim ML Predictor.
Automatically runs SCALE-Sim with various configurations and collects training data.
"""

import os
import sys
import random
import tempfile
import shutil
import csv
import argparse
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Add parent directory to path to import scalesim
sys.path.insert(0, str(Path(__file__).parent.parent))

from scalesim.scale_sim import scalesim
from ml_predictor.config import DATA_GENERATION_CONFIG, OUTPUT_TARGETS


class DataGenerator:
    """
    Generates training data by running SCALE-Sim simulations with
    various hardware configurations and convolution layer parameters.
    """

    def __init__(self, config: Optional[Dict] = None, seed: Optional[int] = None):
        """
        Initialize the data generator.

        Args:
            config: Configuration dictionary. If None, uses default config.
            seed: Random seed. If None, uses current time for unique seed each run.
        """
        self.config = config or DATA_GENERATION_CONFIG

        # Use time-based seed if not specified for unique runs each time
        if seed is None:
            seed = int(time.time() * 1000) % (2**32)
            print(f"Using time-based seed: {seed}")

        self.seed = seed
        random.seed(seed)

        # Create output directories
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for intermediate files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="scalesim_data_gen_"))

        # Set to track existing configurations (for deduplication)
        self.existing_configs: Set[str] = set()

    def _get_config_signature(self, hw_config: Dict, conv_layer: Dict) -> str:
        """
        Generate a unique signature for a configuration combination.
        Used to detect duplicate configurations.
        """
        sig_parts = [
            str(hw_config.get("array_height", "")),
            str(hw_config.get("array_width", "")),
            str(hw_config.get("ifmap_sram_sz_kb", "")),
            str(hw_config.get("filter_sram_sz_kb", "")),
            str(hw_config.get("ofmap_sram_sz_kb", "")),
            str(hw_config.get("dataflow", "")),
            str(hw_config.get("bandwidth", "")),
            str(conv_layer.get("ifmap_height", "")),
            str(conv_layer.get("ifmap_width", "")),
            str(conv_layer.get("filter_height", "")),
            str(conv_layer.get("filter_width", "")),
            str(conv_layer.get("channels", "")),
            str(conv_layer.get("num_filter", "")),
            str(conv_layer.get("strides", "")),
        ]
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()

    def _load_existing_configs(self, output_file: str):
        """
        Load existing configurations from output file to avoid duplicates.
        """
        self.existing_configs.clear()

        if not Path(output_file).exists():
            return

        try:
            with open(output_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Reconstruct hw_config and conv_layer from row
                    hw_config = {
                        "array_height": row.get("array_height", ""),
                        "array_width": row.get("array_width", ""),
                        "ifmap_sram_sz_kb": row.get("ifmap_sram_sz_kb", ""),
                        "filter_sram_sz_kb": row.get("filter_sram_sz_kb", ""),
                        "ofmap_sram_sz_kb": row.get("ofmap_sram_sz_kb", ""),
                        "dataflow": row.get("dataflow", ""),
                        "bandwidth": row.get("bandwidth", ""),
                    }
                    conv_layer = {
                        "ifmap_height": row.get("ifmap_height", ""),
                        "ifmap_width": row.get("ifmap_width", ""),
                        "filter_height": row.get("filter_height", ""),
                        "filter_width": row.get("filter_width", ""),
                        "channels": row.get("channels", ""),
                        "num_filter": row.get("num_filter", ""),
                        "strides": row.get("strides", ""),
                    }
                    sig = self._get_config_signature(hw_config, conv_layer)
                    self.existing_configs.add(sig)
            print(
                f"Loaded {len(self.existing_configs)} existing configurations from {output_file}"
            )
        except Exception as e:
            print(f"Warning: Could not load existing configs: {e}")

    def _generate_random_config(self) -> Dict:
        """Generate a random hardware configuration."""
        return {
            "array_height": random.choice(self.config["array_height_range"]),
            "array_width": random.choice(self.config["array_width_range"]),
            "ifmap_sram_sz_kb": random.choice(self.config["ifmap_sram_sz_kb_range"]),
            "filter_sram_sz_kb": random.choice(self.config["filter_sram_sz_kb_range"]),
            "ofmap_sram_sz_kb": random.choice(self.config["ofmap_sram_sz_kb_range"]),
            "dataflow": random.choice(self.config["dataflow_options"]),
            "bandwidth": random.choice(self.config["bandwidth_range"]),
        }

    def _generate_random_conv_layer(self) -> Dict:
        """Generate a random convolution layer configuration."""
        ifmap_height = random.choice(self.config["ifmap_height_range"])
        ifmap_width = random.choice(self.config["ifmap_width_range"])
        filter_height = random.choice(self.config["filter_height_range"])
        filter_width = random.choice(self.config["filter_width_range"])
        channels = random.choice(self.config["channels_range"])
        num_filter = random.choice(self.config["num_filter_range"])
        strides = random.choice(self.config["strides_range"])

        # Validate: ensure output size is positive
        ofmap_height = (ifmap_height - filter_height) // strides + 1
        ofmap_width = (ifmap_width - filter_width) // strides + 1

        # If invalid, regenerate with stride=1
        if ofmap_height <= 0 or ofmap_width <= 0:
            strides = 1
            ofmap_height = (ifmap_height - filter_height) // strides + 1
            ofmap_width = (ifmap_width - filter_width) // strides + 1

            # If still invalid, make filter smaller
            if ofmap_height <= 0 or ofmap_width <= 0:
                filter_height = min(filter_height, ifmap_height)
                filter_width = min(filter_width, ifmap_width)
                ofmap_height = (ifmap_height - filter_height) // strides + 1
                ofmap_width = (ifmap_width - filter_width) // strides + 1

        return {
            "layer_name": "Conv",
            "ifmap_height": ifmap_height,
            "ifmap_width": ifmap_width,
            "filter_height": filter_height,
            "filter_width": filter_width,
            "channels": channels,
            "num_filter": num_filter,
            "strides": strides,
            "ofmap_height": ofmap_height,
            "ofmap_width": ofmap_width,
        }

    def _create_config_file(self, hw_config: Dict, sample_id: int) -> str:
        """
        Create a temporary config file for SCALE-Sim.

        Args:
            hw_config: Hardware configuration dictionary.
            sample_id: Unique sample identifier.

        Returns:
            Path to the created config file.
        """
        config_path = self.temp_dir / f"config_{sample_id}.cfg"

        config_content = f"""[general]
run_name = sample_{sample_id}

[architecture_presets]
ArrayHeight:    {hw_config['array_height']}
ArrayWidth:     {hw_config['array_width']}
IfmapSramSzkB:    {hw_config['ifmap_sram_sz_kb']}
FilterSramSzkB:   {hw_config['filter_sram_sz_kb']}
OfmapSramSzkB:    {hw_config['ofmap_sram_sz_kb']}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Dataflow : {hw_config['dataflow']}
Bandwidth : {hw_config['bandwidth']}
ReadRequestBuffer: 512
WriteRequestBuffer: 512

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
UseRamulatorTrace: False
InterfaceBandwidth: USER
"""

        with open(config_path, "w") as f:
            f.write(config_content)

        return str(config_path)

    def _create_topology_file(self, conv_layer: Dict, sample_id: int) -> str:
        """
        Create a temporary topology file for SCALE-Sim.

        Args:
            conv_layer: Convolution layer configuration dictionary.
            sample_id: Unique sample identifier.

        Returns:
            Path to the created topology file.
        """
        topo_path = self.temp_dir / f"topology_{sample_id}.csv"

        topo_content = f"""Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
{conv_layer['layer_name']},{conv_layer['ifmap_height']},{conv_layer['ifmap_width']},{conv_layer['filter_height']},{conv_layer['filter_width']},{conv_layer['channels']},{conv_layer['num_filter']},{conv_layer['strides']},
"""

        with open(topo_path, "w") as f:
            f.write(topo_content)

        return str(topo_path)

    def _create_layout_file(self, conv_layer: Dict, sample_id: int) -> str:
        """
        Create a temporary layout file for SCALE-Sim.

        Args:
            conv_layer: Convolution layer configuration dictionary.
            sample_id: Unique sample identifier.

        Returns:
            Path to the created layout file.
        """
        layout_path = self.temp_dir / f"layout_{sample_id}.csv"

        # Default layout with simple factors (no custom layout optimization)
        layout_content = f"""Layer name, IFMAP Height Intraline Factor, IFMAP Width Intraline Factor, Filter Height Intraline Factor, Filter Width Intraline Factor, Channel Intraline Factor, Num Filter Intraline Factor, IFMAP Height Intraline Order, IFMAP Width Intraline Order, Channel Intraline Order, IFMAP Height Interline Order, IFMAP Width Interline Order, Channel Interline Order, Num Filter Intraline Order, Channel Intraline Order, Filter Height Intraline Order, Filter Width Intraline Order, Num Filter Interline Order, Channel Interline Order, Filter Height Interline Order, Filter Width Interline Order,
{conv_layer['layer_name']}, 1, 1, 1, 1, 1, 1, 0, 1, 2, 4, 5, 3, 3, 2, 1, 0, 4, 5, 6, 7,
"""

        with open(layout_path, "w") as f:
            f.write(layout_content)

        return str(layout_path)

    def _run_single_simulation(self, sample_id: int) -> Optional[Dict]:
        """
        Run a single SCALE-Sim simulation and collect results.

        Args:
            sample_id: Unique sample identifier.

        Returns:
            Dictionary containing input features and output targets, or None if failed.
        """
        try:
            # Generate random configurations
            hw_config = self._generate_random_config()
            conv_layer = self._generate_random_conv_layer()

            # Create temporary files
            config_path = self._create_config_file(hw_config, sample_id)
            topo_path = self._create_topology_file(conv_layer, sample_id)
            layout_path = self._create_layout_file(conv_layer, sample_id)
            results_path = self.temp_dir / f"results_{sample_id}"
            results_path.mkdir(exist_ok=True)

            # Run SCALE-Sim simulation
            sim = scalesim(
                save_disk_space=True,
                verbose=False,
                config=config_path,
                topology=topo_path,
                layout=layout_path,
                input_type_gemm=False,
            )
            sim.run_scale(top_path=str(results_path))

            # Parse results from COMPUTE_REPORT.csv
            compute_report_path = (
                results_path / f"sample_{sample_id}" / "COMPUTE_REPORT.csv"
            )

            if not compute_report_path.exists():
                # Try alternative path
                for subdir in results_path.iterdir():
                    if subdir.is_dir():
                        alt_path = subdir / "COMPUTE_REPORT.csv"
                        if alt_path.exists():
                            compute_report_path = alt_path
                            break

            if not compute_report_path.exists():
                print(f"Warning: COMPUTE_REPORT.csv not found for sample {sample_id}")
                return None

            # Read the compute report
            with open(compute_report_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                data_row = next(reader)

            # Parse the output values
            # Expected columns: LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles,
            #                   Overall Util %, Mapping Efficiency %, Compute Util %
            total_cycles_with_prefetch = float(data_row[1].strip())
            total_cycles = float(data_row[2].strip())
            stall_cycles = float(data_row[3].strip())
            overall_util = float(data_row[4].strip())
            mapping_efficiency = float(data_row[5].strip())
            compute_util = float(data_row[6].strip().rstrip(","))

            # Calculate derived features
            ofmap_height = conv_layer["ofmap_height"]
            ofmap_width = conv_layer["ofmap_width"]
            total_macs = (
                ofmap_height
                * ofmap_width
                * conv_layer["filter_height"]
                * conv_layer["filter_width"]
                * conv_layer["channels"]
                * conv_layer["num_filter"]
            )
            ifmap_size = (
                conv_layer["ifmap_height"]
                * conv_layer["ifmap_width"]
                * conv_layer["channels"]
            )
            filter_size = (
                conv_layer["filter_height"]
                * conv_layer["filter_width"]
                * conv_layer["channels"]
                * conv_layer["num_filter"]
            )
            ofmap_size = ofmap_height * ofmap_width * conv_layer["num_filter"]
            compute_intensity = (
                total_macs / (ifmap_size + filter_size + ofmap_size)
                if (ifmap_size + filter_size + ofmap_size) > 0
                else 0
            )

            # Build result dictionary
            result = {
                # Hardware config features
                "array_height": hw_config["array_height"],
                "array_width": hw_config["array_width"],
                "ifmap_sram_sz_kb": hw_config["ifmap_sram_sz_kb"],
                "filter_sram_sz_kb": hw_config["filter_sram_sz_kb"],
                "ofmap_sram_sz_kb": hw_config["ofmap_sram_sz_kb"],
                "dataflow": hw_config["dataflow"],
                "bandwidth": hw_config["bandwidth"],
                # Convolution layer features
                "ifmap_height": conv_layer["ifmap_height"],
                "ifmap_width": conv_layer["ifmap_width"],
                "filter_height": conv_layer["filter_height"],
                "filter_width": conv_layer["filter_width"],
                "channels": conv_layer["channels"],
                "num_filter": conv_layer["num_filter"],
                "strides": conv_layer["strides"],
                # Derived features
                "ofmap_height": ofmap_height,
                "ofmap_width": ofmap_width,
                "total_macs": total_macs,
                "ifmap_size": ifmap_size,
                "filter_size": filter_size,
                "ofmap_size": ofmap_size,
                "compute_intensity": compute_intensity,
                # Output targets
                "total_cycles_with_prefetch": total_cycles_with_prefetch,
                "total_cycles": total_cycles,
                "stall_cycles": stall_cycles,
                "overall_util_percent": overall_util,
                "mapping_efficiency_percent": mapping_efficiency,
                "compute_util_percent": compute_util,
            }

            # Cleanup temporary files for this sample
            os.remove(config_path)
            os.remove(topo_path)
            os.remove(layout_path)
            shutil.rmtree(results_path, ignore_errors=True)

            return result

        except Exception as e:
            print(f"Error running simulation for sample {sample_id}: {e}")
            return None

    def _run_single_simulation_wrapper(self, sample_id):
        """Wrapper for multiprocessing to handle seeds correctly"""
        # Re-seed for each process to ensure randomness
        # In multiprocessing, child processes might inherit state or have same seed
        # Mix the base seed with sample_id and process ID
        np.random.seed((self.seed + sample_id + os.getpid()) % 2**32)
        random.seed((self.seed + sample_id + os.getpid()) % 2**32)
        return self._run_single_simulation(sample_id)

    def generate(
        self,
        num_samples: Optional[int] = None,
        output_file: Optional[str] = None,
        show_progress: bool = True,
        num_workers: int = 1,
    ) -> str:
        """
        Generate training data by running multiple SCALE-Sim simulations.
        Appends to existing file and skips duplicate configurations.

        Args:
            num_samples: Number of NEW samples to generate. If None, uses config value.
            output_file: Output CSV file path. If None, uses default path.
            show_progress: Whether to show progress during generation.
            num_workers: Number of parallel workers.

        Returns:
            Path to the generated CSV file.
        """
        num_samples = num_samples or self.config["num_samples"]
        output_file = output_file or str(self.output_dir / "training_data.csv")

        # Determine number of workers
        if num_workers < 1:
            num_workers = max(1, mp.cpu_count() - 1)
        print(f"Using {num_workers} workers for data generation")

        # Load existing configurations to avoid duplicates
        self._load_existing_configs(output_file)

        print(f"Generating {num_samples} new samples (skipping duplicates)...")

        results = []
        successful = 0
        skipped = 0
        failed = 0

        # Progress bar with detailed stats
        pbar = tqdm(
            total=num_samples,
            desc="Generating samples",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, Success:{postfix[0]}, Skip:{postfix[1]}, Fail:{postfix[2]}]",
            postfix=[successful, skipped, failed],
        )

        with mp.Pool(processes=num_workers) as pool:
            # We urge to generate more than needed to account for duplicates/failures
            # Since we can't easily dynamic loop with pool, we submit a larger batch
            # or use a generator. imap is good.

            # Create an infinite generator of task IDs
            def task_id_generator():
                i = 0
                while True:
                    yield i
                    i += 1

            # Use imap to process tasks as they complete
            # We map a wrapper function to handle exceptions gracefully
            try:
                for result in pool.imap(
                    self._run_single_simulation_wrapper, task_id_generator()
                ):
                    if successful >= num_samples:
                        break

                    if result is not None:
                        # Check for duplicate
                        hw_config = {
                            k: result[k]
                            for k in [
                                "array_height",
                                "array_width",
                                "ifmap_sram_sz_kb",
                                "filter_sram_sz_kb",
                                "ofmap_sram_sz_kb",
                                "dataflow",
                                "bandwidth",
                            ]
                        }
                        conv_layer = {
                            k: result[k]
                            for k in [
                                "ifmap_height",
                                "ifmap_width",
                                "filter_height",
                                "filter_width",
                                "channels",
                                "num_filter",
                                "strides",
                            ]
                        }
                        sig = self._get_config_signature(hw_config, conv_layer)

                        if sig in self.existing_configs:
                            skipped += 1
                            pbar.postfix[1] = skipped
                            pbar.update(0)
                            continue

                        # Add to existing set and results
                        self.existing_configs.add(sig)
                        results.append(result)
                        successful += 1
                        pbar.postfix[0] = successful
                        pbar.update(1)
                    else:
                        failed += 1
                        pbar.postfix[2] = failed
                        pbar.update(0)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving collected data...")
                pool.terminate()
                pool.join()

        pbar.close()
        print(
            f"\nGeneration complete. Success: {successful}, Skipped: {skipped}, Failed: {failed}"
        )

        # Append to CSV (or create new)
        if results:
            file_exists = Path(output_file).exists()
            fieldnames = list(results[0].keys())

            with open(output_file, "a" if file_exists else "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results)

            total_samples = len(self.existing_configs)
            print(f"Data appended to: {output_file} (total samples: {total_samples})")
        else:
            print("Warning: No new samples generated.")

        # Cleanup temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        return output_file

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate training data for SCALE-Sim ML Predictor"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/training_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generator = DataGenerator(seed=args.seed)
    try:
        generator.generate(num_samples=args.num_samples, output_file=args.output)
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()

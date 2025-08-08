#!/bin/bash

python3 create_action_count.py --saved_folder /home/singh16/work/hespas/scale-sim-v3/test_runs --run_name scale_example_run_256x256_os --arch_name systolic_array --SRAM_row_size 2 --DRAM_row_size 2 --config /home/singh16/work/hespas/scale-sim-v3/configs/sweep_study/256_os.cfg

cp /home/singh16/work/hespas/scale-sim-v3/test_runs/scale_example_run_256x256_os/action_count.yaml ./accelergy_input/action_count.yaml

rm -rf /home/singh16/work/hespas/scale-sim-v3/rundir-accelergy/output/scale_sim_output_scale_example_run_256x256_os

mv /home/singh16/work/hespas/scale-sim-v3/test_runs/scale_example_run_256x256_os  /home/singh16/work/hespas/scale-sim-v3/rundir-accelergy/output/scale_sim_output_scale_example_run_256x256_os


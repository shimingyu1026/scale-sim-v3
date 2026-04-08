import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliSaveTraceFlagTest(unittest.TestCase):
    def write_inputs(self, workdir, run_name):
        config_path = workdir / "toy.cfg"
        topology_path = workdir / "toy_topology.csv"
        layout_path = workdir / "toy_layout.csv"

        config_path.write_text(
            "\n".join(
                [
                    "[general]",
                    f"run_name = {run_name}",
                    "",
                    "[architecture_presets]",
                    "ArrayHeight: 2",
                    "ArrayWidth: 2",
                    "IfmapSramSzkB: 1",
                    "FilterSramSzkB: 1",
                    "OfmapSramSzkB: 1",
                    "IfmapOffset: 0",
                    "FilterOffset: 10000000",
                    "OfmapOffset: 20000000",
                    "Bandwidth : 10",
                    "Dataflow : ws",
                    "MemoryBanks: 1",
                    "ReadRequestBuffer: 32",
                    "WriteRequestBuffer: 32",
                    "",
                    "[layout]",
                    "IfmapCustomLayout: False",
                    "IfmapSRAMBankBandwidth: 10",
                    "IfmapSRAMBankNum: 1",
                    "IfmapSRAMBankPort: 1",
                    "FilterCustomLayout: False",
                    "FilterSRAMBankBandwidth: 10",
                    "FilterSRAMBankNum: 1",
                    "FilterSRAMBankPort: 1",
                    "",
                    "[sparsity]",
                    "SparsitySupport : false",
                    "SparseRep : ellpack_block",
                    "OptimizedMapping : false",
                    "BlockSize : 8",
                    "RandomNumberGeneratorSeed : 40",
                    "",
                    "[run_presets]",
                    "InterfaceBandwidth: CALC",
                    "UseRamulatorTrace: False",
                    "SRAM_row_size: 2",
                    "SRAM_bank_size: 5",
                    "DRAM_row_size: 2",
                    "DRAM_bank_size: 5",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        topology_path.write_text(
            "\n".join(
                [
                    "Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,Channels,Num Filter,Strides,",
                    "toy,4,4,1,1,1,2,1,",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        layout_path.write_text(
            "\n".join(
                [
                    "Layer name, IFMAP Height Intraline Factor, IFMAP Width Intraline Factor, Filter Height Intraline Factor, Filter Width Intraline Factor, Channel Intraline Factor, Num Filter Intraline Factor, IFMAP Height Intraline Order, IFMAP Width Intraline Order, Channel Intraline Order, IFMAP Height Interline Order, IFMAP Width Interline Order, Channel Interline Order, Num Filter Intraline Order, Channel Intraline Order, Filter Height Intraline Order, Filter Width Intraline Order, Num Filter Interline Order, Channel Interline Order, Filter Height Interline Order, Filter Width Interline Order,",
                    "toy,1,1,1,1,1,1,0,1,2,3,4,5,0,1,2,3,4,5,6,",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return config_path, topology_path, layout_path

    def run_cli(self, command, *, save_flag, run_name, expect_traces):
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            config_path, topology_path, layout_path = self.write_inputs(workdir, run_name)
            output_root = workdir / "outputs"

            completed = subprocess.run(
                command
                + [
                    "-c",
                    str(config_path),
                    "-t",
                    str(topology_path),
                    "-l",
                    str(layout_path),
                    "-p",
                    str(output_root),
                    "-s",
                    save_flag,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=(
                    f"command failed: {' '.join(command)}\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                ),
            )

            report_dir = output_root / run_name
            self.assertTrue((report_dir / "COMPUTE_REPORT.csv").exists())
            self.assertTrue((report_dir / "BANDWIDTH_REPORT.csv").exists())
            self.assertTrue((report_dir / "DETAILED_ACCESS_REPORT.csv").exists())
            self.assertTrue((report_dir / "REPEAT_CYCLE.csv").exists())
            if expect_traces:
                self.assertTrue((report_dir / "layer0").is_dir())
                self.assertTrue((report_dir / "layer0" / "IFMAP_SRAM_TRACE.csv").exists())
                self.assertTrue((report_dir / "layer0" / "FILTER_SRAM_TRACE.csv").exists())
                self.assertTrue((report_dir / "layer0" / "OFMAP_SRAM_TRACE.csv").exists())
                self.assertTrue((report_dir / "layer0" / "IFMAP_DRAM_TRACE.csv").exists())
                self.assertTrue((report_dir / "layer0" / "FILTER_DRAM_TRACE.csv").exists())
                self.assertTrue((report_dir / "layer0" / "OFMAP_DRAM_TRACE.csv").exists())
            else:
                self.assertFalse((report_dir / "layer0").exists())
                self.assertEqual(list(report_dir.rglob("*TRACE.csv")), [])

    def assert_trace_behavior(self, command):
        self.run_cli(command, save_flag="N", run_name="no_trace_run", expect_traces=False)
        self.run_cli(command, save_flag="Y", run_name="with_trace_run", expect_traces=True)

    def test_root_entrypoint_honors_trace_flag(self):
        self.assert_trace_behavior([sys.executable, str(REPO_ROOT / "scale.py")])

    def test_module_entrypoint_honors_trace_flag(self):
        self.assert_trace_behavior([sys.executable, "-m", "scalesim.scale"])


if __name__ == "__main__":
    unittest.main()

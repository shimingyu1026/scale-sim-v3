import csv
import tempfile
import unittest
from pathlib import Path

from scalesim.layout_utils import layouts
from scalesim.scale_config import scale_config
from scalesim.simulator import simulator
from scalesim.topology_utils import topologies


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase10SmokeTest(unittest.TestCase):
    def make_config(self, dataflow="ws", run_name="phase10_smoke"):
        config = scale_config()
        config.force_valid()
        config.set_dataflow(dataflow)
        config.set_arr_dims(rows=8, cols=8)
        config.set_buffer_sizes_kb(ifmap_size_kb=1024, filter_size_kb=1024, ofmap_size_kb=1024)
        config.set_bw_mode_to_calc()
        config.using_ifmap_custom_layout = False
        config.using_filter_custom_layout = False
        config.run_name = run_name
        return config

    def load_rows(self, csv_path):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def run_smoke(self, topology_relpath, *, mnk_inputs=False, dataflow="ws", run_name="phase10_smoke"):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        topo = topologies()
        topo.load_arrays(str(REPO_ROOT / topology_relpath), mnk_inputs=mnk_inputs)

        sim = simulator()
        sim.set_params(
            config_obj=self.make_config(dataflow=dataflow, run_name=run_name),
            topo_obj=topo,
            layout_obj=layouts(),
            top_path=temp_dir.name,
            verbosity=False,
            save_trace=False,
        )
        sim.run()

        report_dir = Path(temp_dir.name) / run_name
        return topo, report_dir

    def assert_report_set_exists(self, report_dir, expected_layers):
        compute_path = report_dir / "COMPUTE_REPORT.csv"
        bandwidth_path = report_dir / "BANDWIDTH_REPORT.csv"
        detail_path = report_dir / "DETAILED_ACCESS_REPORT.csv"
        repeat_path = report_dir / "REPEAT_CYCLE.csv"

        self.assertTrue(compute_path.exists())
        self.assertTrue(bandwidth_path.exists())
        self.assertTrue(detail_path.exists())
        self.assertTrue(repeat_path.exists())

        compute_rows = self.load_rows(compute_path)
        bandwidth_rows = self.load_rows(bandwidth_path)
        detail_rows = self.load_rows(detail_path)

        self.assertEqual(len(compute_rows), expected_layers + 1)
        self.assertEqual(len(bandwidth_rows), expected_layers + 1)
        self.assertEqual(len(detail_rows), expected_layers + 1)

        for row_id in range(1, expected_layers + 1):
            self.assertGreater(float(compute_rows[row_id][1]), 0.0)
            self.assertGreater(float(compute_rows[row_id][2]), 0.0)
            self.assertGreater(float(detail_rows[row_id][3]), 0.0)
            self.assertGreater(float(detail_rows[row_id][6]), 0.0)
            self.assertGreater(float(detail_rows[row_id][9]), 0.0)

    def test_conv_batch_topology_smoke(self):
        topo, report_dir = self.run_smoke(
            "topologies/conv_nets/conv_batch_4.csv",
            dataflow="ws",
            run_name="conv_batch_smoke",
        )

        self.assertEqual(topo.get_num_layers(), 2)
        self.assertEqual(topo.get_layer_batch_size(0), 4)
        self.assertEqual(topo.get_layer_batch_size(1), 4)
        self.assert_report_set_exists(report_dir, expected_layers=2)

    def test_gemm_batch_topology_smoke(self):
        topo, report_dir = self.run_smoke(
            "topologies/GEMM_mnk/gemm_batch_8.csv",
            mnk_inputs=True,
            dataflow="os",
            run_name="gemm_batch_smoke",
        )

        self.assertEqual(topo.get_num_layers(), 2)
        self.assertEqual(topo.get_layer_batch_size(0), 8)
        self.assertEqual(topo.get_layer_batch_size(1), 8)
        self.assert_report_set_exists(report_dir, expected_layers=2)

    def test_repository_batch_size_header_parser_smoke(self):
        topo = topologies()
        topo.load_arrays(str(REPO_ROOT / "topologies/llama/llama3b.csv"))

        self.assertEqual(topo.get_num_layers(), 51)
        for layer_id in range(topo.get_num_layers()):
            self.assertEqual(topo.get_layer_batch_size(layer_id), 1)


if __name__ == "__main__":
    unittest.main()

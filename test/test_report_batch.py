import csv
import tempfile
import unittest
from pathlib import Path

from scalesim.layout_utils import layouts
from scalesim.scale_config import scale_config
from scalesim.simulator import simulator
from scalesim.single_layer_sim import single_layer_sim
from scalesim.topology_utils import topologies


TOY_NUM_COMPUTE_PER_BATCH = 32
TOY_NUM_MAC_UNITS = 8


class BatchReportTestBase(unittest.TestCase):
    def write_temp_topology(self, batch_size):
        temp_dir = tempfile.TemporaryDirectory()
        topo_path = Path(temp_dir.name) / "toy.csv"
        topo_path.write_text(
            "Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,Channels,Num Filter,Strides,Batch,\n"
            f"toy,4,4,1,1,1,2,1,{batch_size},\n",
            encoding="utf-8",
        )
        self.addCleanup(temp_dir.cleanup)
        return topo_path

    def make_config(self, dataflow):
        config = scale_config()
        config.force_valid()
        config.set_dataflow(dataflow)
        config.set_arr_dims(rows=1, cols=8)
        config.set_buffer_sizes_kb(ifmap_size_kb=1024, filter_size_kb=1024, ofmap_size_kb=1024)
        config.set_bw_mode_to_calc()
        config.using_ifmap_custom_layout = False
        config.using_filter_custom_layout = False
        return config

    def load_topology(self, batch_size):
        topo = topologies()
        topo.load_arrays(str(self.write_temp_topology(batch_size)))
        return topo

    def run_single_layer(self, dataflow, batch_size):
        sim = single_layer_sim()
        sim.set_params(
            layer_id=0,
            config_obj=self.make_config(dataflow),
            topology_obj=self.load_topology(batch_size),
            layout_obj=layouts(),
            verbose=False,
        )
        sim.run()
        return sim

    def parse_csv_rows(self, path):
        with path.open(newline="", encoding="utf-8") as handle:
            return [[cell.strip() for cell in row if cell.strip()] for row in csv.reader(handle)]

    def assert_single_layer_report_formulas(self, sim, expected_num_compute):
        compute = sim.get_compute_report_items()
        bandwidth = sim.get_bandwidth_report_items()
        detail = sim.get_detail_report_items()

        total_cycles = compute[1]
        self.assertEqual(sim.num_compute, expected_num_compute)
        self.assertEqual(detail[2], sim.compute_system.get_ifmap_requests())
        self.assertEqual(detail[5], sim.compute_system.get_filter_requests())
        self.assertEqual(detail[8], sim.compute_system.get_ofmap_requests())
        self.assertEqual(compute[0], int(detail[16] - min(detail[9], detail[12])))
        self.assertAlmostEqual(
            compute[3],
            expected_num_compute * 100 / (total_cycles * sim.num_mac_unit),
        )
        self.assertAlmostEqual(bandwidth[0], detail[2] / total_cycles)
        self.assertAlmostEqual(bandwidth[1], detail[5] / total_cycles)
        self.assertAlmostEqual(bandwidth[2], detail[8] / total_cycles)
        self.assertAlmostEqual(bandwidth[3], detail[11] / (detail[10] - detail[9] + 1))
        self.assertAlmostEqual(bandwidth[4], detail[14] / (detail[13] - detail[12] + 1))
        self.assertAlmostEqual(bandwidth[5], detail[17] / (detail[16] - detail[15] + 1))


class SingleLayerBatchReportTest(BatchReportTestBase):
    def test_single_layer_reports_scale_consistently_with_batch(self):
        filter_sram_scaling = {
            "os": 4,
            "ws": 1,
            "is": 4,
        }

        for dataflow in ("os", "ws", "is"):
            with self.subTest(dataflow=dataflow):
                batch_one = self.run_single_layer(dataflow, 1)
                batch_four = self.run_single_layer(dataflow, 4)

                self.assert_single_layer_report_formulas(batch_one, TOY_NUM_COMPUTE_PER_BATCH)
                self.assert_single_layer_report_formulas(batch_four, TOY_NUM_COMPUTE_PER_BATCH * 4)

                detail_one = batch_one.get_detail_report_items()
                detail_four = batch_four.get_detail_report_items()

                self.assertEqual(batch_four.num_compute, batch_one.num_compute * 4)
                self.assertEqual(detail_one[11], 16)
                self.assertEqual(detail_four[11], 64)
                self.assertEqual(detail_one[14], 2)
                self.assertEqual(detail_four[14], 2)
                self.assertEqual(detail_one[17], 32)
                self.assertEqual(detail_four[17], 128)
                self.assertEqual(detail_four[2], detail_one[2] * 4)
                self.assertEqual(detail_four[5], detail_one[5] * filter_sram_scaling[dataflow])
                self.assertEqual(detail_four[8], detail_one[8] * 4)
                self.assertEqual(detail_four[11], detail_one[11] * 4)
                self.assertEqual(detail_four[14], detail_one[14])
                self.assertEqual(detail_four[17], detail_one[17] * 4)


class SimulatorBatchReportSmokeTest(BatchReportTestBase):
    def test_simulator_writes_batch_consistent_csv_reports(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_root = Path(temp_dir.name)

        config = self.make_config("os")
        config.run_name = "batch_report_smoke"
        topo = self.load_topology(4)

        sim = simulator()
        sim.set_params(
            config_obj=config,
            topo_obj=topo,
            layout_obj=layouts(),
            top_path=str(output_root),
            verbosity=False,
            save_trace=False,
        )
        sim.run()

        report_dir = output_root / config.get_run_name()
        compute_path = report_dir / "COMPUTE_REPORT.csv"
        bandwidth_path = report_dir / "BANDWIDTH_REPORT.csv"
        detail_path = report_dir / "DETAILED_ACCESS_REPORT.csv"
        repeat_path = report_dir / "REPEAT_CYCLE.csv"

        self.assertTrue(compute_path.exists())
        self.assertTrue(bandwidth_path.exists())
        self.assertTrue(detail_path.exists())
        self.assertTrue(repeat_path.exists())

        compute_row = [float(cell) for cell in self.parse_csv_rows(compute_path)[1]]
        bandwidth_row = [float(cell) for cell in self.parse_csv_rows(bandwidth_path)[1]]
        detail_row = [float(cell) for cell in self.parse_csv_rows(detail_path)[1]]

        self.assertEqual(compute_row[0], 0.0)
        self.assertEqual(compute_row[1], 1575.0)
        self.assertEqual(compute_row[2], 511.0)
        self.assertEqual(compute_row[3], 0.0)
        self.assertAlmostEqual(
            compute_row[4],
            TOY_NUM_COMPUTE_PER_BATCH * 4 * 100 / (compute_row[2] * TOY_NUM_MAC_UNITS),
        )
        self.assertAlmostEqual(compute_row[5], 25.0)
        self.assertAlmostEqual(compute_row[6], 3.125)

        self.assertEqual(detail_row[3], 64.0)
        self.assertEqual(detail_row[6], 128.0)
        self.assertEqual(detail_row[9], 704.0)
        self.assertEqual(detail_row[12], 64.0)
        self.assertEqual(detail_row[15], 2.0)
        self.assertEqual(detail_row[18], 128.0)

        total_cycles = compute_row[2]
        self.assertAlmostEqual(bandwidth_row[1], detail_row[3] / total_cycles)
        self.assertAlmostEqual(bandwidth_row[2], detail_row[6] / total_cycles)
        self.assertAlmostEqual(bandwidth_row[3], detail_row[9] / total_cycles)
        self.assertAlmostEqual(
            bandwidth_row[4],
            detail_row[12] / (detail_row[11] - detail_row[10] + 1),
        )
        self.assertAlmostEqual(
            bandwidth_row[5],
            detail_row[15] / (detail_row[14] - detail_row[13] + 1),
        )
        self.assertAlmostEqual(
            bandwidth_row[6],
            detail_row[18] / (detail_row[17] - detail_row[16] + 1),
        )


if __name__ == "__main__":
    unittest.main()

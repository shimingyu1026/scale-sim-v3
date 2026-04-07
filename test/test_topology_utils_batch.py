import tempfile
import unittest
from pathlib import Path

from scalesim.topology_utils import topologies


REPO_ROOT = Path(__file__).resolve().parents[1]


class TopologyUtilsBatchParsingTest(unittest.TestCase):
    def load_topology(self, relative_path, mnk_inputs=False):
        topo = topologies()
        topo.load_arrays(str(REPO_ROOT / relative_path), mnk_inputs=mnk_inputs)
        return topo

    def write_temp_topology(self, content):
        temp_dir = tempfile.TemporaryDirectory()
        topo_path = Path(temp_dir.name) / "topology.csv"
        topo_path.write_text(content, encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return topo_path

    def test_conv_batch_fixtures_parse(self):
        cases = [
            ("topologies/conv_nets/conv_batch_1.csv", 1, [1, 1]),
            ("topologies/conv_nets/conv_batch_4.csv", 4, [1, 1]),
            ("topologies/conv_nets/conv_batch_4_sparse.csv", 4, [2, 4]),
        ]
        for relative_path, expected_batch, expected_sparsity in cases:
            with self.subTest(path=relative_path):
                topo = self.load_topology(relative_path)
                self.assertEqual(topo.get_layer_batch_size(0), expected_batch)
                self.assertEqual(topo.get_layer_sparsity_ratio(0), expected_sparsity)

    def test_gemm_batch_fixtures_parse(self):
        cases = [
            ("topologies/GEMM_mnk/gemm_batch_1.csv", 1, [1, 1]),
            ("topologies/GEMM_mnk/gemm_batch_8.csv", 8, [1, 1]),
            ("topologies/GEMM_mnk/gemm_batch_8_sparse.csv", 8, [2, 4]),
        ]
        for relative_path, expected_batch, expected_sparsity in cases:
            with self.subTest(path=relative_path):
                topo = self.load_topology(relative_path, mnk_inputs=True)
                self.assertEqual(topo.get_layer_batch_size(0), expected_batch)
                self.assertEqual(topo.get_layer_sparsity_ratio(0), expected_sparsity)

    def test_legacy_conv_topology_defaults_batch_to_one(self):
        topo = self.load_topology("topologies/conv_nets/alexnet.csv")
        self.assertEqual(topo.get_layer_batch_size(0), 1)
        self.assertEqual(topo.get_layer_sparsity_ratio(0), [1, 1])

    def test_existing_repository_batch_header_files_load(self):
        cases = [
            "topologies/llama/llama3b.csv",
            "topologies/transformer/transformer_fwd.csv",
        ]
        for relative_path in cases:
            with self.subTest(path=relative_path):
                topo = self.load_topology(relative_path)
                self.assertGreater(topo.get_num_layers(), 0)
                self.assertEqual(topo.get_layer_batch_size(0), 1)

    def test_batch_headers_accept_bom_lowercase_and_extra_spaces(self):
        topo_path = self.write_temp_topology(
            "\ufeff Layer Name , IFMAP Height , IFMAP Width , Filter Height , "
            "Filter Width , Channels , Num Filters , Strides , batch , sparsity ,\n"
            "conv1,8,8,3,3,3,16,1,4,2:4,\n"
        )
        topo = topologies()
        topo.load_arrays(str(topo_path))

        self.assertEqual(topo.get_layer_batch_size(0), 4)
        self.assertEqual(topo.get_layer_sparsity_ratio(0), [2, 4])

    def test_invalid_batch_values_are_rejected(self):
        invalid_values = ["0", "-3", "two"]
        for invalid_value in invalid_values:
            with self.subTest(batch=invalid_value):
                topo_path = self.write_temp_topology(
                    "Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,"
                    "Channels,Num Filter,Strides,Batch Size,\n"
                    "conv1,8,8,3,3,3,16,1,{},\n".format(invalid_value)
                )
                topo = topologies()
                with self.assertRaisesRegex(ValueError, "Invalid batch value"):
                    topo.load_arrays(str(topo_path))

    def test_mixed_batch_topology_is_rejected(self):
        topo = topologies()
        with self.assertRaisesRegex(ValueError, "Mixed per-layer batch sizes"):
            topo.load_arrays(str(REPO_ROOT / "topologies/conv_nets/conv_batch_mixed.csv"))

    def test_conv_write_roundtrip_preserves_batch_and_sparsity(self):
        topo = self.load_topology("topologies/conv_nets/conv_batch_4_sparse.csv")

        with tempfile.TemporaryDirectory() as temp_dir:
            topo.write_topo_file(path=temp_dir, filename="roundtrip.csv")
            reloaded = topologies()
            reloaded.load_arrays(str(Path(temp_dir) / "roundtrip.csv"))

        self.assertEqual(reloaded.get_layer_batch_size(0), 4)
        self.assertEqual(reloaded.get_layer_sparsity_ratio(0), [2, 4])
        self.assertEqual(reloaded.get_layer_batch_size(1), 4)
        self.assertEqual(reloaded.get_layer_sparsity_ratio(1), [1, 4])

    def test_gemm_write_roundtrip_preserves_batch_and_sparsity(self):
        topo = self.load_topology("topologies/GEMM_mnk/gemm_batch_8_sparse.csv", mnk_inputs=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            topo.write_topo_file(path=temp_dir, filename="roundtrip.csv")
            reloaded = topologies()
            reloaded.load_arrays(str(Path(temp_dir) / "roundtrip.csv"), mnk_inputs=True)

        self.assertEqual(reloaded.get_layer_batch_size(0), 8)
        self.assertEqual(reloaded.get_layer_sparsity_ratio(0), [2, 4])
        self.assertEqual(reloaded.get_layer_num_filters(0), 32)
        self.assertEqual(reloaded.get_layer_batch_size(1), 8)
        self.assertEqual(reloaded.get_layer_sparsity_ratio(1), [1, 4])

    def test_load_layer_params_from_list_preserves_batch_and_sparsity(self):
        topo = topologies()
        topo.load_layer_params_from_list("toy", [4, 4, 1, 1, 1, 2, 1, 1, 2, "2:4"])

        self.assertEqual(topo.get_num_layers(), 1)
        self.assertEqual(topo.get_layer_batch_size(0), 2)
        self.assertEqual(topo.get_layer_sparsity_ratio(0), [2, 4])
        self.assertEqual(topo.get_layer_ofmap_dims(0), [4, 4])

    def test_append_topo_entry_from_list_preserves_batch_and_sparsity(self):
        topo = topologies()
        topo.append_topo_entry_from_list(["toy", 4, 4, 1, 1, 1, 2, 1, 2, "2:4"])

        self.assertEqual(topo.get_num_layers(), 1)
        self.assertEqual(topo.get_layer_batch_size(0), 2)
        self.assertEqual(topo.get_layer_sparsity_ratio(0), [2, 4])
        self.assertEqual(topo.get_layer_params(0)[0], "toy")

    def test_append_layer_entry_roundtrip_preserves_batch_and_sparsity(self):
        topo = topologies()
        topo.append_layer_entry(["toy", 4, 4, 1, 1, 1, 2, 1, 1, 2, 2, 4])

        with tempfile.TemporaryDirectory() as temp_dir:
            topo.write_topo_file(path=temp_dir, filename="roundtrip.csv")
            reloaded = topologies()
            reloaded.load_arrays(str(Path(temp_dir) / "roundtrip.csv"))

        self.assertEqual(reloaded.get_num_layers(), 1)
        self.assertEqual(reloaded.get_layer_batch_size(0), 2)
        self.assertEqual(reloaded.get_layer_sparsity_ratio(0), [2, 4])
        self.assertEqual(reloaded.get_layer_ofmap_dims(0), [4, 4])

    def test_conv_derived_metrics_include_batch(self):
        topo = self.load_topology("topologies/conv_nets/conv_batch_4.csv")

        self.assertEqual(topo.get_layer_ofmap_dims(0), [6, 6])
        self.assertEqual(topo.get_layer_num_ofmap_px(0), 4 * 6 * 6 * 16)
        self.assertEqual(topo.get_layer_mac_ops(0), 4 * 6 * 6 * 3 * 3 * 3 * 16)
        self.assertEqual(topo.get_layer_num_ofmap_px(1), 4 * 4 * 4 * 32)
        self.assertEqual(topo.get_layer_mac_ops(1), 4 * 4 * 4 * 1 * 1 * 16 * 32)
        self.assertEqual(topo.get_all_mac_ops(), (4 * 6 * 6 * 3 * 3 * 3 * 16) + (4 * 4 * 4 * 1 * 1 * 16 * 32))

    def test_gemm_derived_metrics_include_batch(self):
        topo = self.load_topology("topologies/GEMM_mnk/gemm_batch_8.csv", mnk_inputs=True)

        self.assertEqual(topo.get_layer_ofmap_dims(0), [16, 1])
        self.assertEqual(topo.get_layer_num_ofmap_px(0), 8 * 16 * 32)
        self.assertEqual(topo.get_layer_mac_ops(0), 8 * 16 * 32 * 8)
        self.assertEqual(topo.get_transformed_mnk_dimensions(), [[8 * 16, 32, 8], [8 * 8, 16, 4]])

    def test_spatiotemporal_params_use_batched_work(self):
        topo = self.load_topology("topologies/conv_nets/conv_batch_4.csv")

        self.assertEqual(topo.calc_spatio_temporal_params(df="os", layer_id=0), (4 * 6 * 6, 16, 27))
        self.assertEqual(topo.calc_spatio_temporal_params(df="ws", layer_id=0), (27, 16, 4 * 6 * 6))
        self.assertEqual(topo.calc_spatio_temporal_params(df="is", layer_id=0), (27, 4 * 6 * 6, 16))

    def test_batch_one_metrics_match_historical_values(self):
        topo = self.load_topology("topologies/conv_nets/alexnet.csv")

        self.assertEqual(topo.get_layer_batch_size(0), 1)
        self.assertEqual(topo.get_layer_ofmap_dims(0), [55, 55])
        self.assertEqual(topo.get_layer_num_ofmap_px(0), 55 * 55 * 96)
        self.assertEqual(topo.get_layer_mac_ops(0), 55 * 55 * 11 * 11 * 3 * 96)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from ml.errors import SophiaUsageError
from ml.data.jsonl_stream import (
    JsonlFieldStats,
    JsonlReadStats,
    is_json_dataset_file,
    iter_jsonl_field,
    iter_jsonl_objects,
)


class TestJsonlStream(unittest.TestCase):
    def test_is_json_dataset_file_excludes_artifacts(self) -> None:
        self.assertTrue(is_json_dataset_file("data.jsonl"))
        self.assertTrue(is_json_dataset_file("data.json"))
        self.assertFalse(is_json_dataset_file("manifest.json"))
        self.assertFalse(is_json_dataset_file("data_512.jsonl.clean_report.json"))
        self.assertFalse(is_json_dataset_file("mix_fin.clean_summary.json"))
        self.assertFalse(is_json_dataset_file("data_512.jsonl.dingo_patch_report.json"))
        self.assertFalse(is_json_dataset_file("pretrain_fin.shard_stats.jsonl"))

    def test_iter_jsonl_objects_counts_and_yields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "data.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"a": 1}\n')
                f.write("[1, 2]\n")
                f.write('{"a":\n')
                f.write("\n")

            stats = JsonlReadStats()
            out = list(
                iter_jsonl_objects(
                    path, stats=stats, strict_json=False, max_json_errors=0
                )
            )
            self.assertEqual(out, [{"a": 1}])
            self.assertEqual(stats.lines, 3)
            self.assertEqual(stats.decoded, 2)
            self.assertEqual(stats.decode_errors, 1)
            self.assertEqual(stats.non_object, 1)

    def test_iter_jsonl_objects_strict_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{bad json\n")

            stats = JsonlReadStats()
            with self.assertRaises(ValueError):
                list(iter_jsonl_objects(path, stats=stats, strict_json=True))
            self.assertEqual(stats.lines, 1)
            self.assertEqual(stats.decode_errors, 1)

    def test_iter_jsonl_objects_max_json_errors_exits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{bad json\n")
                f.write("{still bad\n")

            stats = JsonlReadStats()
            with self.assertRaises(SophiaUsageError):
                list(
                    iter_jsonl_objects(
                        path, stats=stats, strict_json=False, max_json_errors=1
                    )
                )
            self.assertEqual(stats.lines, 1)
            self.assertEqual(stats.decode_errors, 1)

    def test_iter_jsonl_objects_rejects_json_array_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "array.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('[{"a": 1}]\n')
            with self.assertRaises(ValueError):
                list(iter_jsonl_objects(path))

    def test_max_json_errors_without_stats_exits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{bad json\n")
            with self.assertRaises(SophiaUsageError):
                list(iter_jsonl_objects(path, strict_json=False, max_json_errors=1))

    def test_iter_jsonl_field_yields_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "data.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"text": "a"}\n')
                f.write('{"text": 1}\n')
                f.write('{"text": null}\n')
                f.write('{"foo": "bar"}\n')
                f.write('["x"]\n')
                f.write("{bad json\n")

            stats = JsonlFieldStats()
            out = list(
                iter_jsonl_field(
                    path, "text", stats=stats, strict_json=False, max_json_errors=0
                )
            )
            self.assertEqual(out, ["a", "1"])
            self.assertEqual(stats.lines, 6)
            self.assertEqual(stats.decoded, 5)
            self.assertEqual(stats.decode_errors, 1)
            self.assertEqual(stats.non_object, 1)
            self.assertEqual(stats.missing_field, 2)
            self.assertEqual(stats.yielded, 2)
            self.assertEqual(stats.empty_after_strip, 0)

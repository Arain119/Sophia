from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from ml.tooling.pipelines.pretrain_mix_token_shards import _read_parquet_rows


def test_read_parquet_rows_survives_hive_style_dir_with_conflicting_column(
    tmp_path,
) -> None:
    # Files under ``source=<value>/`` directories must read as plain single files.
    # Dataset-level hive-partition inference would turn the path segment into a
    # ``source`` partition field that collides with the file's own column.
    hive_dir = tmp_path / "code" / "source=Python"
    hive_dir.mkdir(parents=True)
    table = pa.table(
        {
            "text": pa.array(["print('hi')", "x = 1"], type=pa.large_string()),
            "source": pa.array(["Python", "Python"], type=pa.large_string()),
            "score": pa.array([6.0, 5.0], type=pa.float32()),
        }
    )
    fp = hive_dir / "python_part_001.parquet"
    pq.write_table(table, fp)

    batch = _read_parquet_rows(fp=str(fp), cols=["text", "source", "score"])

    assert batch is not None
    assert batch.height == 2
    assert batch.get_column("source").to_list() == ["Python", "Python"]

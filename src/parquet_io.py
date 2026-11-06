"""Bounded parquet consolidation; no feature construction or imputation."""
import os

import pyarrow as pa
import pyarrow.parquet as pq


def _row_group_blocks(metadata, limit):
    """Row groups gathered into blocks of at most ``limit`` rows.

    Decoding a whole row group is much cheaper than slicing it into batches,
    and a block keeps memory bounded on a store too large to read at once.
    """
    block, rows = [], 0
    for index in range(metadata.num_row_groups):
        size = metadata.row_group(index).num_rows
        if block and rows + size > limit:
            yield block
            block, rows = [], 0
        block.append(index)
        rows += size
    if block:
        yield block


class RollingParquetWriter:
    """Streams frames into one parquet file, opening another on a schema change.

    Every row is encoded once. Only a frame whose types differ from the one
    before it leaves a further file, for ``combine_parquet_parts`` to reconcile.
    """

    def __init__(self, directory, *, batch_rows=16384):
        self._directory = directory
        self._batch_rows = batch_rows
        self._writer = None
        self.paths = []
        self.rows = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def write(self, frame):
        table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata()
        if self._writer is None or not self._writer.schema.equals(table.schema):
            self.close()
            self.paths.append(os.path.join(self._directory, f"part_{len(self.paths):04d}.parquet"))
            self._writer = pq.ParquetWriter(self.paths[-1], table.schema)
        self._writer.write_table(table, row_group_size=self._batch_rows)
        self.rows += table.num_rows

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def combine_parquet_parts(paths, destination, *, batch_rows=16384, block_rows=65536):
    if batch_rows < 1 or not paths:
        raise ValueError("Parquet parts and a positive batch size are required")
    schema = pa.unify_schemas([pq.read_schema(p) for p in paths],
                             promote_options="permissive").remove_metadata()
    rows = 0
    with pq.ParquetWriter(destination, schema) as writer:
        for path in paths:
            with pq.ParquetFile(path) as source:
                for block in _row_group_blocks(source.metadata, max(block_rows, batch_rows)):
                    table = source.read_row_groups(block)
                    arrays = [table.column(field.name).cast(field.type)
                              if field.name in table.column_names
                              else pa.nulls(table.num_rows, field.type) for field in schema]
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema),
                                       row_group_size=batch_rows)
                    rows += table.num_rows
    return rows

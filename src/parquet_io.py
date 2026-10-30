"""Bounded parquet consolidation; no feature construction or imputation."""
import pyarrow as pa
import pyarrow.parquet as pq


def combine_parquet_parts(paths, destination, *, batch_rows=16384):
    if batch_rows < 1 or not paths:
        raise ValueError("Parquet parts and a positive batch size are required")
    schema = pa.unify_schemas([pq.read_schema(p) for p in paths],
                             promote_options="permissive").remove_metadata()
    rows = 0
    with pq.ParquetWriter(destination, schema) as writer:
        for path in paths:
            with pq.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=batch_rows):
                    table = pa.Table.from_batches([batch])
                    arrays = [table.column(field.name).cast(field.type)
                              if field.name in table.column_names
                              else pa.nulls(table.num_rows, field.type) for field in schema]
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    rows += table.num_rows
    return rows

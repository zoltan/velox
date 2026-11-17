#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Export an existing DuckDB TPC-H database and SQL reference results.

Requires the Python duckdb package and its tpch extension already installed.
The destination must be new, so results cannot silently mix different datasets.
"""

import argparse
import json
from pathlib import Path

import duckdb

QUERIES = (2, 11, 15, 17, 20, 22)
TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"threads": args.threads, "memory_limit": "12GB"}
    with duckdb.connect(str(args.database), read_only=True, config=config) as source:
        source.execute("LOAD tpch")
        queries = dict(
            source.execute("SELECT query_nr, query FROM tpch_queries()").fetchall()
        )
        manifest = {
            "duckdb_version": duckdb.__version__,
            "database": str(args.database.resolve()),
            "numeric_storage": "DECIMAL and ps_availqty converted to DOUBLE",
            "tables": {},
        }
        with duckdb.connect(config=config) as reference:
            for table in TABLES:
                directory = output / table
                directory.mkdir()
                parquet = str(directory / "data.parquet").replace("'", "''")
                schema = source.execute(f"DESCRIBE {table}").fetchall()
                # Match the existing benchmark's floating-point arithmetic.
                columns = [
                    f'CAST("{name}" AS DOUBLE) AS "{name}"'
                    if typ.startswith("DECIMAL") or name == "ps_availqty"
                    else f'"{name}"'
                    for name, typ, *_ in schema
                ]
                source.execute(
                    f"COPY (SELECT {','.join(columns)} FROM {table}) TO '{parquet}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
                )
                reference.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet}')"
                )
                manifest["tables"][table] = reference.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0]
            expected = output / "expected"
            expected.mkdir()
            sql_dir = output / "sql"
            sql_dir.mkdir()
            for query in QUERIES:
                sql = queries[query].strip().rstrip(";")
                if query == 22:
                    sql = sql.replace("cntrycode", "country_code")
                (sql_dir / f"q{query}.sql").write_text(sql + ";\n")
                parquet = str(expected / f"q{query}.parquet").replace("'", "''")
                reference.execute(f"COPY ({sql}) TO '{parquet}' (FORMAT PARQUET)")
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

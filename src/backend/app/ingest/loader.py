"""
FAERS raw-file loader.

Reads FAERS ASCII exports into DuckDB relations using the verified schema.
Raw files are NEVER modified — all transformations happen in normalise.py.

Usage
-----
    from backend.app.ingest.loader import FAERSLoader

    loader = FAERSLoader(raw_dir="data/raw", db_path=":memory:")
    loader.load_all()
    con = loader.con          # duckdb.DuckDBPyConnection with raw_* views
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .constants import DELIMITER, RAW_FILES

logger = logging.getLogger(__name__)

# DuckDB table names for raw data (prefix = "raw_")
_RAW_TABLE = {key: f"raw_{key.lower()}" for key in RAW_FILES}


class FAERSLoader:
    """
    Loads the seven FAERS ASCII files into a DuckDB connection as raw views.

    Parameters
    ----------
    raw_dir : str | Path
        Directory containing the FAERS .txt files.
    db_path : str
        DuckDB database path.  Use ":memory:" for in-process tests.
    """

    def __init__(self, raw_dir: str | Path, db_path: str = ":memory:") -> None:
        self.raw_dir = Path(raw_dir)
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._loaded: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all seven FAERS files."""
        for key in RAW_FILES:
            self.load_table(key)

    def load_table(self, key: str) -> None:
        """
        Load a single FAERS file into DuckDB as a view named raw_<key lower>.

        All columns are read as VARCHAR so no data is silently coerced.
        The raw file is memory-mapped via DuckDB's read_csv — no copy is made.
        """
        key = key.upper()
        if key not in RAW_FILES:
            raise ValueError(f"Unknown FAERS table key: {key!r}. Valid: {set(RAW_FILES)}")

        file_path = self.raw_dir / RAW_FILES[key]
        if not file_path.exists():
            raise FileNotFoundError(f"FAERS file not found: {file_path}")

        table_name = _RAW_TABLE[key]

        # Read every column as VARCHAR — no silent type coercion on ingest.
        # DuckDB's read_csv all_varchar=true is the safest way to preserve raw
        # values (e.g. partial dates, leading zeros).
        # Note: FAERS files do not use quoting; omitting quote= lets DuckDB
        # use its default (double-quote), which is harmless for these files.
        sql = f"""
            CREATE OR REPLACE VIEW {table_name} AS
            SELECT *
            FROM read_csv(
                '{file_path.as_posix()}',
                delim='{DELIMITER}',
                header=true,
                all_varchar=true
            )
        """
        self.con.execute(sql)
        self._loaded.add(key)
        logger.debug("Loaded raw view: %s  (%s)", table_name, file_path.name)

    def row_count(self, key: str) -> int:
        """Return the number of rows in a raw view."""
        table_name = _RAW_TABLE[key.upper()]
        return self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # type: ignore[index]

    def raw_counts(self) -> dict[str, int]:
        """Return {table_key: row_count} for all loaded tables."""
        return {key: self.row_count(key) for key in self._loaded}

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FAERSLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

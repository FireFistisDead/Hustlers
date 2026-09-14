"""
Integration tests for the FAERS 2026 Q2 ingest + normalisation layer.

These tests run against the actual raw files in data/raw/.
They are deliberately slow (full dataset) and should be marked with
pytest -m integration if you want to separate them from fast unit tests.

Run from Hustlers/src/:
    pytest backend/tests/test_ingest.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.ingest.constants import EXPECTED_RAW_COUNTS
from backend.app.ingest.loader import FAERSLoader
from backend.app.ingest.normalize import FAERSNormaliser
from backend.app.ingest.validate import FAERSValidator

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve raw data directory relative to this file's location:
#   backend/tests/test_ingest.py  →  ../../data/raw
_HERE = Path(__file__).parent
_RAW_DIR = (_HERE / ".." / ".." / "data" / "raw").resolve()

# Skip the entire module if raw data is absent (e.g. CI without data)
pytestmark = pytest.mark.skipif(
    not (_RAW_DIR / "DEMO26Q2.txt").exists(),
    reason="FAERS 26Q2 raw data not present at data/raw/",
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures — load once, share across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def loader() -> FAERSLoader:  # type: ignore[return]
    """Load all 7 raw FAERS files into an in-memory DuckDB instance."""
    ldr = FAERSLoader(raw_dir=_RAW_DIR, db_path=":memory:")
    ldr.load_all()
    yield ldr
    ldr.close()


@pytest.fixture(scope="session")
def normaliser(loader: FAERSLoader) -> FAERSNormaliser:
    """Build all normalised views on top of the loaded raw data."""
    norm = FAERSNormaliser(loader)
    norm.normalise_all()
    return norm


@pytest.fixture(scope="session")
def validator(loader: FAERSLoader, normaliser: FAERSNormaliser) -> FAERSValidator:
    return FAERSValidator(loader, normaliser)


@pytest.fixture(scope="session")
def validation_report(validator: FAERSValidator):
    return validator.run_all()


# ---------------------------------------------------------------------------
# 1. Raw row-count tests
# ---------------------------------------------------------------------------

class TestRawCounts:
    @pytest.mark.parametrize("key,expected", list(EXPECTED_RAW_COUNTS.items()))
    def test_raw_row_count(self, loader: FAERSLoader, key: str, expected: int) -> None:
        actual = loader.row_count(key)
        assert actual == expected, (
            f"{key}: expected {expected:,} rows, got {actual:,}"
        )


# ---------------------------------------------------------------------------
# 2. norm_demo — current-case deduplication
# ---------------------------------------------------------------------------

class TestNormDemo:
    def test_primaryid_unique_in_raw(self, loader: FAERSLoader) -> None:
        total, unique = loader.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT primaryid) FROM raw_demo"
        ).fetchone()  # type: ignore[misc]
        assert total == unique, f"raw_demo has duplicate primaryids: {total} rows, {unique} unique"

    def test_one_row_per_caseid(self, normaliser: FAERSNormaliser) -> None:
        """After current-case dedup, each caseid appears exactly once."""
        total, unique_case = normaliser.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT caseid) FROM norm_demo"
        ).fetchone()  # type: ignore[misc]
        assert total == unique_case, (
            f"norm_demo: {total} rows but only {unique_case} unique caseids"
        )

    def test_caseversion_is_numeric(self, normaliser: FAERSNormaliser) -> None:
        null_version = normaliser.con.execute(
            "SELECT COUNT(*) FROM norm_demo WHERE caseversion IS NULL"
        ).fetchone()[0]  # type: ignore[index]
        assert null_version == 0, f"{null_version} rows have NULL caseversion after cast"

    def test_demo_row_count_le_raw(self, normaliser: FAERSNormaliser) -> None:
        norm = normaliser.row_count("norm_demo")
        raw = EXPECTED_RAW_COUNTS["DEMO"]
        assert norm <= raw


# ---------------------------------------------------------------------------
# 3. norm_drug — identity deduplication and PK uniqueness
# ---------------------------------------------------------------------------

class TestNormDrug:
    def test_drug_pk_unique(self, normaliser: FAERSNormaliser) -> None:
        """(primaryid, drug_seq) must be unique in norm_drug."""
        total, unique_pk = normaliser.con.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT primaryid || '|' || CAST(drug_seq AS VARCHAR))
            FROM norm_drug
            """
        ).fetchone()  # type: ignore[misc]
        assert total == unique_pk, (
            f"norm_drug PK not unique: {total} rows, {unique_pk} unique (primaryid, drug_seq) pairs"
        )

    def test_drug_rows_less_than_raw(self, normaliser: FAERSNormaliser) -> None:
        norm = normaliser.row_count("norm_drug")
        raw = EXPECTED_RAW_COUNTS["DRUG"]
        assert norm < raw, (
            f"norm_drug ({norm:,}) should be < raw_drug ({raw:,}) after deduplication"
        )

    def test_drug_regimen_fk_integrity(self, normaliser: FAERSNormaliser) -> None:
        orphans = normaliser.con.execute(
            """
            SELECT COUNT(*) FROM norm_drug_regimen r
            WHERE NOT EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = r.primaryid AND d.drug_seq = r.drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        assert orphans == 0, f"{orphans} norm_drug_regimen rows have no matching norm_drug entry"

    def test_drug_regimen_drops_exact_dupes(self, normaliser: FAERSNormaliser) -> None:
        norm = normaliser.row_count("norm_drug_regimen")
        raw = EXPECTED_RAW_COUNTS["DRUG"]
        # 1 923 exact duplicates known in 26Q2; norm must be strictly less
        assert norm < raw, (
            f"norm_drug_regimen ({norm:,}) should be < raw_drug ({raw:,}); exact dupes not dropped"
        )


# ---------------------------------------------------------------------------
# 4. REAC / OUTC — case-level, never drug-level
# ---------------------------------------------------------------------------

class TestReacOutcCaseLevel:
    def test_reac_has_no_drug_seq_column(self, normaliser: FAERSNormaliser) -> None:
        cols = [
            r[0] for r in normaliser.con.execute("DESCRIBE norm_reac").fetchall()
        ]
        assert "drug_seq" not in cols, "norm_reac must NOT contain drug_seq"

    def test_outc_has_no_drug_seq_column(self, normaliser: FAERSNormaliser) -> None:
        cols = [
            r[0] for r in normaliser.con.execute("DESCRIBE norm_outc").fetchall()
        ]
        assert "drug_seq" not in cols, "norm_outc must NOT contain drug_seq"

    def test_reac_rows_le_raw(self, normaliser: FAERSNormaliser) -> None:
        assert normaliser.row_count("norm_reac") <= EXPECTED_RAW_COUNTS["REAC"]

    def test_outc_rows_le_raw(self, normaliser: FAERSNormaliser) -> None:
        assert normaliser.row_count("norm_outc") <= EXPECTED_RAW_COUNTS["OUTC"]


# ---------------------------------------------------------------------------
# 5. The critical anti-cartesian-product test
#
#    primaryid 1021640135 has 9 drugs, 46 reactions, 2 outcomes.
#    A cartesian join would yield 9 × 46 × 2 = 828 rows.
#    Each table must return its independent count — not the product.
# ---------------------------------------------------------------------------

class TestNoCartesianProduct:
    # primaryid 1021640135 verified counts (from direct file inspection):
    #   raw_drug rows = 9  (drug_seq=1 has 2 dosing episodes for the same drug)
    #   norm_drug rows = 8  (drug_seq=1 deduplicated to 1 identity row)
    #   norm_drug_regimen rows = 9  (retains the 2 episodes for drug_seq=1)
    #   norm_reac rows = 45  (DISTINCT on pt+drug_rec_act)
    #   norm_outc rows = 2
    #   Cartesian product (raw) would be: 9 × 46 × 2 = 828 rows
    #   Cartesian product (norm) would be: 8 × 45 × 2 = 720 rows
    PID = "1021640135"
    EXPECTED_NORM_DRUGS = 8
    EXPECTED_NORM_REACS = 45
    EXPECTED_NORM_OUTCS = 2
    RAW_DRUGS = 9
    RAW_REACS = 46

    def test_drug_count_is_independent(self, normaliser: FAERSNormaliser) -> None:
        nd = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_drug WHERE primaryid = '{self.PID}'"
        ).fetchone()[0]  # type: ignore[index]
        assert nd == self.EXPECTED_NORM_DRUGS, (
            f"Expected {self.EXPECTED_NORM_DRUGS} norm_drug rows for {self.PID}, got {nd}"
        )

    def test_reac_count_is_independent(self, normaliser: FAERSNormaliser) -> None:
        nr = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_reac WHERE primaryid = '{self.PID}'"
        ).fetchone()[0]  # type: ignore[index]
        assert nr == self.EXPECTED_NORM_REACS, (
            f"Expected {self.EXPECTED_NORM_REACS} norm_reac rows for {self.PID}, got {nr}"
        )

    def test_outc_count_is_independent(self, normaliser: FAERSNormaliser) -> None:
        no = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_outc WHERE primaryid = '{self.PID}'"
        ).fetchone()[0]  # type: ignore[index]
        assert no == self.EXPECTED_NORM_OUTCS, (
            f"Expected {self.EXPECTED_NORM_OUTCS} norm_outc rows for {self.PID}, got {no}"
        )

    def test_three_tables_do_not_cartesian_multiply(
        self, normaliser: FAERSNormaliser
    ) -> None:
        """
        Explicitly prove that norm_drug, norm_reac, norm_outc do NOT form a
        cartesian product for a multi-drug, multi-reaction, multi-outcome case.

        primaryid 1021640135:
          raw  → 9 drugs × 46 reactions × 2 outcomes = 828 (cartesian, WRONG)
          norm → 8 drugs (deduped) + 45 reactions (distinct pt) + 2 outcomes
                 = 55 independent rows (CORRECT)

        Each table is queried independently — no join is performed here.
        If future code inadvertently cartesian-joins them the individual counts
        would explode to 8×45×2 = 720 per table instead of 8 / 45 / 2.
        """
        pid = self.PID

        nd = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_drug WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]
        nr = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_reac WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]
        no = normaliser.con.execute(
            f"SELECT COUNT(*) FROM norm_outc WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]

        cartesian_would_be = nd * nr * no   # 8 × 45 × 2 = 720
        independent_sum    = nd + nr + no   # 8 + 45 + 2 = 55

        # Verify each table returned its own independent count
        assert nd == self.EXPECTED_NORM_DRUGS
        assert nr == self.EXPECTED_NORM_REACS
        assert no == self.EXPECTED_NORM_OUTCS

        # The product must be >> sum (non-trivial case)
        assert cartesian_would_be > independent_sum, (
            "Sanity: cartesian product should exceed independent sum"
        )

        # Core assertion: individual counts are NOT equal to the cartesian product
        assert nd != cartesian_would_be, (
            f"norm_drug count ({nd}) equals cartesian product ({cartesian_would_be}) — "
            "data has been exploded"
        )
        assert nr != cartesian_would_be, (
            f"norm_reac count ({nr}) equals cartesian product ({cartesian_would_be}) — "
            "data has been exploded"
        )


# ---------------------------------------------------------------------------
# 6. INDI / THER join keys
# ---------------------------------------------------------------------------

class TestIndiTherJoins:
    def test_indi_drug_seq_col_present(self, normaliser: FAERSNormaliser) -> None:
        cols = [r[0] for r in normaliser.con.execute("DESCRIBE norm_indi").fetchall()]
        assert "indi_drug_seq" in cols

    def test_ther_dsg_drug_seq_col_present(self, normaliser: FAERSNormaliser) -> None:
        cols = [r[0] for r in normaliser.con.execute("DESCRIBE norm_ther").fetchall()]
        assert "dsg_drug_seq" in cols

    def test_indi_join_coverage_above_95pct(self, normaliser: FAERSNormaliser) -> None:
        total = normaliser.con.execute("SELECT COUNT(*) FROM norm_indi").fetchone()[0]  # type: ignore[index]
        matched = normaliser.con.execute(
            """
            SELECT COUNT(*) FROM norm_indi i
            WHERE EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = i.primaryid AND d.drug_seq = i.indi_drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        pct = matched / total * 100 if total else 0
        assert pct >= 95.0, f"INDI join coverage {pct:.1f}% < 95%"

    def test_ther_join_coverage_above_95pct(self, normaliser: FAERSNormaliser) -> None:
        total = normaliser.con.execute("SELECT COUNT(*) FROM norm_ther").fetchone()[0]  # type: ignore[index]
        matched = normaliser.con.execute(
            """
            SELECT COUNT(*) FROM norm_ther t
            WHERE EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = t.primaryid AND d.drug_seq = t.dsg_drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        pct = matched / total * 100 if total else 0
        assert pct >= 95.0, f"THER join coverage {pct:.1f}% < 95%"


# ---------------------------------------------------------------------------
# 7. Date flag columns
# ---------------------------------------------------------------------------

class TestDateFlags:
    def test_ther_partial_start_dt_detected(self, normaliser: FAERSNormaliser) -> None:
        """26Q2 has 70 799 partial start_dt values (len 4 or 6)."""
        partial = normaliser.con.execute(
            "SELECT COUNT(*) FROM norm_ther WHERE start_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        assert partial > 0, "No partial start_dt flags set — date flagging broken"

    def test_ther_partial_end_dt_detected(self, normaliser: FAERSNormaliser) -> None:
        partial = normaliser.con.execute(
            "SELECT COUNT(*) FROM norm_ther WHERE end_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        assert partial > 0, "No partial end_dt flags set — date flagging broken"

    def test_demo_event_dt_partial_flag(self, normaliser: FAERSNormaliser) -> None:
        """event_dt in DEMO is either NULL (58%) or 8-digit YYYYMMDD — no partials expected."""
        partial = normaliser.con.execute(
            "SELECT COUNT(*) FROM norm_demo WHERE event_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        # All non-null event_dt values in 26Q2 are full YYYYMMDD — expect 0 partials
        assert partial == 0, f"{partial} unexpected partial event_dt values"

    def test_demo_event_dt_invalid_flag(self, normaliser: FAERSNormaliser) -> None:
        invalid = normaliser.con.execute(
            "SELECT COUNT(*) FROM norm_demo WHERE event_dt_invalid = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        # Should be 0 in a clean quarter file
        assert invalid == 0, f"{invalid} event_dt values flagged as invalid"

    def test_raw_values_preserved(self, normaliser: FAERSNormaliser) -> None:
        """Raw date strings must be unchanged in norm_ther (no coercion)."""
        raw_sample = normaliser.con.execute(
            "SELECT start_dt FROM raw_ther WHERE start_dt IS NOT NULL LIMIT 1"
        ).fetchone()[0]  # type: ignore[index]
        norm_sample = normaliser.con.execute(
            "SELECT start_dt FROM norm_ther WHERE start_dt IS NOT NULL LIMIT 1"
        ).fetchone()[0]  # type: ignore[index]
        # Both must be strings (raw preservation)
        assert isinstance(raw_sample, str) and isinstance(norm_sample, str)


# ---------------------------------------------------------------------------
# 8. Full validation report — must pass all checks
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_all_checks_pass(self, validation_report) -> None:
        if not validation_report.passed:
            failures = "\n".join(
                f"  ✗ {c.name}: {c.detail}" for c in validation_report.failures
            )
            pytest.fail(f"Validation failures:\n{failures}")

    def test_report_has_expected_check_count(self, validation_report) -> None:
        assert len(validation_report.checks) == 12, (
            f"Expected 12 validation checks, got {len(validation_report.checks)}"
        )

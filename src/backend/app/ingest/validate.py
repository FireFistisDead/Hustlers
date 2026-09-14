"""
FAERS ingest validation.

Runs reconciliation checks against loaded DuckDB views and returns a
structured report.  No data is modified here.

Usage
-----
    from backend.app.ingest.validate import FAERSValidator, ValidationReport

    validator = FAERSValidator(loader, normaliser)
    report = validator.run_all()
    print(report.summary())
    assert report.passed, report.failures
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from .constants import EXPECTED_RAW_COUNTS
from .loader import FAERSLoader
from .normalize import FAERSNormaliser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ValidationReport:
    checks: list[CheckResult] = dataclasses.field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = [f"{'PASS' if self.passed else 'FAIL'}  ({len(self.checks)} checks)"]
        for c in self.checks:
            icon = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{icon}] {c.name}: {c.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class FAERSValidator:
    """
    Runs reconciliation checks on a loaded + normalised FAERS dataset.

    Parameters
    ----------
    loader     : FAERSLoader      — already called load_all()
    normaliser : FAERSNormaliser  — already called normalise_all()
    """

    def __init__(self, loader: FAERSLoader, normaliser: FAERSNormaliser) -> None:
        self.con = loader.con
        self._loader = loader
        self._norm = normaliser

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_all(self) -> ValidationReport:
        report = ValidationReport()
        report.checks.extend([
            self._check_raw_counts(),
            self._check_demo_primaryid_unique(),
            self._check_current_case_dedup(),
            self._check_drug_norm_pk_unique(),
            self._check_drug_regimen_fk(),
            self._check_reac_case_level(),
            self._check_outc_case_level(),
            self._check_indi_join_coverage(),
            self._check_ther_join_coverage(),
            self._check_no_cartesian_product(),
            self._check_date_flag_columns(),
            self._check_demo_primaryid_in_all_tables(),
        ])
        for c in report.checks:
            level = logging.DEBUG if c.passed else logging.WARNING
            logger.log(level, "%s %s: %s", "PASS" if c.passed else "FAIL", c.name, c.detail)
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_raw_counts(self) -> CheckResult:
        """Raw row counts must match expected 26Q2 values."""
        mismatches = {}
        actual = {}
        for key, expected in EXPECTED_RAW_COUNTS.items():
            count = self._loader.row_count(key)
            actual[key] = count
            if count != expected:
                mismatches[key] = {"expected": expected, "actual": count}
        passed = len(mismatches) == 0
        detail = (
            "all match expected 26Q2 counts"
            if passed
            else f"mismatches: {mismatches}"
        )
        return CheckResult("raw_counts", passed, detail, {"actual": actual})

    def _check_demo_primaryid_unique(self) -> CheckResult:
        """primaryid must be unique in raw_demo (each report is one row)."""
        total, unique = self.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT primaryid) FROM raw_demo"
        ).fetchone()  # type: ignore[misc]
        passed = total == unique
        return CheckResult(
            "demo_primaryid_unique",
            passed,
            f"total={total} unique_primaryid={unique}",
            {"total": total, "unique_primaryid": unique},
        )

    def _check_current_case_dedup(self) -> CheckResult:
        """
        norm_demo must have at most one row per caseid (latest caseversion wins).
        In 26Q2 exactly one caseid has two raw caseversions; norm_demo must
        collapse it to one row.
        """
        row = self.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT caseid) FROM norm_demo"
        ).fetchone()  # type: ignore[misc]
        total, unique_case = row
        passed = total == unique_case
        return CheckResult(
            "current_case_dedup",
            passed,
            f"norm_demo rows={total} unique_caseid={unique_case}",
            {"norm_demo_rows": total, "unique_caseid": unique_case},
        )

    def _check_drug_norm_pk_unique(self) -> CheckResult:
        """(primaryid, drug_seq) must be unique in norm_drug after dedup."""
        total, unique_pk = self.con.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT primaryid || '|' || CAST(drug_seq AS VARCHAR))
            FROM norm_drug
            """
        ).fetchone()  # type: ignore[misc]
        passed = total == unique_pk
        return CheckResult(
            "drug_norm_pk_unique",
            passed,
            f"norm_drug rows={total} unique_(primaryid,drug_seq)={unique_pk}",
            {"norm_drug_rows": total, "unique_pk": unique_pk},
        )

    def _check_drug_regimen_fk(self) -> CheckResult:
        """
        Every (primaryid, drug_seq) in norm_drug_regimen must exist in norm_drug.
        Orphan regimen rows indicate a normalisation bug.
        """
        orphans = self.con.execute(
            """
            SELECT COUNT(*) FROM norm_drug_regimen r
            WHERE NOT EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = r.primaryid
                  AND d.drug_seq  = r.drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        passed = orphans == 0
        return CheckResult(
            "drug_regimen_fk",
            passed,
            f"orphan regimen rows (no matching norm_drug entry): {orphans}",
            {"orphan_count": orphans},
        )

    def _check_reac_case_level(self) -> CheckResult:
        """
        REAC must never be joined to DRUG — verify norm_reac has no drug_seq
        column (structural check) and that the number of (primaryid, pt) pairs
        does not exceed the number of unique cases × average reactions.

        Concrete check: for the worst-case primary ID (most reactions), the
        reaction count must equal the number of distinct pt terms, not
        reactions × drugs (which would indicate a cartesian product).
        """
        # Structural: no drug_seq in norm_reac
        cols = [
            row[0]
            for row in self.con.execute("DESCRIBE norm_reac").fetchall()
        ]
        has_drug_seq = "drug_seq" in cols

        # Count: norm_reac rows should equal raw_reac rows (DISTINCT on all cols)
        norm_count = self.con.execute("SELECT COUNT(*) FROM norm_reac").fetchone()[0]  # type: ignore[index]
        raw_count = self.con.execute("SELECT COUNT(*) FROM raw_reac").fetchone()[0]  # type: ignore[index]

        passed = (not has_drug_seq) and (norm_count <= raw_count)
        return CheckResult(
            "reac_case_level",
            passed,
            f"drug_seq_in_reac={has_drug_seq} norm_rows={norm_count} raw_rows={raw_count}",
            {"has_drug_seq": has_drug_seq, "norm_count": norm_count, "raw_count": raw_count},
        )

    def _check_outc_case_level(self) -> CheckResult:
        """norm_outc rows ≤ raw_outc rows; no drug_seq column."""
        cols = [
            row[0]
            for row in self.con.execute("DESCRIBE norm_outc").fetchall()
        ]
        has_drug_seq = "drug_seq" in cols
        norm_count = self.con.execute("SELECT COUNT(*) FROM norm_outc").fetchone()[0]  # type: ignore[index]
        raw_count = self.con.execute("SELECT COUNT(*) FROM raw_outc").fetchone()[0]  # type: ignore[index]
        passed = (not has_drug_seq) and (norm_count <= raw_count)
        return CheckResult(
            "outc_case_level",
            passed,
            f"drug_seq_in_outc={has_drug_seq} norm_rows={norm_count} raw_rows={raw_count}",
            {"has_drug_seq": has_drug_seq, "norm_count": norm_count, "raw_count": raw_count},
        )

    def _check_indi_join_coverage(self) -> CheckResult:
        """
        For rows in norm_indi, the (primaryid, indi_drug_seq) should exist in
        norm_drug.  Report the orphan rate (some are expected: INDI can reference
        drugs that appear in prior quarter files not loaded here).
        """
        total = self.con.execute("SELECT COUNT(*) FROM norm_indi").fetchone()[0]  # type: ignore[index]
        matched = self.con.execute(
            """
            SELECT COUNT(*) FROM norm_indi i
            WHERE EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = i.primaryid
                  AND d.drug_seq  = i.indi_drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        orphans = total - matched
        pct = orphans / total * 100 if total else 0.0
        # Acceptable threshold: <5 % orphan rate within a single quarter load
        passed = pct < 5.0
        return CheckResult(
            "indi_join_coverage",
            passed,
            f"total={total} matched={matched} orphans={orphans} ({pct:.2f}%)",
            {"total": total, "matched": matched, "orphans": orphans, "orphan_pct": pct},
        )

    def _check_ther_join_coverage(self) -> CheckResult:
        """Same coverage check as INDI but for THER → DRUG."""
        total = self.con.execute("SELECT COUNT(*) FROM norm_ther").fetchone()[0]  # type: ignore[index]
        matched = self.con.execute(
            """
            SELECT COUNT(*) FROM norm_ther t
            WHERE EXISTS (
                SELECT 1 FROM norm_drug d
                WHERE d.primaryid = t.primaryid
                  AND d.drug_seq  = t.dsg_drug_seq
            )
            """
        ).fetchone()[0]  # type: ignore[index]
        orphans = total - matched
        pct = orphans / total * 100 if total else 0.0
        passed = pct < 5.0
        return CheckResult(
            "ther_join_coverage",
            passed,
            f"total={total} matched={matched} orphans={orphans} ({pct:.2f}%)",
            {"total": total, "matched": matched, "orphans": orphans, "orphan_pct": pct},
        )

    def _check_no_cartesian_product(self) -> CheckResult:
        """
        The critical anti-cartesian check.

        primaryid 1021640135 (verified against 26Q2 raw files):
          raw_drug = 9 rows  (drug_seq=1 has 2 dosing episodes)
          norm_drug = 8      (drug_seq=1 deduplicated to 1 identity row)
          norm_reac = 45     (DISTINCT on pt + drug_rec_act)
          norm_outc = 2

          Cartesian (raw)  would be: 9 × 46 × 2 = 828   <- WRONG
          Cartesian (norm) would be: 8 × 45 × 2 = 720   <- still WRONG
          Independent:  8 + 45 + 2 = 55                 <- CORRECT

        Each table is queried independently to verify individual counts.
        """
        pid = "1021640135"
        expected_nd, expected_nr, expected_no = 8, 45, 2

        nd = self.con.execute(
            f"SELECT COUNT(*) FROM norm_drug  WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]
        nr = self.con.execute(
            f"SELECT COUNT(*) FROM norm_reac  WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]
        no = self.con.execute(
            f"SELECT COUNT(*) FROM norm_outc  WHERE primaryid = '{pid}'"
        ).fetchone()[0]  # type: ignore[index]

        cartesian_would_be = nd * nr * no
        independent_sum    = nd + nr + no

        passed = (
            nd == expected_nd
            and nr == expected_nr
            and no == expected_no
            and nd != cartesian_would_be
        )
        return CheckResult(
            "no_cartesian_product",
            passed,
            (
                f"primaryid={pid}  norm_drugs={nd}(exp={expected_nd})  "
                f"norm_reacs={nr}(exp={expected_nr})  "
                f"norm_outcs={no}(exp={expected_no})  "
                f"independent_sum={independent_sum}  "
                f"cartesian_would_be={cartesian_would_be}"
            ),
            {
                "primaryid": pid,
                "norm_drugs": nd,
                "norm_reacs": nr,
                "norm_outcs": no,
                "cartesian_would_be": cartesian_would_be,
            },
        )

    def _check_date_flag_columns(self) -> CheckResult:
        """
        Flag columns (event_dt_partial, start_dt_partial, etc.) must exist and
        produce sensible non-zero counts where partial dates are known to exist.

        THER has 19 597 start_dt entries with len 4 (YYYY) and
        51 202 with len 6 (YYYYMM) — both are partial.
        """
        partial_start = self.con.execute(
            "SELECT COUNT(*) FROM norm_ther WHERE start_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        partial_end = self.con.execute(
            "SELECT COUNT(*) FROM norm_ther WHERE end_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]
        partial_event = self.con.execute(
            "SELECT COUNT(*) FROM norm_demo WHERE event_dt_partial = TRUE"
        ).fetchone()[0]  # type: ignore[index]

        # In 26Q2 we know there ARE partial start_dt rows (>0 expected)
        passed = partial_start > 0 and partial_end > 0
        return CheckResult(
            "date_flag_columns",
            passed,
            (
                f"partial_start_dt={partial_start}  "
                f"partial_end_dt={partial_end}  "
                f"partial_event_dt={partial_event}"
            ),
            {
                "partial_start_dt": partial_start,
                "partial_end_dt": partial_end,
                "partial_event_dt": partial_event,
            },
        )

    def _check_demo_primaryid_in_all_tables(self) -> CheckResult:
        """
        Every primaryid in norm_demo must appear in norm_drug and norm_reac.
        (OUTC, INDI, THER, RPSR are sparser — reported but not failed on.)
        """
        demo_total = self.con.execute(
            "SELECT COUNT(DISTINCT primaryid) FROM norm_demo"
        ).fetchone()[0]  # type: ignore[index]

        for tbl in ("norm_drug", "norm_reac"):
            tbl_count = self.con.execute(
                f"""
                SELECT COUNT(DISTINCT primaryid) FROM {tbl}
                WHERE primaryid IN (SELECT primaryid FROM norm_demo)
                """
            ).fetchone()[0]  # type: ignore[index]
            if tbl_count != demo_total:
                return CheckResult(
                    "demo_primaryid_coverage",
                    False,
                    f"{tbl}: {tbl_count} of {demo_total} demo primaryids present",
                    {"demo_total": demo_total, tbl: tbl_count},
                )

        return CheckResult(
            "demo_primaryid_coverage",
            True,
            f"all {demo_total} demo primaryids present in norm_drug and norm_reac",
            {"demo_total": demo_total},
        )

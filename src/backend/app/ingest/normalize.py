"""
FAERS normalisation layer.

Transforms raw DuckDB views (raw_*) into clean, analysis-ready DuckDB views
(norm_*).  Raw data is NEVER modified.

Design principles
-----------------
* All joins use `primaryid` (the report-level surrogate key).
* Current-case deduplication: for each caseid keep only the row with the
  highest numeric caseversion.  This is done in `norm_demo` so that all
  downstream tables can join against it.
* DRUG is split into two views:
    norm_drug         — one row per unique drug identity per case
                        PK: (primaryid, drug_seq)
    norm_drug_regimen — one row per dosing episode per (primaryid, drug_seq)
                        FK: (primaryid, drug_seq) → norm_drug
  Only *exact* duplicate rows are dropped; no data is corrected.
* REAC and OUTC are case-level; they are never joined directly to DRUG.
* INDI links to DRUG via  indi_drug_seq = drug_seq  (on primaryid).
* THER links to DRUG via  dsg_drug_seq  = drug_seq  (on primaryid).
* Invalid/partial dates are flagged by companion boolean columns
  (`<col>_partial`, `<col>_invalid`) instead of being corrected.

Views created
-------------
    norm_demo          current-case patient/report demographics
    norm_drug          drug identity (deduplicated)
    norm_drug_regimen  dosing episodes (deduplicated)
    norm_reac          MedDRA PT reactions  (case-level)
    norm_outc          outcomes             (case-level)
    norm_indi          drug indications     (drug-level, via indi_drug_seq)
    norm_ther          therapy dates/durations (drug-level, via dsg_drug_seq)
    norm_rpsr          report source codes  (case-level)
"""
from __future__ import annotations

import logging

import duckdb

from .loader import FAERSLoader

logger = logging.getLogger(__name__)


class FAERSNormaliser:
    """
    Builds normalised DuckDB views on top of the raw views created by
    :class:`FAERSLoader`.

    Parameters
    ----------
    loader : FAERSLoader
        A loader instance that has already called ``load_all()``.
    """

    def __init__(self, loader: FAERSLoader) -> None:
        self.con: duckdb.DuckDBPyConnection = loader.con
        self._loader = loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalise_all(self) -> None:
        """Build all seven normalised views."""
        self._normalise_demo()
        self._normalise_drug()
        self._normalise_reac()
        self._normalise_outc()
        self._normalise_indi()
        self._normalise_ther()
        self._normalise_rpsr()
        logger.info("All normalised views created.")

    def row_count(self, view: str) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]  # type: ignore[index]

    def norm_counts(self) -> dict[str, int]:
        views = [
            "norm_demo",
            "norm_drug",
            "norm_drug_regimen",
            "norm_reac",
            "norm_outc",
            "norm_indi",
            "norm_ther",
            "norm_rpsr",
        ]
        return {v: self.row_count(v) for v in views}

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _normalise_demo(self) -> None:
        """
        Current-case DEMO.

        Strategy: for each caseid, keep the row with the maximum numeric
        caseversion.  Where a caseid has exactly one primaryid (the normal
        case) this is a no-op.  The one known caseid with two caseversions
        in 26Q2 is handled correctly.

        Date flag columns added:
          event_dt_partial  — BOOL  (len < 8, i.e. YYYYMM or YYYY)
          event_dt_invalid  — BOOL  (non-null but not 4/6/8 digits)
          fda_dt_partial    — BOOL
          rept_dt_partial   — BOOL
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_demo AS
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY caseid
                        ORDER BY TRY_CAST(caseversion AS INTEGER) DESC NULLS LAST,
                                 primaryid DESC
                    ) AS rn
                FROM raw_demo
            )
            SELECT
                primaryid,
                caseid,
                TRY_CAST(caseversion AS INTEGER)          AS caseversion,
                i_f_code,
                -- dates: preserve raw string, add partial/invalid flags
                event_dt,
                CASE
                    WHEN event_dt IS NULL          THEN FALSE
                    WHEN LENGTH(event_dt) = 8
                         AND event_dt ~ '^[0-9]{8}$' THEN FALSE
                    ELSE TRUE
                END                                       AS event_dt_partial,
                CASE
                    WHEN event_dt IS NULL          THEN FALSE
                    WHEN event_dt ~ '^[0-9]{4,8}$' THEN FALSE
                    ELSE TRUE
                END                                       AS event_dt_invalid,
                mfr_dt,
                init_fda_dt,
                fda_dt,
                CASE
                    WHEN fda_dt IS NULL            THEN FALSE
                    WHEN LENGTH(fda_dt) < 8        THEN TRUE
                    ELSE FALSE
                END                                       AS fda_dt_partial,
                rept_dt,
                CASE
                    WHEN rept_dt IS NULL           THEN FALSE
                    WHEN LENGTH(rept_dt) < 8       THEN TRUE
                    ELSE FALSE
                END                                       AS rept_dt_partial,
                rept_cod,
                auth_num,
                mfr_num,
                mfr_sndr,
                lit_ref,
                TRY_CAST(age AS DOUBLE)                   AS age,
                age_cod,
                age_grp,
                sex,
                e_sub,
                TRY_CAST(wt AS DOUBLE)                    AS wt,
                wt_cod,
                to_mfr,
                occp_cod,
                reporter_country,
                occr_country
            FROM ranked
            WHERE rn = 1
        """)
        logger.debug("norm_demo created.")

    def _normalise_drug(self) -> None:
        """
        norm_drug — one row per unique drug identity per (primaryid, drug_seq).

        Background: (primaryid, drug_seq) is NOT unique in the raw DRUG file.
        The same drug can appear in multiple rows when dosing changed (different
        dose_amt, route, lot_num, etc.).  Identity columns (role_cod, drugname,
        prod_ai, val_vbm, cum_dose_chr, cum_dose_unit, dechal, rechal, nda_num)
        DO NOT vary within a (primaryid, drug_seq) group in 26Q2.

        We deduplicate by selecting DISTINCT on identity columns only.

        norm_drug_regimen — one row per dosing episode.
        Exact duplicate rows (1 923 in 26Q2) are dropped via DISTINCT.
        """
        # --- norm_drug (identity layer) ---
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_drug AS
            SELECT DISTINCT
                primaryid,
                caseid,
                TRY_CAST(drug_seq AS INTEGER)   AS drug_seq,
                role_cod,
                drugname,
                prod_ai,
                val_vbm,
                cum_dose_chr,
                cum_dose_unit,
                dechal,
                rechal,
                nda_num
            FROM raw_drug
        """)

        # --- norm_drug_regimen (dosing episodes) ---
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_drug_regimen AS
            SELECT DISTINCT
                primaryid,
                TRY_CAST(drug_seq AS INTEGER)   AS drug_seq,
                route,
                dose_vbm,
                dose_amt,
                dose_unit,
                dose_form,
                dose_freq,
                lot_num,
                exp_dt,
                -- date flag for exp_dt
                CASE
                    WHEN exp_dt IS NULL               THEN FALSE
                    WHEN LENGTH(exp_dt) < 8           THEN TRUE
                    ELSE FALSE
                END                                   AS exp_dt_partial
            FROM raw_drug
        """)
        logger.debug("norm_drug and norm_drug_regimen created.")

    def _normalise_reac(self) -> None:
        """
        norm_reac — case-level MedDRA PT reactions.
        REAC is NEVER drug-level; one row per (primaryid, pt).
        drug_rec_act (0.3 % non-null) is preserved but not promoted to a join key.
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_reac AS
            SELECT DISTINCT
                primaryid,
                caseid,
                pt,
                drug_rec_act
            FROM raw_reac
        """)
        logger.debug("norm_reac created.")

    def _normalise_outc(self) -> None:
        """
        norm_outc — case-level outcomes.
        One row per (primaryid, outc_cod); OUTC is case-level, not drug-level.
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_outc AS
            SELECT DISTINCT
                primaryid,
                caseid,
                outc_cod
            FROM raw_outc
        """)
        logger.debug("norm_outc created.")

    def _normalise_indi(self) -> None:
        """
        norm_indi — drug indications.
        Joins to norm_drug via (primaryid, indi_drug_seq = drug_seq).
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_indi AS
            SELECT DISTINCT
                primaryid,
                caseid,
                TRY_CAST(indi_drug_seq AS INTEGER)  AS indi_drug_seq,
                indi_pt
            FROM raw_indi
        """)
        logger.debug("norm_indi created.")

    def _normalise_ther(self) -> None:
        """
        norm_ther — therapy dates and durations.
        Joins to norm_drug via (primaryid, dsg_drug_seq = drug_seq).

        Date formats observed in 26Q2:
          YYYYMMDD (len 8) — full date
          YYYYMM   (len 6) — year+month partial
          YYYY     (len 4) — year-only partial

        Partial dates are flagged; raw strings are preserved.
        dur is numeric (nullable); dur_cod is the unit.
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_ther AS
            SELECT DISTINCT
                primaryid,
                caseid,
                TRY_CAST(dsg_drug_seq AS INTEGER)   AS dsg_drug_seq,
                start_dt,
                CASE
                    WHEN start_dt IS NULL             THEN FALSE
                    WHEN LENGTH(start_dt) < 8         THEN TRUE
                    ELSE FALSE
                END                                   AS start_dt_partial,
                end_dt,
                CASE
                    WHEN end_dt IS NULL               THEN FALSE
                    WHEN LENGTH(end_dt) < 8           THEN TRUE
                    ELSE FALSE
                END                                   AS end_dt_partial,
                TRY_CAST(dur AS DOUBLE)               AS dur,
                dur_cod
            FROM raw_ther
        """)
        logger.debug("norm_ther created.")

    def _normalise_rpsr(self) -> None:
        """
        norm_rpsr — report source codes.
        One row per (primaryid, rpsr_cod).
        """
        self.con.execute("""
            CREATE OR REPLACE VIEW norm_rpsr AS
            SELECT DISTINCT
                primaryid,
                caseid,
                rpsr_cod
            FROM raw_rpsr
        """)
        logger.debug("norm_rpsr created.")

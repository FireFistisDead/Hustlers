"""
Signal detection engine.

Runs disproportionality analysis over normalised FAERS DuckDB views and
returns structured SignalResult objects.

Design
------
- Signal unit: (ingredient_key, reaction_pt, role_scope)
- Counts are unique primaryid counts — never raw FAERS row counts.
- A signal result is only emitted when a >= 3 (minimum case threshold).
- prod_ai is used as ingredient_key; combination products are NOT exploded.
- The 2×2 table is computed in a single DuckDB pass per batch to avoid
  O(N²) Python loops.
- role_scope controls which role_cod values are included in the drug arm.

DISCLAIMER
----------
Results are POTENTIAL DISPROPORTIONATE REPORTING SIGNALS only.
Not confirmed safety signals. Not evidence of causality.
All results require clinical and regulatory expert review.
"""
from __future__ import annotations

import logging
from typing import Optional

import duckdb

from .models import MetricValue, RoleScope, SignalResult, SignalUnit, TwoByTwo
from . import stats as _stats

logger = logging.getLogger(__name__)

# Minimum case count — no result is returned when a < MIN_A
MIN_A = 3


class SignalEngine:
    """
    Disproportionality analysis engine over normalised FAERS DuckDB views.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection that already has norm_drug and norm_reac views available
        (i.e. FAERSNormaliser.normalise_all() has been called).
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        ingredient_key: str,
        reaction_pt: str,
        role_scope: RoleScope = RoleScope.PS,
    ) -> SignalResult:
        """
        Compute a single (ingredient_key, reaction_pt, role_scope) signal.

        Always returns a SignalResult.  When a < 3 or denominators are zero,
        the metric fields are None and meets_threshold is False.
        """
        unit   = SignalUnit(
            ingredient_key=ingredient_key,
            reaction_pt=reaction_pt,
            role_scope=role_scope,
        )
        table  = self._compute_2x2(ingredient_key, reaction_pt, role_scope)
        return _build_result(unit, table)

    def run_batch(
        self,
        ingredients: list[str],
        pts: list[str],
        role_scope: RoleScope = RoleScope.PS,
        min_a: int = MIN_A,
    ) -> list[SignalResult]:
        """
        Compute signals for all (ingredient, PT) pairs in one DuckDB pass.

        Returns only pairs where a >= min_a.  The list is ordered by `a`
        descending (most frequently co-reported first).

        Parameters
        ----------
        ingredients : list of prod_ai values
        pts         : list of MedDRA PT values
        role_scope  : drug role filter
        min_a       : minimum co-report count (default 3)
        """
        if not ingredients or not pts:
            return []

        rows = self._batch_2x2(ingredients, pts, role_scope, min_a)
        results: list[SignalResult] = []
        for row in rows:
            ingredient_key, reaction_pt, a, b, c, d = (
                row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5])
            )
            unit  = SignalUnit(
                ingredient_key=ingredient_key,
                reaction_pt=reaction_pt,
                role_scope=role_scope,
            )
            table = TwoByTwo(a=a, b=b, c=c, d=d)
            results.append(_build_result(unit, table))
        return results

    def top_signals(
        self,
        *,
        role_scope: RoleScope = RoleScope.PS,
        min_a: int = MIN_A,
        top_n_ingredients: int = 200,
        top_n_pts: int = 500,
    ) -> list[SignalResult]:
        """
        Run signals for the top-N ingredients × top-M PT pairs by case count.

        This is the primary entry point for exploratory analysis.
        Returns results ordered by IC025 descending (highest signal strength
        first), filtered to meets_threshold=True.
        """
        ingredients = self._top_ingredients(top_n_ingredients, role_scope)
        pts         = self._top_pts(top_n_pts)
        logger.info(
            "top_signals: %d ingredients × %d PTs, scope=%s",
            len(ingredients), len(pts), role_scope.value,
        )
        results = self.run_batch(ingredients, pts, role_scope, min_a)
        # Sort by IC025 descending; put None last
        results.sort(
            key=lambda r: r.ic.lower if (r.ic and r.ic.lower is not None) else float("-inf"),
            reverse=True,
        )
        return [r for r in results if r.meets_threshold]

    # ------------------------------------------------------------------
    # Internal SQL helpers
    # ------------------------------------------------------------------

    def _universe_n(self) -> int:
        """Total unique primaryid count in norm_demo (current-case universe)."""
        return self.con.execute(
            "SELECT COUNT(DISTINCT primaryid) FROM norm_demo"
        ).fetchone()[0]  # type: ignore[index]

    def _compute_2x2(
        self,
        ingredient_key: str,
        reaction_pt: str,
        role_scope: RoleScope,
    ) -> TwoByTwo:
        """
        Single-pair 2×2 computation.

        Join strategy: LEFT JOIN from the universe so every case is
        categorised into exactly one cell.  Counts unique primaryids.
        """
        roles_list = _roles_sql_list(role_scope)
        row = self.con.execute(
            f"""
            WITH
              drug_cases AS (
                SELECT DISTINCT primaryid
                FROM norm_drug
                WHERE prod_ai = ?
                  AND role_cod IN ({roles_list})
              ),
              reac_cases AS (
                SELECT DISTINCT primaryid
                FROM norm_reac
                WHERE pt = ?
              ),
              cells AS (
                SELECT
                  u.primaryid,
                  (dc.primaryid IS NOT NULL) AS has_drug,
                  (rc.primaryid IS NOT NULL) AS has_reac
                FROM (SELECT DISTINCT primaryid FROM norm_demo) u
                LEFT JOIN drug_cases dc ON dc.primaryid = u.primaryid
                LEFT JOIN reac_cases rc ON rc.primaryid = u.primaryid
              )
            SELECT
              SUM(CASE WHEN has_drug AND     has_reac  THEN 1 ELSE 0 END)::BIGINT AS a,
              SUM(CASE WHEN has_drug AND NOT has_reac  THEN 1 ELSE 0 END)::BIGINT AS b,
              SUM(CASE WHEN NOT has_drug AND has_reac  THEN 1 ELSE 0 END)::BIGINT AS c,
              SUM(CASE WHEN NOT has_drug AND NOT has_reac THEN 1 ELSE 0 END)::BIGINT AS d
            FROM cells
            """,
            [ingredient_key, reaction_pt],
        ).fetchone()  # type: ignore[misc]
        a, b, c, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        return TwoByTwo(a=a, b=b, c=c, d=d)

    def _batch_2x2(
        self,
        ingredients: list[str],
        pts: list[str],
        role_scope: RoleScope,
        min_a: int,
    ) -> list[tuple]:
        """
        Batch 2×2 for all (ingredient, PT) pairs using a single DuckDB pass.

        Returns list of tuples:
          (ingredient_key, reaction_pt, a, b, c, d)
        filtered to a >= min_a, ordered by a DESC.
        """
        roles_list = _roles_sql_list(role_scope)

        # Register ingredient and PT lists as in-memory tables to avoid
        # generating enormous IN() clauses for large lists.
        self.con.execute("CREATE OR REPLACE TEMP TABLE _ing_list (ingredient_key VARCHAR)")
        self.con.executemany(
            "INSERT INTO _ing_list VALUES (?)",
            [[i] for i in ingredients],
        )
        self.con.execute("CREATE OR REPLACE TEMP TABLE _pt_list (reaction_pt VARCHAR)")
        self.con.executemany(
            "INSERT INTO _pt_list VALUES (?)",
            [[p] for p in pts],
        )

        rows = self.con.execute(
            f"""
            WITH
              n_total AS (
                SELECT COUNT(DISTINCT primaryid) AS n FROM norm_demo
              ),
              drug_cases AS (
                SELECT prod_ai AS ingredient_key, primaryid
                FROM norm_drug
                WHERE prod_ai IN (SELECT ingredient_key FROM _ing_list)
                  AND role_cod IN ({roles_list})
                  AND prod_ai IS NOT NULL
              ),
              reac_cases AS (
                SELECT pt AS reaction_pt, primaryid
                FROM norm_reac
                WHERE pt IN (SELECT reaction_pt FROM _pt_list)
              ),
              -- a: cases with both drug and reaction
              a_counts AS (
                SELECT dc.ingredient_key, rc.reaction_pt,
                       COUNT(DISTINCT dc.primaryid) AS a
                FROM drug_cases dc
                JOIN reac_cases rc ON rc.primaryid = dc.primaryid
                GROUP BY dc.ingredient_key, rc.reaction_pt
              ),
              drug_totals AS (
                SELECT ingredient_key,
                       COUNT(DISTINCT primaryid) AS n_drug
                FROM drug_cases
                GROUP BY ingredient_key
              ),
              reac_totals AS (
                SELECT reaction_pt,
                       COUNT(DISTINCT primaryid) AS n_reac
                FROM reac_cases
                GROUP BY reaction_pt
              ),
              pairs AS (
                SELECT i.ingredient_key, p.reaction_pt
                FROM _ing_list i CROSS JOIN _pt_list p
              )
            SELECT
              p.ingredient_key,
              p.reaction_pt,
              COALESCE(ac.a, 0)::BIGINT                          AS a,
              (dt.n_drug - COALESCE(ac.a, 0))::BIGINT            AS b,
              (rt.n_reac - COALESCE(ac.a, 0))::BIGINT            AS c,
              ((SELECT n FROM n_total)
                - dt.n_drug
                - rt.n_reac
                + COALESCE(ac.a, 0))::BIGINT                     AS d
            FROM pairs p
            JOIN drug_totals dt ON dt.ingredient_key = p.ingredient_key
            JOIN reac_totals rt ON rt.reaction_pt    = p.reaction_pt
            LEFT JOIN a_counts ac
              ON ac.ingredient_key = p.ingredient_key
             AND ac.reaction_pt    = p.reaction_pt
            WHERE COALESCE(ac.a, 0) >= {min_a}
            ORDER BY a DESC
            """,
        ).fetchall()

        # Clean up temp tables
        self.con.execute("DROP TABLE IF EXISTS _ing_list")
        self.con.execute("DROP TABLE IF EXISTS _pt_list")
        return rows  # type: ignore[return-value]

    def _top_ingredients(self, n: int, role_scope: RoleScope) -> list[str]:
        roles_list = _roles_sql_list(role_scope)
        return [
            row[0]
            for row in self.con.execute(
                f"""
                SELECT prod_ai
                FROM norm_drug
                WHERE prod_ai IS NOT NULL AND role_cod IN ({roles_list})
                GROUP BY prod_ai
                ORDER BY COUNT(DISTINCT primaryid) DESC
                LIMIT {n}
                """
            ).fetchall()
        ]

    def _top_pts(self, n: int) -> list[str]:
        return [
            row[0]
            for row in self.con.execute(
                f"""
                SELECT pt
                FROM norm_reac
                GROUP BY pt
                ORDER BY COUNT(DISTINCT primaryid) DESC
                LIMIT {n}
                """
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roles_sql_list(role_scope: RoleScope) -> str:
    """Return a SQL IN-list string for the role_scope, e.g. \"'PS','SS'\"."""
    return ", ".join(f"'{r}'" for r in role_scope.role_codes)


def _build_result(unit: SignalUnit, table: TwoByTwo) -> SignalResult:
    """
    Compute all metrics from a 2×2 table and assemble a SignalResult.

    Returns a result with metrics=None and meets_threshold=False when
    a < MIN_A or any required denominator is zero.
    """
    a, b, c, d = table.a, table.b, table.c, table.d

    if not table.sufficient:
        return SignalResult(unit=unit, table=table, meets_threshold=False)

    # --- PRR ---
    prr_val   = _stats.prr(a, b, c, d)
    prr_lower, prr_upper = _stats.prr_ci(a, b, c, d)
    prr_metric: Optional[MetricValue] = (
        MetricValue(value=prr_val, lower=prr_lower, upper=prr_upper)
        if prr_val is not None else None
    )

    # --- ROR ---
    ror_val   = _stats.ror(a, b, c, d)
    ror_lower, ror_upper = _stats.ror_ci(a, b, c, d)
    ror_metric: Optional[MetricValue] = (
        MetricValue(value=ror_val, lower=ror_lower, upper=ror_upper)
        if ror_val is not None else None
    )

    # --- Yates chi2 ---
    chi2_val = _stats.chi2_yates(a, b, c, d)

    # --- IC and IC025 ---
    ic_val   = _stats.ic(a, b, c, d)
    ic025_val = _stats.ic025(a, b, c, d)
    ic_metric: Optional[MetricValue] = (
        MetricValue(value=ic_val, lower=ic025_val, upper=None)
        if ic_val is not None else None
    )

    # meets_threshold: a >= 3 AND all metrics are computable
    meets = (
        table.sufficient
        and prr_metric is not None
        and ror_metric is not None
        and chi2_val is not None
        and ic_metric is not None
    )

    return SignalResult(
        unit=unit,
        table=table,
        prr=prr_metric,
        ror=ror_metric,
        chi2_yates=chi2_val,
        ic=ic_metric,
        meets_threshold=meets,
    )

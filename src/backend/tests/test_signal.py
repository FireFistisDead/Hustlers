"""
Signal detection tests.

Structure
---------
TestStats              — pure unit tests for stats.py (no I/O, deterministic)
TestTwoByTwo           — Pydantic model validation
TestSignalResultModel  — SignalResult model validation
TestSyntheticEngine    — synthetic DuckDB fixture, deterministic results
TestEdgeCases          — zero cells, a < 3, role_scope isolation
TestRoleScope          — ALL vs PS vs SS vs C filtering
TestRealDataSmoke      — smoke tests against actual 2026 Q2 data (session-scoped)

IMPORTANT: all assertions about "signal detected" refer to
*potential disproportionate reporting signals* only —
not confirmed safety conclusions.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pytest

from backend.app.signal.models import RoleScope, SignalUnit, TwoByTwo
from backend.app.signal import stats as S
from backend.app.signal.engine import MIN_A, SignalEngine, _build_result
from backend.app.ingest.loader import FAERSLoader
from backend.app.ingest.normalize import FAERSNormaliser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).parent
_RAW_DIR = (_HERE / ".." / ".." / "data" / "raw").resolve()
_HAS_RAW = (_RAW_DIR / "DEMO26Q2.txt").exists()


def _approx(x: float, y: float, rel: float = 1e-4) -> bool:
    """True when |x - y| / max(|y|, 1e-10) <= rel."""
    return abs(x - y) / max(abs(y), 1e-10) <= rel


# ---------------------------------------------------------------------------
# 1. Pure stats unit tests — deterministic known values
# ---------------------------------------------------------------------------

class TestStats:
    """
    Verified against hand-calculations using the textbook 2×2:
      a=100, b=900, c=50, d=8950   (N=10 000)

    PRR = (100/1000) / (50/9000) = 0.1 / 0.00556 = 18.0
    ROR = (100×8950) / (900×50)  = 895000/45000  = 19.889
    chi2_yates: using the Yates formula (large N, expected > 5 → result ~large)
    IC = log2((100.5) / (E+0.5)) where E = 1000*150/10000 = 15
    """

    # 2×2 used across all test methods
    A, B, C, D = 100, 900, 50, 8950

    def test_prr_value(self) -> None:
        result = S.prr(self.A, self.B, self.C, self.D)
        assert result is not None
        expected = (self.A / (self.A + self.B)) / (self.C / (self.C + self.D))
        assert _approx(result, expected), f"PRR={result} expected≈{expected}"

    def test_prr_exact(self) -> None:
        # (100/1000) / (50/9000) = 0.1 / (50/9000) = 0.1 * 9000/50 = 18.0
        result = S.prr(self.A, self.B, self.C, self.D)
        assert result is not None
        assert _approx(result, 18.0), f"PRR={result} expected≈18.0"

    def test_ror_value(self) -> None:
        result = S.ror(self.A, self.B, self.C, self.D)
        assert result is not None
        expected = (self.A * self.D) / (self.B * self.C)
        assert _approx(result, expected)

    def test_ror_exact(self) -> None:
        # (100*8950)/(900*50) = 895000/45000 ≈ 19.889
        result = S.ror(self.A, self.B, self.C, self.D)
        assert result is not None
        assert _approx(result, 895000 / 45000), f"ROR={result}"

    def test_prr_gt_1_when_disproportionate(self) -> None:
        result = S.prr(self.A, self.B, self.C, self.D)
        assert result is not None and result > 1.0

    def test_ror_gt_1_when_disproportionate(self) -> None:
        result = S.ror(self.A, self.B, self.C, self.D)
        assert result is not None and result > 1.0

    def test_prr_equals_1_when_proportional(self) -> None:
        # a/(a+b) == c/(c+d) when a=100, b=900, c=100, d=900
        result = S.prr(100, 900, 100, 900)
        assert result is not None
        assert _approx(result, 1.0), f"PRR={result} expected=1.0"

    def test_ror_equals_1_when_proportional(self) -> None:
        result = S.ror(100, 900, 100, 900)
        assert result is not None
        assert _approx(result, 1.0), f"ROR={result} expected=1.0"

    def test_chi2_yates_large_for_strong_signal(self) -> None:
        result = S.chi2_yates(self.A, self.B, self.C, self.D)
        assert result is not None and result > 100.0

    def test_chi2_yates_near_zero_for_no_signal(self) -> None:
        # Perfect proportionality → chi2 ≈ 0
        result = S.chi2_yates(100, 900, 100, 900)
        assert result is not None and result < 0.1

    def test_chi2_yates_symmetry(self) -> None:
        # Swapping rows (a↔c, b↔d) gives same chi2
        r1 = S.chi2_yates(self.A, self.B, self.C, self.D)
        r2 = S.chi2_yates(self.C, self.D, self.A, self.B)
        assert r1 is not None and r2 is not None
        assert _approx(r1, r2), f"chi2 not symmetric: {r1} vs {r2}"

    def test_ic_positive_when_disproportionate(self) -> None:
        result = S.ic(self.A, self.B, self.C, self.D)
        assert result is not None and result > 0.0

    def test_ic_near_zero_when_proportional(self) -> None:
        # E_a = (a+b)*(a+c)/N = 1000*200/10000 = 20
        # IC = log2(20.5/20.5) = log2(1) = 0.0 (with shrinkage, near 0)
        result = S.ic(100, 900, 100, 900)
        assert result is not None and abs(result) < 0.1

    def test_ic025_less_than_ic(self) -> None:
        ic_val   = S.ic(self.A, self.B, self.C, self.D)
        ic025_val = S.ic025(self.A, self.B, self.C, self.D)
        assert ic_val is not None and ic025_val is not None
        assert ic025_val < ic_val

    def test_ic025_positive_for_strong_signal(self) -> None:
        ic025_val = S.ic025(self.A, self.B, self.C, self.D)
        assert ic025_val is not None and ic025_val > 0.0

    def test_prr_none_when_c_zero(self) -> None:
        # c=0 → background proportion undefined
        assert S.prr(10, 90, 0, 900) is None

    def test_ror_none_when_b_zero(self) -> None:
        assert S.ror(10, 0, 50, 940) is None

    def test_ror_none_when_c_zero(self) -> None:
        assert S.ror(10, 90, 0, 900) is None

    def test_chi2_yates_none_when_marginal_zero(self) -> None:
        # a+b=0 makes the row marginal zero
        assert S.chi2_yates(0, 0, 50, 950) is None

    def test_ic_none_when_N_zero(self) -> None:
        assert S.ic(0, 0, 0, 0) is None

    def test_prr_ci_lower_lt_upper(self) -> None:
        lo, hi = S.prr_ci(self.A, self.B, self.C, self.D)
        assert lo is not None and hi is not None
        assert lo < hi

    def test_ror_ci_lower_lt_upper(self) -> None:
        lo, hi = S.ror_ci(self.A, self.B, self.C, self.D)
        assert lo is not None and hi is not None
        assert lo < hi

    def test_prr_ci_contains_point_estimate(self) -> None:
        point = S.prr(self.A, self.B, self.C, self.D)
        lo, hi = S.prr_ci(self.A, self.B, self.C, self.D)
        assert lo is not None and hi is not None and point is not None
        assert lo < point < hi

    def test_ror_ci_contains_point_estimate(self) -> None:
        point = S.ror(self.A, self.B, self.C, self.D)
        lo, hi = S.ror_ci(self.A, self.B, self.C, self.D)
        assert lo is not None and hi is not None and point is not None
        assert lo < point < hi

    def test_known_dupilumab_conjunctivitis_approximation(self) -> None:
        """
        Cross-check against independently verified 26Q2 value for
        DUPILUMAB / Conjunctivitis (PS scope):
          a=282, b=46624, c=282, d=375270
        PRR≈8.006, ROR≈8.049, chi2_yates≈861.7, IC≈2.16
        """
        a, b, c, d = 282, 46624, 282, 375270
        prr_val  = S.prr(a, b, c, d)
        ror_val  = S.ror(a, b, c, d)
        chi2_val = S.chi2_yates(a, b, c, d)
        ic_val   = S.ic(a, b, c, d)
        ic025_val = S.ic025(a, b, c, d)
        assert prr_val  is not None and _approx(prr_val,   8.006, rel=0.01)
        assert ror_val  is not None and _approx(ror_val,   8.049, rel=0.01)
        assert chi2_val is not None and _approx(chi2_val, 861.7,  rel=0.02)
        assert ic_val   is not None and _approx(ic_val,    2.162, rel=0.01)
        assert ic025_val is not None and ic025_val < ic_val and ic025_val > 1.9


# ---------------------------------------------------------------------------
# 2. TwoByTwo model tests
# ---------------------------------------------------------------------------

class TestTwoByTwo:
    def test_basic_construction(self) -> None:
        t = TwoByTwo(a=10, b=90, c=5, d=895)
        assert t.N == 1000
        assert t.n_drug == 100
        assert t.n_reac == 15

    def test_sufficient_when_a_ge_3(self) -> None:
        assert TwoByTwo(a=3, b=97, c=5, d=895).sufficient is True

    def test_not_sufficient_when_a_lt_3(self) -> None:
        assert TwoByTwo(a=2, b=98, c=5, d=895).sufficient is False
        assert TwoByTwo(a=0, b=100, c=5, d=895).sufficient is False

    def test_n_equals_sum(self) -> None:
        t = TwoByTwo(a=7, b=43, c=12, d=938)
        assert t.N == 7 + 43 + 12 + 938

    def test_rejects_negative_cell(self) -> None:
        with pytest.raises(Exception):
            TwoByTwo(a=-1, b=100, c=5, d=894)


# ---------------------------------------------------------------------------
# 3. SignalResult model tests
# ---------------------------------------------------------------------------

class TestSignalResultModel:
    def test_disclaimer_always_present(self) -> None:
        unit  = SignalUnit(ingredient_key="DRUG_X", reaction_pt="PT_Y")
        table = TwoByTwo(a=1, b=99, c=5, d=895)
        result = _build_result(unit, table)
        assert result.disclaimer
        assert "NOT" in result.disclaimer or "not" in result.disclaimer.lower()

    def test_metrics_none_when_below_threshold(self) -> None:
        unit  = SignalUnit(ingredient_key="DRUG_X", reaction_pt="PT_Y")
        table = TwoByTwo(a=2, b=98, c=5, d=895)  # a < 3
        result = _build_result(unit, table)
        assert result.meets_threshold is False
        assert result.prr is None
        assert result.ror is None
        assert result.chi2_yates is None
        assert result.ic is None

    def test_metrics_populated_when_above_threshold(self) -> None:
        unit  = SignalUnit(ingredient_key="DRUG_X", reaction_pt="PT_Y")
        table = TwoByTwo(a=100, b=900, c=50, d=8950)
        result = _build_result(unit, table)
        assert result.meets_threshold is True
        assert result.prr   is not None
        assert result.ror   is not None
        assert result.chi2_yates is not None
        assert result.ic    is not None
        assert result.ic.lower is not None   # IC025

    def test_ic025_in_ic_lower_field(self) -> None:
        unit  = SignalUnit(ingredient_key="X", reaction_pt="Y")
        table = TwoByTwo(a=100, b=900, c=50, d=8950)
        result = _build_result(unit, table)
        assert result.ic is not None
        assert result.ic.lower == S.ic025(100, 900, 50, 8950)

    def test_prr_bounds_present(self) -> None:
        unit  = SignalUnit(ingredient_key="X", reaction_pt="Y")
        table = TwoByTwo(a=100, b=900, c=50, d=8950)
        result = _build_result(unit, table)
        assert result.prr is not None
        assert result.prr.lower is not None
        assert result.prr.upper is not None
        assert result.prr.lower < result.prr.value < result.prr.upper


# ---------------------------------------------------------------------------
# 4. Synthetic engine tests — fully in-memory DuckDB fixture
# ---------------------------------------------------------------------------

def _make_synthetic_db() -> duckdb.DuckDBPyConnection:
    """
    Build a minimal DuckDB connection with norm_demo, norm_drug, norm_reac views
    populated from synthetic data for fully deterministic testing.

    Universe: 1 000 cases (primaryid 1..1000)

    Drug population (prod_ai, role_cod, primaryid range):
      DRUG_A (PS): cases 1-200   (200 cases)
      DRUG_B (PS): cases 101-250 (150 cases, 101-200 overlap with DRUG_A)
      DRUG_C (SS): cases 1-100   (100 cases, same as first half of DRUG_A)
      DRUG_D (C):  cases 1-50    (50 cases)

    Reaction population (pt, primaryid range):
      REAC_X: cases 1-100    (100 cases — all in DRUG_A arm)
      REAC_Y: cases 51-300   (250 cases — partial overlap)
      REAC_Z: cases 901-1000 (100 cases — outside DRUG_A arm entirely)

    Known 2×2 values (DRUG_A / REAC_X / PS):
      a = cases with DRUG_A (PS) AND REAC_X = cases in [1-200] ∩ [1-100] = 100
      b = cases with DRUG_A (PS) but NOT REAC_X = 200-100 = 100
      c = cases without DRUG_A but with REAC_X = 100-100 = 0   ← c=0
      → PRR undefined (c=0); meets_threshold=False despite a=100

    Known 2×2 values (DRUG_A / REAC_Y / PS):
      drug arm: cases 1-200
      REAC_Y:   cases 51-300
      a = overlap [1-200] ∩ [51-300] = [51-200] = 150
      b = [1-200] not in [51-300] = [1-50] = 50
      c = [51-300] not in [1-200] = [201-300] = 100
      d = 1000 - 150 - 50 - 100 = 700
      N = 1000  ✓
      PRR = (150/200) / (100/800) = 0.75 / 0.125 = 6.0
      ROR = (150*700) / (50*100)  = 105000/5000  = 21.0

    Known 2×2 values (DRUG_A / REAC_Z / PS):
      a = [1-200] ∩ [901-1000] = 0  → a < 3 → no result
    """
    con = duckdb.connect(":memory:")

    # norm_demo: 1000 cases
    con.execute("""
        CREATE TABLE norm_demo AS
        SELECT CAST(i AS VARCHAR) AS primaryid,
               CAST(i AS VARCHAR) AS caseid,
               1                  AS caseversion,
               'I'                AS i_f_code
        FROM generate_series(1, 1000) t(i)
    """)

    # norm_drug: DRUG_A (PS) 1-200, DRUG_B (PS) 101-250,
    #            DRUG_C (SS) 1-100, DRUG_D (C) 1-50
    con.execute("""
        CREATE TABLE norm_drug AS
        SELECT CAST(i AS VARCHAR) AS primaryid,
               CAST(i AS VARCHAR) AS caseid,
               1                  AS drug_seq,
               'PS'               AS role_cod,
               'DRUG_A_BRAND'     AS drugname,
               'DRUG_A'           AS prod_ai,
               '1'                AS val_vbm,
               NULL::VARCHAR      AS cum_dose_chr,
               NULL::VARCHAR      AS cum_dose_unit,
               NULL::VARCHAR      AS dechal,
               NULL::VARCHAR      AS rechal,
               NULL::VARCHAR      AS nda_num
        FROM generate_series(1, 200) t(i)
        UNION ALL
        SELECT CAST(i AS VARCHAR), CAST(i AS VARCHAR), 2, 'PS',
               'DRUG_B_BRAND', 'DRUG_B', '1',
               NULL,NULL,NULL,NULL,NULL
        FROM generate_series(101, 250) t(i)
        UNION ALL
        SELECT CAST(i AS VARCHAR), CAST(i AS VARCHAR), 3, 'SS',
               'DRUG_C_BRAND', 'DRUG_C', '1',
               NULL,NULL,NULL,NULL,NULL
        FROM generate_series(1, 100) t(i)
        UNION ALL
        SELECT CAST(i AS VARCHAR), CAST(i AS VARCHAR), 4, 'C',
               'DRUG_D_BRAND', 'DRUG_D', '1',
               NULL,NULL,NULL,NULL,NULL
        FROM generate_series(1, 50) t(i)
    """)

    # norm_reac: REAC_X 1-100, REAC_Y 51-300, REAC_Z 901-1000
    con.execute("""
        CREATE TABLE norm_reac AS
        SELECT CAST(i AS VARCHAR) AS primaryid,
               CAST(i AS VARCHAR) AS caseid,
               'REAC_X'          AS pt,
               NULL::VARCHAR      AS drug_rec_act
        FROM generate_series(1, 100) t(i)
        UNION ALL
        SELECT CAST(i AS VARCHAR), CAST(i AS VARCHAR),
               'REAC_Y', NULL::VARCHAR
        FROM generate_series(51, 300) t(i)
        UNION ALL
        SELECT CAST(i AS VARCHAR), CAST(i AS VARCHAR),
               'REAC_Z', NULL::VARCHAR
        FROM generate_series(901, 1000) t(i)
    """)

    return con


@pytest.fixture(scope="module")
def synth_engine() -> SignalEngine:
    con = _make_synthetic_db()
    return SignalEngine(con)


class TestSyntheticEngine:
    def test_drug_a_reac_y_ps_2x2_exact(self, synth_engine: SignalEngine) -> None:
        """
        DRUG_A / REAC_Y / PS:
          a=150  b=50  c=100  d=700  N=1000
        """
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        t = result.table
        assert t.a == 150, f"a={t.a}"
        assert t.b == 50,  f"b={t.b}"
        assert t.c == 100, f"c={t.c}"
        assert t.d == 700, f"d={t.d}"
        assert t.N == 1000

    def test_drug_a_reac_y_ps_prr(self, synth_engine: SignalEngine) -> None:
        """PRR = (150/200) / (100/800) = 6.0"""
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        assert result.prr is not None
        assert _approx(result.prr.value, 6.0), f"PRR={result.prr.value}"

    def test_drug_a_reac_y_ps_ror(self, synth_engine: SignalEngine) -> None:
        """ROR = (150*700)/(50*100) = 21.0"""
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        assert result.ror is not None
        assert _approx(result.ror.value, 21.0), f"ROR={result.ror.value}"

    def test_drug_a_reac_y_ps_meets_threshold(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        assert result.meets_threshold is True

    def test_drug_a_reac_y_ps_ic_positive(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        assert result.ic is not None and result.ic.value > 0.0

    def test_drug_a_reac_y_ps_ic025_present(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS)
        assert result.ic is not None and result.ic.lower is not None

    def test_2x2_sums_to_N(self, synth_engine: SignalEngine) -> None:
        """a + b + c + d must always equal N regardless of pair."""
        for ing, pt in [("DRUG_A", "REAC_Y"), ("DRUG_B", "REAC_Y"), ("DRUG_A", "REAC_Z")]:
            result = synth_engine.run_single(ing, pt, RoleScope.PS)
            t = result.table
            assert t.N == 1000, f"{ing}/{pt}: N={t.N}"

    def test_batch_returns_subset_of_single(self, synth_engine: SignalEngine) -> None:
        """run_batch results must match run_single for the same pair."""
        singles = {
            ("DRUG_A", "REAC_Y"): synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.PS),
        }
        batch = synth_engine.run_batch(
            ["DRUG_A", "DRUG_B"], ["REAC_X", "REAC_Y", "REAC_Z"], RoleScope.PS
        )
        batch_map = {
            (r.unit.ingredient_key, r.unit.reaction_pt): r for r in batch
        }
        # DRUG_A / REAC_Y must appear in batch (a=150 >= 3)
        assert ("DRUG_A", "REAC_Y") in batch_map
        b_result = batch_map[("DRUG_A", "REAC_Y")]
        s_result = singles[("DRUG_A", "REAC_Y")]
        assert b_result.table.a == s_result.table.a
        assert b_result.table.b == s_result.table.b
        assert b_result.table.c == s_result.table.c
        assert b_result.table.d == s_result.table.d


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_a_below_3_no_metrics(self, synth_engine: SignalEngine) -> None:
        """DRUG_B / REAC_X: cases [101-250] ∩ [1-100] = 0 → a=0 < 3."""
        result = synth_engine.run_single("DRUG_B", "REAC_X", RoleScope.PS)
        assert result.table.a == 0
        assert result.meets_threshold is False
        assert result.prr is None

    def test_a_exactly_3_meets_threshold(self) -> None:
        """Boundary: a=3 must meet threshold."""
        from backend.app.signal.models import SignalUnit
        unit  = SignalUnit(ingredient_key="X", reaction_pt="Y")
        table = TwoByTwo(a=3, b=97, c=10, d=890)
        result = _build_result(unit, table)
        assert result.meets_threshold is True
        assert result.prr is not None

    def test_a_equals_2_does_not_meet_threshold(self) -> None:
        unit  = SignalUnit(ingredient_key="X", reaction_pt="Y")
        table = TwoByTwo(a=2, b=98, c=10, d=890)
        result = _build_result(unit, table)
        assert result.meets_threshold is False

    def test_c_zero_prr_undefined(self) -> None:
        """DRUG_A / REAC_X / PS: all reaction cases are in the drug arm → c=0."""
        result = synth_engine_global().run_single("DRUG_A", "REAC_X", RoleScope.PS)
        t = result.table
        assert t.c == 0
        # a=100 >= 3 but PRR undefined (c=0)
        assert result.prr is None
        assert result.meets_threshold is False

    def test_disclaimer_always_in_result(self) -> None:
        unit  = SignalUnit(ingredient_key="X", reaction_pt="Y")
        table = TwoByTwo(a=0, b=100, c=0, d=900)
        result = _build_result(unit, table)
        assert len(result.disclaimer) > 0

    def test_unknown_drug_returns_zero_a(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("NO_SUCH_DRUG", "REAC_Y", RoleScope.PS)
        assert result.table.a == 0
        assert result.meets_threshold is False

    def test_unknown_pt_returns_zero_a(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_A", "NO_SUCH_PT", RoleScope.PS)
        assert result.table.a == 0
        assert result.meets_threshold is False


def synth_engine_global() -> SignalEngine:
    """Helper for edge-case tests that can't use the module fixture."""
    return SignalEngine(_make_synthetic_db())


# ---------------------------------------------------------------------------
# 6. Role scope isolation
# ---------------------------------------------------------------------------

class TestRoleScope:
    """
    DRUG_C appears only as SS (cases 1-100).
    DRUG_D appears only as C  (cases 1-50).
    When querying PS scope, neither should contribute any drug-arm cases.
    """

    def test_ps_scope_excludes_ss_drug(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_C", "REAC_Y", RoleScope.PS)
        assert result.table.n_drug == 0, (
            f"DRUG_C is SS only; PS scope should give n_drug=0, got {result.table.n_drug}"
        )

    def test_ss_scope_finds_drug_c(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_C", "REAC_Y", RoleScope.SS)
        # DRUG_C (SS): cases 1-100; REAC_Y: 51-300 → overlap [51-100] = 50
        assert result.table.a == 50, f"DRUG_C/REAC_Y/SS a={result.table.a}, expected 50"

    def test_c_scope_finds_drug_d(self, synth_engine: SignalEngine) -> None:
        result = synth_engine.run_single("DRUG_D", "REAC_Y", RoleScope.C)
        # DRUG_D (C): cases 1-50; REAC_Y: 51-300 → overlap = 0 → a=0
        assert result.table.a == 0, f"DRUG_D/REAC_Y/C a={result.table.a}, expected 0"

    def test_all_scope_includes_all_roles(self, synth_engine: SignalEngine) -> None:
        # For DRUG_A (ALL scope), n_drug should still be 200 (only PS rows for DRUG_A)
        result = synth_engine.run_single("DRUG_A", "REAC_Y", RoleScope.ALL)
        assert result.table.n_drug == 200

    def test_all_scope_picks_up_ss_drug(self, synth_engine: SignalEngine) -> None:
        # DRUG_C appears under SS; ALL scope must find it
        result_all = synth_engine.run_single("DRUG_C", "REAC_Y", RoleScope.ALL)
        result_ps  = synth_engine.run_single("DRUG_C", "REAC_Y", RoleScope.PS)
        assert result_all.table.n_drug > result_ps.table.n_drug

    def test_role_scope_codes_all(self) -> None:
        assert set(RoleScope.ALL.role_codes) == {"PS", "SS", "C", "I", "DN"}

    def test_role_scope_codes_ps(self) -> None:
        assert RoleScope.PS.role_codes == ["PS"]


# ---------------------------------------------------------------------------
# 7. Real data smoke tests (session-scoped, skipped when data absent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_engine():
    if not _HAS_RAW:
        pytest.skip("FAERS 26Q2 raw data not present")
    ldr  = FAERSLoader(raw_dir=_RAW_DIR, db_path=":memory:")
    ldr.load_all()
    norm = FAERSNormaliser(ldr)
    norm.normalise_all()
    return SignalEngine(norm.con)


@pytest.mark.skipif(not _HAS_RAW, reason="FAERS 26Q2 raw data not present")
class TestRealDataSmoke:
    """
    Smoke tests against actual 2026 Q2 FAERS data.

    Expected values were independently verified via direct SQL during
    pre-implementation audit (see comments in test_ingest.py).
    """

    def test_dupilumab_conjunctivitis_ps_2x2(self, real_engine: SignalEngine) -> None:
        """
        Independently verified 26Q2 value:
          DUPILUMAB / Conjunctivitis / PS → a=282, b=46624, c=282, d=375270
        """
        result = real_engine.run_single("DUPILUMAB", "Conjunctivitis", RoleScope.PS)
        t = result.table
        assert t.a   == 282,    f"a={t.a}"
        assert t.b   == 46624,  f"b={t.b}"
        assert t.c   == 282,    f"c={t.c}"
        assert t.d   == 375270, f"d={t.d}"
        assert t.N   == 422458, f"N={t.N}"

    def test_dupilumab_conjunctivitis_ps_prr(self, real_engine: SignalEngine) -> None:
        result = real_engine.run_single("DUPILUMAB", "Conjunctivitis", RoleScope.PS)
        assert result.prr is not None
        assert _approx(result.prr.value, 8.006, rel=0.01)

    def test_dupilumab_conjunctivitis_ps_ror(self, real_engine: SignalEngine) -> None:
        result = real_engine.run_single("DUPILUMAB", "Conjunctivitis", RoleScope.PS)
        assert result.ror is not None
        assert _approx(result.ror.value, 8.049, rel=0.01)

    def test_dupilumab_conjunctivitis_ps_ic025_positive(self, real_engine: SignalEngine) -> None:
        result = real_engine.run_single("DUPILUMAB", "Conjunctivitis", RoleScope.PS)
        assert result.ic is not None
        assert result.ic.lower is not None
        assert result.ic.lower > 0, (
            "IC025 for a known strong signal should be > 0 (WHO UMC criterion)"
        )

    def test_dupilumab_conjunctivitis_meets_threshold(self, real_engine: SignalEngine) -> None:
        result = real_engine.run_single("DUPILUMAB", "Conjunctivitis", RoleScope.PS)
        assert result.meets_threshold is True

    def test_2x2_always_sums_to_N(self, real_engine: SignalEngine) -> None:
        """For multiple real pairs, a+b+c+d must equal the universe N."""
        N = 422_458
        pairs = [
            ("DUPILUMAB",  "Conjunctivitis",     RoleScope.PS),
            ("TIRZEPATIDE","Nausea",              RoleScope.PS),
            ("SEMAGLUTIDE", "Drug ineffective",  RoleScope.PS),
            ("ACETAMINOPHEN","Hepatotoxicity",    RoleScope.ALL),
        ]
        for ing, pt, scope in pairs:
            result = real_engine.run_single(ing, pt, scope)
            t = result.table
            assert t.N == N, f"{ing}/{pt}/{scope}: N={t.N} expected {N}"

    def test_role_scope_ps_le_all(self, real_engine: SignalEngine) -> None:
        """PS scope must never produce more drug-arm cases than ALL scope."""
        for ing in ["DUPILUMAB", "TIRZEPATIDE", "SEMAGLUTIDE"]:
            ps  = real_engine.run_single(ing, "Nausea", RoleScope.PS)
            all_ = real_engine.run_single(ing, "Nausea", RoleScope.ALL)
            assert ps.table.n_drug <= all_.table.n_drug, (
                f"{ing}: PS n_drug={ps.table.n_drug} > ALL n_drug={all_.table.n_drug}"
            )

    def test_batch_consistent_with_single(self, real_engine: SignalEngine) -> None:
        """Batch 2×2 must match single 2×2 for every pair in the batch."""
        ingredients = ["DUPILUMAB", "TIRZEPATIDE"]
        pts         = ["Conjunctivitis", "Nausea", "Fatigue"]
        batch = real_engine.run_batch(ingredients, pts, RoleScope.PS, min_a=1)
        batch_map = {
            (r.unit.ingredient_key, r.unit.reaction_pt): r for r in batch
        }
        for ing in ingredients:
            for pt in pts:
                single = real_engine.run_single(ing, pt, RoleScope.PS)
                if single.table.a < 1:
                    continue
                key = (ing, pt)
                assert key in batch_map, f"{key} missing from batch"
                bt = batch_map[key].table
                st = single.table
                assert bt.a == st.a, f"{key} batch a={bt.a} single a={st.a}"
                assert bt.b == st.b, f"{key} batch b={bt.b} single b={st.b}"
                assert bt.c == st.c, f"{key} batch c={bt.c} single c={st.c}"
                assert bt.d == st.d, f"{key} batch d={bt.d} single d={st.d}"

    def test_top_signals_returns_results(self, real_engine: SignalEngine) -> None:
        results = real_engine.top_signals(
            role_scope=RoleScope.PS,
            top_n_ingredients=50,
            top_n_pts=50,
        )
        assert len(results) > 0
        # All returned results must meet threshold
        assert all(r.meets_threshold for r in results)
        # All must have a disclaimer
        assert all(len(r.disclaimer) > 0 for r in results)
        # IC025 should be sorted descending
        ic025s = [r.ic.lower for r in results if r.ic and r.ic.lower is not None]
        assert ic025s == sorted(ic025s, reverse=True)

    def test_top_signals_disclaimer_present(self, real_engine: SignalEngine) -> None:
        results = real_engine.top_signals(
            role_scope=RoleScope.PS, top_n_ingredients=20, top_n_pts=20
        )
        for r in results:
            assert "NOT" in r.disclaimer or "not" in r.disclaimer.lower(), (
                "Disclaimer must contain a negation (NOT/not)"
            )

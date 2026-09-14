"""
Signal detection — pure statistical functions.

All functions take raw 2×2 cell counts and return float metrics.
No DuckDB, no I/O, no side effects — fully testable in isolation.

DISCLAIMER
----------
These metrics measure *disproportionate reporting* in spontaneous
adverse-event data.  They are NOT measures of causality or clinical risk.

References
----------
Evans SJW, Waller PC, Davis S (2001).
  "Use of proportional reporting ratios (PRRs) for signal generation
   from spontaneous adverse drug reaction reports."
  Pharmacoepidemiol Drug Saf 10: 483-486.

Bate A, Evans SJW (2009).
  "Quantitative signal detection using spontaneous ADR reporting."
  Pharmacoepidemiol Drug Saf 18: 427-436.

Rothman KJ, Lanes S, Sacks ST (2004).
  "The reporting odds ratio and its advantages over the proportional
   reporting ratio." Pharmacoepidemiol Drug Saf 13: 519-523.
"""
from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# PRR — Proportional Reporting Ratio
# ---------------------------------------------------------------------------

def prr(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    PRR = (a / (a+b)) / (c / (c+d))

    Returns None when the background proportion is zero or the drug arm
    is empty (avoids division by zero; caller must handle None).

    Evans et al. signal criterion: PRR >= 2, chi2 >= 4, a >= 3.
    This function returns the point estimate only; criterion application
    is the caller's responsibility.
    """
    n_drug  = a + b
    n_other = c + d
    if n_drug == 0 or n_other == 0 or c == 0:
        return None
    return (a / n_drug) / (c / n_other)


def prr_se(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    Standard error of log(PRR) on the log scale.
    SE(ln PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    Returns None when any cell is zero.
    """
    n_drug  = a + b
    n_other = c + d
    if a == 0 or b == 0 or c == 0 or d == 0:
        return None
    return math.sqrt(1/a - 1/n_drug + 1/c - 1/n_other)


def prr_ci(a: int, b: int, c: int, d: int,
           z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """
    95 % CI for PRR on the log scale.
    Returns (lower, upper) or (None, None) if SE is undefined.
    """
    point = prr(a, b, c, d)
    se    = prr_se(a, b, c, d)
    if point is None or se is None:
        return None, None
    log_prr   = math.log(point)
    return math.exp(log_prr - z * se), math.exp(log_prr + z * se)


# ---------------------------------------------------------------------------
# ROR — Reporting Odds Ratio
# ---------------------------------------------------------------------------

def ror(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    ROR = (a × d) / (b × c)

    Equivalent to the odds ratio from a logistic regression with a single
    binary exposure and a single binary outcome.
    Returns None when b or c is zero.
    """
    if b == 0 or c == 0:
        return None
    return (a * d) / (b * c)


def ror_se(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    SE of log(ROR) = sqrt(1/a + 1/b + 1/c + 1/d)
    Returns None when any cell is zero.
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        return None
    return math.sqrt(1/a + 1/b + 1/c + 1/d)


def ror_ci(a: int, b: int, c: int, d: int,
           z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """95 % CI for ROR.  Returns (lower, upper) or (None, None)."""
    point = ror(a, b, c, d)
    se    = ror_se(a, b, c, d)
    if point is None or se is None:
        return None, None
    log_ror = math.log(point)
    return math.exp(log_ror - z * se), math.exp(log_ror + z * se)


# ---------------------------------------------------------------------------
# Yates-corrected chi-squared
# ---------------------------------------------------------------------------

def chi2_yates(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    Yates-corrected chi-squared statistic for a 2×2 table.

    χ²_Yates = N × (|ad − bc| − N/2)² / [(a+b)(c+d)(a+c)(b+d)]

    Yates correction reduces Type I error for sparse tables.
    Returns None when any marginal is zero (undefined statistic).
    A chi2 >= 3.84 corresponds to p < 0.05 (1 df).
    Evans criterion uses chi2 >= 4 as a convenient threshold.
    """
    N = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0 or N == 0:
        return None
    numerator   = N * (max(0.0, abs(a * d - b * c) - N / 2.0) ** 2)
    denominator = row1 * row2 * col1 * col2
    if denominator == 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# IC and IC025 — Information Component (Bayesian, Bate & Evans 2009)
# ---------------------------------------------------------------------------

def ic(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    IC = log2((a + 0.5) / (E_a + 0.5))

    where E_a = (a+b)(a+c) / N  is the expected count under independence.

    The 0.5 shrinkage terms prevent log(0) and reduce variance for small a.
    Returns None when N is zero or E_a is indeterminate.
    """
    N = a + b + c + d
    if N == 0:
        return None
    n_drug = a + b
    n_reac = a + c
    E_a    = (n_drug * n_reac) / N
    return math.log2((a + 0.5) / (E_a + 0.5))


def ic025(a: int, b: int, c: int, d: int) -> Optional[float]:
    """
    IC025 — approximate 2.5th-percentile lower bound of the IC.

    Formula (Bate & Evans 2009, simplified):
      IC025 ≈ IC − 3.3 × (1/√(a+0.5) − 1/√(N+0.5))

    A positive IC025 is the signal criterion used by the WHO Uppsala
    Monitoring Centre.  This implementation uses the closed-form
    approximation; it does not perform a full Gamma-Poisson integral.

    Returns None when ic() is None.
    """
    ic_val = ic(a, b, c, d)
    if ic_val is None:
        return None
    N = a + b + c + d
    # Variance approximation on the log2 scale
    variance_term = 3.3 * (1.0 / math.sqrt(a + 0.5) - 1.0 / math.sqrt(N + 0.5))
    return ic_val - variance_term

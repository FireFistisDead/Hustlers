"""
Signal detection — Pydantic models.

IMPORTANT DISCLAIMER
--------------------
All results produced by this module represent potential disproportionate
reporting signals in spontaneous adverse-event data.  They are purely
statistical associations between an ingredient and a MedDRA PT term in the
FDA FAERS database.

They are NOT:
  - Evidence of causality
  - Confirmed safety signals
  - Regulatory conclusions
  - Medical advice

Spontaneous reporting data has well-known biases (notoriety bias,
stimulated reporting, indication confounding).  Every result requires
clinical and regulatory expert review before any regulatory action.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Role scope
# ---------------------------------------------------------------------------

class RoleScope(str, Enum):
    """
    Which drug role_cod values are included in the drug arm of the 2×2.

    ALL  — any role (PS + SS + C + I + DN)
    PS   — Primary Suspect only
    SS   — Secondary Suspect only
    C    — Concomitant only
    """
    ALL = "ALL"
    PS  = "PS"
    SS  = "SS"
    C   = "C"

    @property
    def role_codes(self) -> list[str]:
        """Return the list of raw FAERS role_cod values for this scope."""
        if self is RoleScope.ALL:
            return ["PS", "SS", "C", "I", "DN"]
        return [self.value]


# ---------------------------------------------------------------------------
# Signal unit — the key that uniquely identifies a signal candidate
# ---------------------------------------------------------------------------

class SignalUnit(BaseModel):
    """
    The unit of analysis for a disproportionality signal.

    ingredient_key : prod_ai value from norm_drug (e.g. "DUPILUMAB")
    reaction_pt    : MedDRA PT term from norm_reac  (e.g. "Conjunctivitis")
    role_scope     : which drug roles are included in the drug arm
    """
    model_config = ConfigDict(frozen=True)

    ingredient_key: str = Field(..., description="prod_ai — active ingredient")
    reaction_pt: str    = Field(..., description="MedDRA Preferred Term")
    role_scope: RoleScope = Field(RoleScope.PS, description="Drug role filter")


# ---------------------------------------------------------------------------
# 2×2 contingency table
# ---------------------------------------------------------------------------

class TwoByTwo(BaseModel):
    """
    Standard 2×2 pharmacovigilance contingency table.

    Rows:    drug exposed / not exposed
    Columns: reaction present / absent

    All counts are unique primaryid counts (never raw FAERS row counts).

                 Reaction+   Reaction-
    Drug+          a            b        → n_drug  = a + b
    Drug-          c            d        → n_other = c + d
                 -------    --------
                  n_reac    n_no_reac     N = a + b + c + d
    """
    model_config = ConfigDict(frozen=True)

    a: int = Field(..., ge=0, description="Cases with drug AND reaction")
    b: int = Field(..., ge=0, description="Cases with drug but NOT reaction")
    c: int = Field(..., ge=0, description="Cases WITHOUT drug but WITH reaction")
    d: int = Field(..., ge=0, description="Cases with NEITHER drug NOR reaction")

    @model_validator(mode="after")
    def _all_non_negative(self) -> "TwoByTwo":
        for name, val in [("a", self.a), ("b", self.b), ("c", self.c), ("d", self.d)]:
            if val < 0:
                raise ValueError(f"2×2 cell '{name}' must be >= 0, got {val}")
        return self

    @property
    def n_drug(self) -> int:
        """Total cases in the drug-exposed arm (a + b)."""
        return self.a + self.b

    @property
    def n_reac(self) -> int:
        """Total cases with this reaction across all drugs (a + c)."""
        return self.a + self.c

    @property
    def n_other(self) -> int:
        """Total cases NOT in the drug arm (c + d)."""
        return self.c + self.d

    @property
    def N(self) -> int:
        """Total case universe (a + b + c + d)."""
        return self.a + self.b + self.c + self.d

    @property
    def sufficient(self) -> bool:
        """True if the minimum case threshold (a >= 3) is met."""
        return self.a >= 3


# ---------------------------------------------------------------------------
# Individual metric result
# ---------------------------------------------------------------------------

class MetricValue(BaseModel):
    """A single computed metric with its point estimate and optional bounds."""
    model_config = ConfigDict(frozen=True)

    value: float = Field(..., description="Point estimate")
    lower: Optional[float] = Field(None, description="Lower confidence bound (e.g. IC025)")
    upper: Optional[float] = Field(None, description="Upper confidence bound")


# ---------------------------------------------------------------------------
# Signal result — the full output for one SignalUnit
# ---------------------------------------------------------------------------

class SignalResult(BaseModel):
    """
    Disproportionality analysis result for one (ingredient, PT, role_scope) unit.

    DISCLAIMER: This is a *potential disproportionate reporting signal* only.
    It is NOT a confirmed safety signal or causal association.
    All results require clinical and regulatory expert review.
    """
    model_config = ConfigDict(frozen=True)

    # Identity
    unit: SignalUnit

    # Raw 2×2 table (fully transparent)
    table: TwoByTwo

    # Metrics (None if a < 3 or denominator is zero)
    prr:   Optional[MetricValue] = Field(None, description="Proportional Reporting Ratio")
    ror:   Optional[MetricValue] = Field(None, description="Reporting Odds Ratio")
    chi2_yates: Optional[float]  = Field(None, description="Yates-corrected chi-squared")
    ic:    Optional[MetricValue] = Field(
        None,
        description="Information Component (log2 observed/expected, Bayesian shrinkage)"
    )

    # Status
    meets_threshold: bool = Field(
        False,
        description="True when a >= 3 AND all denominators are non-zero"
    )

    # Mandatory disclaimer — never omit from serialised output
    disclaimer: str = Field(
        default=(
            "POTENTIAL DISPROPORTIONATE REPORTING SIGNAL ONLY. "
            "Not a confirmed safety signal. Not evidence of causality. "
            "Requires clinical and regulatory expert review before any action."
        ),
        description="Regulatory disclaimer — must accompany every result",
    )

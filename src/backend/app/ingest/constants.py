"""
FAERS 2026 Q2 — verified column schema.

Source of truth: direct file inspection of the actual 2026 Q2 ASCII exports.
Do NOT modify these definitions without re-auditing the raw files.
"""

# ---------------------------------------------------------------------------
# Raw file names (quarter-stamped; update the suffix for new quarters)
# ---------------------------------------------------------------------------
QUARTER = "26Q2"

RAW_FILES = {
    "DEMO": f"DEMO{QUARTER}.txt",
    "DRUG": f"DRUG{QUARTER}.txt",
    "REAC": f"REAC{QUARTER}.txt",
    "OUTC": f"OUTC{QUARTER}.txt",
    "INDI": f"INDI{QUARTER}.txt",
    "THER": f"THER{QUARTER}.txt",
    "RPSR": f"RPSR{QUARTER}.txt",
}

# All files use '$' as the column delimiter and UTF-8 encoding.
DELIMITER = "$"

# ---------------------------------------------------------------------------
# Verified column lists (from header row of each file)
# ---------------------------------------------------------------------------
DEMO_COLS = [
    "primaryid", "caseid", "caseversion", "i_f_code", "event_dt",
    "mfr_dt", "init_fda_dt", "fda_dt", "rept_cod", "auth_num",
    "mfr_num", "mfr_sndr", "lit_ref", "age", "age_cod", "age_grp",
    "sex", "e_sub", "wt", "wt_cod", "rept_dt", "to_mfr", "occp_cod",
    "reporter_country", "occr_country",
]

DRUG_COLS = [
    "primaryid", "caseid", "drug_seq", "role_cod", "drugname",
    "prod_ai", "val_vbm", "route", "dose_vbm", "cum_dose_chr",
    "cum_dose_unit", "dechal", "rechal", "lot_num", "exp_dt",
    "nda_num", "dose_amt", "dose_unit", "dose_form", "dose_freq",
]

REAC_COLS = ["primaryid", "caseid", "pt", "drug_rec_act"]

OUTC_COLS = ["primaryid", "caseid", "outc_cod"]

INDI_COLS = ["primaryid", "caseid", "indi_drug_seq", "indi_pt"]

THER_COLS = [
    "primaryid", "caseid", "dsg_drug_seq",
    "start_dt", "end_dt", "dur", "dur_cod",
]

RPSR_COLS = ["primaryid", "caseid", "rpsr_cod"]

# ---------------------------------------------------------------------------
# Column splits for the DRUG → drug + drug_regimen normalisation
#
# Background:  (primaryid, drug_seq) is NOT unique in raw DRUG — the same
# drug identity can appear in multiple rows when dosing changed over time
# (e.g. different dose_amt, route, lot_num).  We call those extra rows
# "regimen episodes".
#
# drug        → one row per unique drug identity per case
#               PK: (primaryid, drug_seq)    [after dedup on identity cols]
#
# drug_regimen → one row per dosing episode
#                FK: (primaryid, drug_seq) → drug
#                Additional dedup: drop fully identical rows (1 923 in 26Q2)
# ---------------------------------------------------------------------------
DRUG_IDENTITY_COLS = [
    "primaryid", "caseid", "drug_seq", "role_cod", "drugname",
    "prod_ai", "val_vbm", "cum_dose_chr", "cum_dose_unit",
    "dechal", "rechal", "nda_num",
]

DRUG_REGIMEN_COLS = [
    "primaryid", "drug_seq",           # FK back to drug
    "route", "dose_vbm", "dose_amt", "dose_unit",
    "dose_form", "dose_freq", "lot_num", "exp_dt",
]

# ---------------------------------------------------------------------------
# Date column registry
# Observed formats in 26Q2: YYYYMMDD (len 8), YYYYMM (len 6), YYYY (len 4)
# Partial dates are flagged rather than corrected.
# ---------------------------------------------------------------------------
DATE_COLUMNS = {
    "DEMO":  ["event_dt", "mfr_dt", "init_fda_dt", "fda_dt", "rept_dt"],
    "DRUG":  ["exp_dt"],
    "THER":  ["start_dt", "end_dt"],
}

# Valid FAERS coded-value sets (for validation only; not enforced on ingest)
VALID_I_F_CODE   = {"I", "F"}
VALID_REPT_COD   = {"EXP", "PER", "DIR", "30DAY", "5DAY"}
VALID_AGE_COD    = {"YR", "DEC", "MON", "WK", "DY", "HR"}
VALID_SEX        = {"M", "F"}
VALID_WT_COD     = {"KG", "LBS"}
VALID_OCCP_COD   = {"MD", "PH", "CN", "LW", "HP"}
VALID_ROLE_COD   = {"PS", "SS", "C", "I", "DN"}
VALID_OUTC_COD   = {"DE", "LT", "HO", "DS", "RI", "CA", "OT"}
VALID_DUR_COD    = {"YR", "MON", "WK", "DY", "DAY", "HR", "MIN", "SEC"}
VALID_RPSR_COD   = {"CSM", "HP", "FGN"}

# Expected row counts for 2026 Q2 (used in smoke tests)
EXPECTED_RAW_COUNTS = {
    "DEMO": 422_459,
    "DRUG": 1_627_225,
    "REAC": 1_394_751,
    "OUTC": 303_705,
    "INDI": 1_124_333,
    "THER": 326_027,
    "RPSR": 11_295,
}

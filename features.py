"""
Feature engineering for VCBench founder success prediction.

Builds ~30 structured features from the raw JSON fields:
- Exit signals (prior IPOs, acquisitions)
- Education signals (degrees, QS ranking, fields)
- Career signals (roles, company sizes, durations)
- Industry signals
"""

import ast
import json
import numpy as np
import pandas as pd


# ---------- helpers ----------

def safe_parse(s):
    """Parse a JSON-like string to a Python list. Returns [] on failure or NaN."""
    if pd.isna(s) or s is None:
        return []
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s) if isinstance(s, str) else s
    except (ValueError, SyntaxError):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return []


def parse_qs_ranking(qs):
    """Convert QS ranking string to a numeric value (smaller = better).
    "1" -> 1, "1-50" -> 25, "200+" -> 250, "" -> None."""
    if not qs or pd.isna(qs):
        return None
    qs = str(qs).strip()
    if qs == "":
        return None
    if qs == "200+":
        return 250.0
    if "-" in qs:
        try:
            low, high = qs.split("-")
            return (int(low) + int(high)) / 2.0
        except (ValueError, TypeError):
            return None
    try:
        return float(qs)
    except ValueError:
        return None


def parse_duration(d):
    """Convert duration string to numeric years.
    "<2" -> 1, "2-4" -> 3, "4-5" -> 4.5, "5+" -> 6."""
    if not d or pd.isna(d):
        return 0.0
    d = str(d).strip()
    if d in ("", "nan"):
        return 0.0
    if d == "<2":
        return 1.0
    if d == "5+":
        return 6.0
    if "-" in d:
        try:
            low, high = d.split("-")
            return (float(low) + float(high)) / 2.0
        except (ValueError, TypeError):
            return 0.0
    try:
        return float(d)
    except ValueError:
        return 0.0


def parse_company_size(s):
    """Convert company size string to numeric (mid-bucket employee count)."""
    if not s or pd.isna(s):
        return 0
    s = str(s).strip().lower()
    mapping = {
        "myself only employees": 1,
        "myself only": 1,
        "2-10 employees": 6,
        "11-50 employees": 30,
        "51-200 employees": 125,
        "201-500 employees": 350,
        "501-1000 employees": 750,
        "1001-5000 employees": 3000,
        "5001-10000 employees": 7500,
        "10000+ employees": 15000,
        "10001+ employees": 15000,
    }
    return mapping.get(s, 0)


# C-level / leadership keywords
C_LEVEL_KEYWORDS = ["ceo", "cto", "cfo", "coo", "cmo", "cio", "cpo", "cso", "chief"]
FOUNDER_KEYWORDS = ["founder", "co-founder", "cofounder", "founding"]
LEADERSHIP_KEYWORDS = ["director", "vp", "vice president", "head of", "president"]
SENIOR_TECH_KEYWORDS = ["principal", "staff", "senior", "lead"]


def has_keyword(role, keywords):
    """Returns 1 if any keyword is in the role string."""
    if not role:
        return 0
    role_lower = str(role).lower()
    return int(any(kw in role_lower for kw in keywords))


# ---------- main feature builder ----------

def build_features(df):
    """Build all features from the raw VCBench dataframe.
    Returns a new dataframe with feature columns + 'success' target.
    """
    out = pd.DataFrame(index=df.index)
    out["success"] = df["success"].astype(int)

    # Pre-parse JSON columns once
    edus_list = df["educations_json"].apply(safe_parse)
    jobs_list = df["jobs_json"].apply(safe_parse)
    ipos_list = df["ipos"].apply(safe_parse)
    acq_list = df["acquisitions"].apply(safe_parse)

    # ===== TIER 1: EXIT SIGNALS (strongest predictors) =====
    out["n_prior_ipos"] = ipos_list.apply(len)
    out["n_prior_acquisitions"] = acq_list.apply(len)
    out["n_prior_exits"] = out["n_prior_ipos"] + out["n_prior_acquisitions"]
    out["has_prior_ipo"] = (out["n_prior_ipos"] > 0).astype(int)
    out["has_prior_acquisition"] = (out["n_prior_acquisitions"] > 0).astype(int)
    out["has_any_prior_exit"] = (out["n_prior_exits"] > 0).astype(int)
    out["has_multiple_exits"] = (out["n_prior_exits"] >= 2).astype(int)

    # Big IPO signal: any prior IPO with >$500M valuation
    def has_big_ipo(ipos):
        for ipo in ipos:
            val = str(ipo.get("valuation_usd", "")).lower()
            if ">500m" in val or ">1b" in val or ">5b" in val:
                return 1
        return 0
    out["has_big_prior_ipo"] = ipos_list.apply(has_big_ipo)

    # Well-known acquirer signal
    def has_well_known_acq(acqs):
        for acq in acqs:
            if acq.get("acquired_by_well_known", False):
                return 1
        return 0
    out["has_well_known_acquirer"] = acq_list.apply(has_well_known_acq)

    # ===== TIER 2: EDUCATION SIGNALS =====
    out["n_degrees"] = edus_list.apply(len)

    # Best (smallest) QS ranking
    def best_qs(edus):
        ranks = [parse_qs_ranking(e.get("qs_ranking", "")) for e in edus]
        ranks = [r for r in ranks if r is not None]
        return min(ranks) if ranks else 999.0  # 999 = no ranking / unknown
    out["best_qs_ranking"] = edus_list.apply(best_qs)

    out["has_top_10_qs"] = (out["best_qs_ranking"] <= 10).astype(int)
    out["has_top_50_qs"] = (out["best_qs_ranking"] <= 50).astype(int)
    out["has_top_200_qs"] = (out["best_qs_ranking"] <= 200).astype(int)

    # Degree types (any of multiple degrees)
    def has_degree_type(edus, kw):
        return int(any(kw.lower() in str(e.get("degree", "")).lower() for e in edus))
    out["has_phd"] = edus_list.apply(lambda e: has_degree_type(e, "phd")
                                     or has_degree_type(e, "doctor"))
    out["has_mba"] = edus_list.apply(lambda e: has_degree_type(e, "mba"))
    out["has_master"] = edus_list.apply(lambda e: has_degree_type(e, "ma")
                                        or has_degree_type(e, "ms")
                                        or has_degree_type(e, "msc")
                                        or has_degree_type(e, "master"))
    out["has_bachelor"] = edus_list.apply(lambda e: has_degree_type(e, "ba")
                                          or has_degree_type(e, "bs")
                                          or has_degree_type(e, "bsc")
                                          or has_degree_type(e, "bachelor"))

    # Field signals
    def has_field(edus, kws):
        for e in edus:
            field = str(e.get("field", "")).lower()
            if any(kw in field for kw in kws):
                return 1
        return 0
    out["has_cs_field"] = edus_list.apply(
        lambda e: has_field(e, ["computer", "software", "informatics"])
    )
    out["has_engineering_field"] = edus_list.apply(
        lambda e: has_field(e, ["engineering"])
    )
    out["has_business_field"] = edus_list.apply(
        lambda e: has_field(e, ["business", "management", "finance", "economic"])
    )
    out["has_stem_field"] = edus_list.apply(
        lambda e: has_field(e, ["math", "physics", "biology", "chemistry", "science",
                                "engineering", "computer"])
    )

    # ===== TIER 3: CAREER SIGNALS =====
    out["n_jobs"] = jobs_list.apply(len)

    # Total career years (sum of durations)
    def total_career(jobs):
        return sum(parse_duration(j.get("duration", "")) for j in jobs)
    out["total_career_years"] = jobs_list.apply(total_career)

    # Number of distinct industries worked in
    def unique_industries(jobs):
        inds = [str(j.get("industry", "")).strip() for j in jobs]
        inds = [i for i in inds if i]
        return len(set(inds))
    out["n_unique_industries_worked"] = jobs_list.apply(unique_industries)

    # Role-based features
    def any_role(jobs, keywords):
        return int(any(has_keyword(j.get("role", ""), keywords) for j in jobs))
    out["had_c_level_role"] = jobs_list.apply(lambda j: any_role(j, C_LEVEL_KEYWORDS))
    out["had_founder_role"] = jobs_list.apply(lambda j: any_role(j, FOUNDER_KEYWORDS))
    out["had_leadership_role"] = jobs_list.apply(
        lambda j: any_role(j, LEADERSHIP_KEYWORDS)
    )
    out["had_senior_tech_role"] = jobs_list.apply(
        lambda j: any_role(j, SENIOR_TECH_KEYWORDS)
    )

    # Company size features
    def max_company_size(jobs):
        sizes = [parse_company_size(j.get("company_size", "")) for j in jobs]
        sizes = [s for s in sizes if s > 0]
        return max(sizes) if sizes else 0
    out["max_company_size"] = jobs_list.apply(max_company_size)

    out["worked_at_big_company"] = (out["max_company_size"] >= 1000).astype(int)
    out["worked_at_huge_company"] = (out["max_company_size"] >= 5000).astype(int)

    # Worked at small (startup-like) company
    def n_startup_jobs(jobs):
        cnt = 0
        for j in jobs:
            sz = parse_company_size(j.get("company_size", ""))
            if 0 < sz <= 30:
                cnt += 1
        return cnt
    out["n_startup_jobs"] = jobs_list.apply(n_startup_jobs)

    # ===== TIER 4: INDUSTRY SIGNALS =====
    # Tech industries (the dominant cluster in VCBench)
    tech_industries = [
        "Software Development",
        "Technology, Information & Internet Platforms",
        "IT Services & Digital Solutions",
    ]
    out["industry_is_tech"] = df["industry"].isin(tech_industries).astype(int)

    # Healthcare/biotech
    health_industries = [
        "Biotechnology & Nanotechnology Research",
        "Clinical & Diagnostic Healthcare",
        "Wellness & Community Health",
        "Pharmaceutical Manufacturing",
    ]
    out["industry_is_health"] = df["industry"].isin(health_industries).astype(int)

    # Finance
    finance_industries = ["Financial Services"]
    out["industry_is_finance"] = df["industry"].isin(finance_industries).astype(int)

    # ===== TIER 5: COMPOSITE / INTERACTION SIGNALS =====
    # Career-industry alignment: did the founder previously work in the same industry?
    target_industry = df["industry"].astype(str)
    def industry_match(row_jobs, target):
        target_l = str(target).lower()
        for j in row_jobs:
            if target_l in str(j.get("industry", "")).lower():
                return 1
        return 0
    out["worked_in_target_industry"] = [
        industry_match(j, t) for j, t in zip(jobs_list, target_industry)
    ]

    # Career breadth: average industries per job (lower = more focused)
    out["career_breadth"] = np.where(
        out["n_jobs"] > 0,
        out["n_unique_industries_worked"] / out["n_jobs"].clip(lower=1),
        0.0,
    )

    # Average duration per job
    out["avg_job_duration"] = np.where(
        out["n_jobs"] > 0,
        out["total_career_years"] / out["n_jobs"].clip(lower=1),
        0.0,
    )

    # Strong combined signals (interactions)
    out["elite_education_x_leadership"] = out["has_top_50_qs"] * out["had_c_level_role"]
    out["serial_entrepreneur"] = out["had_founder_role"] * out["has_any_prior_exit"]
    out["technical_founder"] = out["has_cs_field"] * out["had_senior_tech_role"]

    return out


def feature_columns(df_features):
    """Returns the list of feature column names (everything except 'success')."""
    return [c for c in df_features.columns if c != "success"]

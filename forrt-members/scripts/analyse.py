"""Analysis functions for the FORRT members report.

No network access. Reads the CSVs written by fetch_forrt_works.py.
Imported by report.qmd so the same code produces the prose numbers and the figures.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.path.join(BASE, "data")

# Result 2 window: long enough to be informative, trimmed at both ends because a
# journal's current 2-year metric is a poor match for very old works and recent
# works have not accrued citations.
CORR_YEAR_MIN, CORR_YEAR_MAX = 2000, 2023
# Result 3 window: FORRT's own outputs only exist from 2019 on.
FORRT_YEAR_MIN = 2019

MIN_WORKS = 10
MIN_JOURNALS = 5
MIN_WORKS_STRICT = 20
MIN_JOURNALS_STRICT = 8

REQUIRED = [
    "contributors.csv",
    "works.csv",
    "author_works.csv",
    "sources.csv",
    "forrt_publications.csv",
    "orcid_normalization.csv",
    "fetch_meta.json",
]


def load() -> dict:
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA, f))]
    if missing:
        raise FileNotFoundError(
            "Missing data files: "
            + ", ".join(missing)
            + ". Run: py -3.12 forrt-members/scripts/fetch_forrt_works.py"
        )
    d = {
        name.replace(".csv", ""): pd.read_csv(os.path.join(DATA, name))
        for name in REQUIRED
        if name.endswith(".csv")
    }
    with open(os.path.join(DATA, "fetch_meta.json"), encoding="utf-8") as fh:
        d["meta"] = json.load(fh)
    return d


def build_frame(d: dict) -> pd.DataFrame:
    """Work-level frame with journal metrics attached. One row per unique work."""
    works = d["works"].copy()
    src = d["sources"][["source_id", "display_name", "type", "mean_citedness_2yr"]].rename(
        columns={"type": "src_type_lookup", "display_name": "journal_name"}
    )
    w = works.merge(src, on="source_id", how="left")
    w["cited_by_count"] = pd.to_numeric(w["cited_by_count"], errors="coerce")
    w["publication_year"] = pd.to_numeric(w["publication_year"], errors="coerce")
    w["mean_citedness_2yr"] = pd.to_numeric(w["mean_citedness_2yr"], errors="coerce")
    w["n_authors"] = pd.to_numeric(w["n_authors"], errors="coerce")
    return w


def attrition(w: pd.DataFrame, year_min: int, year_max: int | None) -> pd.DataFrame:
    """Stepwise filter counts, so the report can show what was dropped and why."""
    steps = []
    cur = w
    steps.append(("All unique works retrieved", len(cur)))
    cur = cur[cur["source_type"] == "journal"]
    steps.append(("In a journal source (excludes repositories, book platforms)", len(cur)))
    cur = cur[cur["mean_citedness_2yr"].notna()]
    steps.append(("Journal has a 2-year mean citedness", len(cur)))
    cur = cur[cur["type"] == "article"]
    steps.append(("Work type is article", len(cur)))
    cur = cur[cur["cited_by_count"].notna()]
    steps.append(("Citation count present", len(cur)))
    if year_max is None:
        cur = cur[cur["publication_year"] >= year_min]
        steps.append((f"Published {year_min} or later", len(cur)))
    else:
        cur = cur[
            (cur["publication_year"] >= year_min) & (cur["publication_year"] <= year_max)
        ]
        steps.append((f"Published {year_min} to {year_max}", len(cur)))
    return pd.DataFrame(steps, columns=["Step", "Works remaining"])


def eligible(w: pd.DataFrame, year_min: int, year_max: int | None) -> pd.DataFrame:
    m = (
        (w["source_type"] == "journal")
        & w["mean_citedness_2yr"].notna()
        & (w["type"] == "article")
        & w["cited_by_count"].notna()
        & (w["publication_year"] >= year_min)
    )
    if year_max is not None:
        m = m & (w["publication_year"] <= year_max)
    return w[m].copy()


# ---------------------------------------------------------------- result 2

def author_correlations(
    elig: pd.DataFrame,
    author_works: pd.DataFrame,
    min_works: int = MIN_WORKS,
    min_journals: int = MIN_JOURNALS,
) -> pd.DataFrame:
    """Per-researcher Spearman rho between journal mean citedness and own citations.

    A work shared by several FORRT members contributes to each of their
    correlations, which is why this joins through author_works rather than using
    the deduplicated work table directly.
    """
    aw = author_works.merge(
        elig[["work_id", "cited_by_count", "mean_citedness_2yr", "source_id"]],
        on="work_id",
        how="inner",
    )
    out = []
    for pid, g in aw.groupby("person_id"):
        n_w = len(g)
        n_j = g["source_id"].nunique()
        if n_w < min_works or n_j < min_journals:
            continue
        if g["mean_citedness_2yr"].nunique() < 2 or g["cited_by_count"].nunique() < 2:
            out.append({"person_id": pid, "n_works": n_w, "n_journals": n_j, "rho": np.nan})
            continue
        rho, _ = stats.spearmanr(g["mean_citedness_2yr"], g["cited_by_count"])
        out.append({"person_id": pid, "n_works": n_w, "n_journals": n_j, "rho": rho})
    return pd.DataFrame(out)


def summarise_rho(corr: pd.DataFrame) -> dict:
    r = corr["rho"].dropna()
    if len(r) == 0:
        return {"n": 0}
    pos = int((r > 0).sum())
    binom = stats.binomtest(pos, len(r), 0.5)
    return {
        "n": len(r),
        "n_undefined": int(corr["rho"].isna().sum()),
        "median": float(r.median()),
        "q1": float(r.quantile(0.25)),
        "q3": float(r.quantile(0.75)),
        "prop_positive": pos / len(r),
        "n_positive": pos,
        "sign_p": float(binom.pvalue),
    }


def plot_rho_hist(corr: pd.DataFrame, ax) -> None:
    r = corr["rho"].dropna()
    bins = np.arange(-1.0, 1.0001, 0.1)
    ax.hist(r, bins=bins, color="#4878a8", edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.axvline(
        r.median(),
        color="#c0504d",
        linewidth=1.6,
        label=f"median = {r.median():.2f}",
    )
    ax.set_xlabel("Spearman rho within a researcher's own works")
    ax.set_ylabel("Number of researchers")
    ax.set_xlim(-1, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------- result 3

def paired_forrt(elig19: pd.DataFrame, author_works: pd.DataFrame) -> pd.DataFrame:
    """Within-researcher contrast: own FORRT-listed works vs own other works."""
    aw = author_works.merge(
        elig19[["work_id", "cited_by_count", "forrt_listed"]], on="work_id", how="inner"
    )
    rows = []
    for pid, g in aw.groupby("person_id"):
        f = g[g["forrt_listed"] == 1]["cited_by_count"]
        o = g[g["forrt_listed"] == 0]["cited_by_count"]
        if len(f) == 0 or len(o) == 0:
            continue
        rows.append(
            {
                "person_id": pid,
                "n_forrt": len(f),
                "n_other": len(o),
                "median_forrt": float(f.median()),
                "median_other": float(o.median()),
                "log_ratio": float(np.log((f.median() + 1) / (o.median() + 1))),
            }
        )
    return pd.DataFrame(rows)


def summarise_paired(p: pd.DataFrame, n_boot: int = 5000, seed: int = 7) -> dict:
    if len(p) < 3:
        return {"n": len(p)}
    lr = p["log_ratio"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = [
        np.median(rng.choice(lr, size=len(lr), replace=True)) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        w = stats.wilcoxon(lr, zero_method="wilcox")
        wp, wstat = float(w.pvalue), float(w.statistic)
    except ValueError:
        wp, wstat = float("nan"), float("nan")
    return {
        "n": len(p),
        "median_log_ratio": float(np.median(lr)),
        "ratio": float(np.exp(np.median(lr))),
        "ratio_lo": float(np.exp(lo)),
        "ratio_hi": float(np.exp(hi)),
        "prop_higher": float((lr > 0).mean()),
        "n_higher": int((lr > 0).sum()),
        "wilcoxon_p": wp,
        "wilcoxon_stat": wstat,
    }


def plot_paired(p: pd.DataFrame, axes) -> None:
    ax1, ax2 = axes
    for _, r in p.iterrows():
        ax1.plot(
            [0, 1],
            [r["median_other"] + 1, r["median_forrt"] + 1],
            color="#4878a8",
            alpha=0.18,
            linewidth=0.9,
        )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Own other works", "FORRT-listed works"])
    ax1.set_yscale("log")
    ax1.set_ylabel("Median citations + 1 (log scale)")
    ax1.set_xlim(-0.25, 1.25)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.boxplot(
        p["log_ratio"], vert=True, widths=0.5, patch_artist=True,
        boxprops={"facecolor": "#dce6f0", "edgecolor": "#4878a8"},
        medianprops={"color": "#c0504d", "linewidth": 1.6},
    )
    ax2.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax2.set_xticks([])
    ax2.set_ylabel("log ratio (FORRT vs own other works)")
    ax2.spines[["top", "right", "bottom"]].set_visible(False)


def model_frame(elig19: pd.DataFrame, author_works: pd.DataFrame, sample_ids) -> pd.DataFrame:
    """Unique works belonging to the paired-sample contributors."""
    ids = set(sample_ids)
    wids = set(author_works[author_works["person_id"].isin(ids)]["work_id"])
    m = elig19[elig19["work_id"].isin(wids)].copy()
    m = m[m["n_authors"] > 0]
    snapshot = int(m["publication_year"].max())
    m["age_years"] = snapshot - m["publication_year"] + 1
    m["log_age"] = np.log(m["age_years"].clip(lower=1))
    m["log_n_authors"] = np.log(m["n_authors"])
    m["log_mc"] = np.log1p(m["mean_citedness_2yr"])
    m["forrt_listed"] = m["forrt_listed"].astype(float)
    return m


def fit_models(m: pd.DataFrame) -> dict:
    """Poisson dispersion check, then negative binomial with cluster-robust SEs."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    terms_full = ["forrt_listed", "log_n_authors", "log_mc", "log_age"]
    terms_noauth = ["forrt_listed", "log_mc", "log_age"]
    y = m["cited_by_count"].astype(float)
    groups = m["source_id"].astype(str)

    def fit(terms):
        X = sm.add_constant(m[terms])
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        dispersion = float(pois.pearson_chi2 / pois.df_resid)
        try:
            nb_mle = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=200)
            alpha = float(nb_mle.params.get("alpha", 1.0))
            converged = bool(nb_mle.mle_retvals.get("converged", False))
        except Exception:  # noqa: BLE001
            alpha, converged = 1.0, False
        alpha = min(max(alpha, 1e-4), 50.0)
        res = sm.GLM(
            y, X, family=sm.families.NegativeBinomial(alpha=alpha)
        ).fit(cov_type="cluster", cov_kwds={"groups": groups})
        ci = res.conf_int()
        tab = pd.DataFrame(
            {
                "term": res.params.index,
                "irr": np.exp(res.params.to_numpy()),
                "lo": np.exp(ci[0].to_numpy()),
                "hi": np.exp(ci[1].to_numpy()),
                "p": res.pvalues.to_numpy(),
            }
        )
        return {
            "table": tab,
            "dispersion": dispersion,
            "alpha": alpha,
            "converged": converged,
            "n": int(len(m)),
            "n_clusters": int(groups.nunique()),
            "res": res,
        }

    X_vif = sm.add_constant(m[terms_full])
    vif = pd.DataFrame(
        {
            "term": X_vif.columns,
            "vif": [variance_inflation_factor(X_vif.to_numpy(), i) for i in range(X_vif.shape[1])],
        }
    )
    corr_fa = float(
        np.corrcoef(m["forrt_listed"].to_numpy(), m["log_n_authors"].to_numpy())[0, 1]
    )

    out = {
        "full": fit(terms_full),
        "no_authors": fit(terms_noauth),
        "vif": vif,
        "corr_forrt_authors": corr_fa,
        "n_forrt_works": int(m["forrt_listed"].sum()),
    }
    cens = m[m["n_authors_censored"] == 0]
    if len(cens) > 50 and cens["forrt_listed"].sum() > 0:
        y2 = cens["cited_by_count"].astype(float)
        X2 = sm.add_constant(cens[terms_full])
        try:
            nb2 = sm.NegativeBinomial(y2, X2).fit(disp=False, maxiter=200)
            a2 = min(max(float(nb2.params.get("alpha", 1.0)), 1e-4), 50.0)
            r2 = sm.GLM(y2, X2, family=sm.families.NegativeBinomial(alpha=a2)).fit(
                cov_type="cluster", cov_kwds={"groups": cens["source_id"].astype(str)}
            )
            out["uncensored_irr_forrt"] = float(np.exp(r2.params["forrt_listed"]))
            out["uncensored_n"] = int(len(cens))
        except Exception:  # noqa: BLE001
            pass
    return out


LABELS = {
    "const": "Intercept",
    "forrt_listed": "On FORRT publication list",
    "log_n_authors": "Number of contributors (log)",
    "log_mc": "Journal mean citedness (log)",
    "log_age": "Years since publication (log)",
}


def irr_table(fit: dict) -> pd.DataFrame:
    t = fit["table"].copy()
    t["Predictor"] = t["term"].map(lambda x: LABELS.get(x, x))
    t["IRR (95% CI)"] = t.apply(
        lambda r: f"{r['irr']:.2f} ({r['lo']:.2f} to {r['hi']:.2f})", axis=1
    )
    t["p"] = t["p"].map(lambda v: "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0"))
    return t[["Predictor", "IRR (95% CI)", "p"]]


def fmt_p(v: float) -> str:
    if v != v:
        return "not defined"
    if v < 0.001:
        return "p < .001"
    return "p = " + f"{v:.3f}".lstrip("0")

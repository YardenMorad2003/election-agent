"""
Baseline forecast: 2026 U.S. House midterm seat change for the President's party.

Trains an OLS model on 12 midterm cycles (1978-2022) using:
  - Net presidential approval at midterm (Gallup-style approve - disapprove, late Sept-Oct of midterm year)
  - CPI year-over-year inflation reading at midterm
  - Civilian unemployment rate at midterm

Targets the President's-party net House seat change vs. the prior cycle. Compares
three candidate specifications (approval-only, macro-only, combined) by
leave-one-out cross-validation, then refits the winner on the full sample and
emits a 2026 forecast with a 95% prediction interval.

Run:  python predict_house.py
Output: forecast_2026_house.json + console summary

Notes
-----
- Pres approval is averaged over Sept-Oct of the midterm year from three
  source files:
    * 1978-2014: data/macro/approval/pres_approval_data.csv (Carter through Obama, individual polls)
    * 2018: Prediction_Data/Approval_Data/trump1_approval_polls_538.csv (538 raw polls; archived
      from `simonw/fivethirtyeight-polls` since 538's projects URL was retired
      when Disney shut down 538 in 2024)
    * 2022: Prediction_Data/Approval_Data/biden_approval_topline_538.csv (538 daily smoothed
      topline; archived from `stiles/biden-polls`, same schema as the
      `trump_approval_raw.csv` already in this repo)
- Sample size is small (n=12). We deliberately keep models to 1-3 features and
  rely on LOOCV rather than a held-out split.
- Fusion-ticket votes (e.g. NY DEM + WORKING FAMILIES on the same candidate) are
  summed per (state, district, candidate) before picking the district winner.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parent
HOUSE = ROOT / "data/elections_extra/house_elections.csv"
CHAMBER = ROOT / "data/elections_extra/chamber_control.csv"
APPROVAL = ROOT / "data/macro/approval/pres_approval_data.csv"
TRUMP1_APPROVAL = ROOT / "Prediction_Data/Approval_Data/trump1_approval_polls_538.csv"
BIDEN_APPROVAL = ROOT / "Prediction_Data/Approval_Data/biden_approval_topline_538.csv"
TRUMP_APPROVAL = ROOT / "data/macro/approval/trump_approval_raw.csv"
GENERIC_BALLOT_SB = ROOT / "Prediction_Data/silverbulletin_generic_ballot_2025_2026.csv"
CPI = ROOT / "data/macro/fred/fred_cpi.csv"
UNRATE = ROOT / "data/macro/fred/fred_unrate.csv"
GAS = ROOT / "data/macro/gas_prices.csv"
RETIREMENTS = ROOT / "data/elections_extra/house_retirements_features.csv"
OUT = ROOT / "forecast_2026_house.json"

MIDTERMS = [1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
FORECAST_YEAR = 2026

# Generic ballot is recorded as a reference number, not a model feature.
# `data/macro/generic_topline_historical.csv` (538) only covers 1995-2016 (5
# of our 12 midterms), so coverage is too shallow to enter the training set.
# The current value is read live from Silver Bulletin's 2025-2026 daily
# topline (`silverbulletin_generic_ballot_2025_2026.csv`).
GENERIC_BALLOT_SOURCE = (
    "Silver Bulletin (Nate Silver) — 2026 generic congressional ballot average"
)
GENERIC_BALLOT_SOURCE_URL = (
    "https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls"
)


def latest_generic_ballot_reference() -> dict:
    """Return the latest 7-day-average D-R margin from Silver Bulletin, plus
    the asof date and source. Reference value only — not a model feature.
    Coverage in `generic_topline_historical.csv` (1995-2016) is too shallow
    to use as a training feature for the 1978-2022 sample."""
    df = pd.read_csv(GENERIC_BALLOT_SB)
    df["modeldate"] = pd.to_datetime(df["modeldate"])
    df = df.sort_values("modeldate")
    last7 = df.tail(7)
    margin_today = float(df.iloc[-1]["dem"] - df.iloc[-1]["rep"])
    margin_7d = float((last7["dem"] - last7["rep"]).mean())
    return {
        "margin_d_minus_r_today": round(margin_today, 2),
        "margin_d_minus_r_7d_avg": round(margin_7d, 2),
        "asof": df["modeldate"].max().strftime("%Y-%m-%d"),
        "rows_used_in_7d_avg": int(len(last7)),
        "source": GENERIC_BALLOT_SOURCE,
        "source_url": GENERIC_BALLOT_SOURCE_URL,
        "note": "Reference value, not a model feature. Historical generic-ballot coverage in this repo (1995-2016) is too shallow for the 1978-2022 training sample.",
    }


# ── Y: President's-party seat change ──────────────────────────────────────

def winners_per_year(house: pd.DataFrame, year: int) -> pd.Series:
    """Count House district winners by party for a given general election year.

    Aggregates fusion-ticket votes (e.g. NY DEM + WORKING FAMILIES rows for the
    same candidate) by summing candidatevotes per (state, district, candidate)
    before picking the district winner.
    """
    sub = house[
        (house["year"] == year)
        & (house["stage"] == "GEN")
        & (~house["special"])
        & (~house["writein"].fillna(False))
    ]
    agg = sub.groupby(["state_po", "district", "candidate"], as_index=False).agg(
        candidatevotes=("candidatevotes", "sum"),
        primary_party=("party", "first"),
    )
    # Prefer DEMOCRAT/REPUBLICAN label when a fusion candidate has one
    dr = sub[sub["party"].isin(["DEMOCRAT", "REPUBLICAN"])][
        ["state_po", "district", "candidate", "party"]
    ].drop_duplicates(["state_po", "district", "candidate"])
    merged = agg.merge(dr, on=["state_po", "district", "candidate"], how="left")
    merged["party"] = merged["party"].fillna(merged["primary_party"])
    winners = merged.loc[merged.groupby(["state_po", "district"])["candidatevotes"].idxmax()]
    return winners["party"].value_counts()


def build_target(house: pd.DataFrame, chamber: pd.DataFrame) -> pd.DataFrame:
    """Compute President's-party net seat change at each midterm cycle."""
    seats = {}
    for y in sorted(set(MIDTERMS) | {y - 2 for y in MIDTERMS}):
        c = winners_per_year(house, y)
        seats[y] = {"D": int(c.get("DEMOCRAT", 0)), "R": int(c.get("REPUBLICAN", 0))}

    rows = []
    for y in MIDTERMS:
        prev = y - 2
        pres_dem = bool(
            chamber.loc[chamber["year"] == y, "dem_presidency"].iloc[0]
        )
        pres_party = "D" if pres_dem else "R"
        change = seats[y][pres_party] - seats[prev][pres_party]
        rows.append({
            "year": y,
            "pres_party": pres_party,
            f"{pres_party}_seats_t": seats[y][pres_party],
            f"{pres_party}_seats_t_minus_2": seats[prev][pres_party],
            "pres_party_seats_t_minus_2": seats[prev][pres_party],
            "pres_party_seat_change": change,
        })
    return pd.DataFrame(rows)


# ── X: features ──────────────────────────────────────────────────────────

def net_approval_at_midterm(approval: pd.DataFrame, year: int) -> float:
    """Average net approval (approve - disapprove) over Sept-Oct of `year`.

    Routes to the appropriate source file:
      * 2018 -> Trump 1 raw polls (538 archive via simonw/fivethirtyeight-polls)
      * 2022 -> Biden daily topline (538 archive via stiles/biden-polls)
      * else -> pres_approval_data.csv (Carter through Obama, individual polls)
    """
    if year == 2018:
        t = pd.read_csv(TRUMP1_APPROVAL, low_memory=False)
        t["end_date"] = pd.to_datetime(t["end_date"], errors="coerce")
        w = t[(t["end_date"].dt.year == year) & (t["end_date"].dt.month.isin([9, 10]))]
        if w.empty:
            raise ValueError(f"No Trump 1 approval polls for {year}")
        return float((w["yes"] - w["no"]).mean())
    if year == 2022:
        b = pd.read_csv(BIDEN_APPROVAL)
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        w = b[(b["date"].dt.year == year) & (b["date"].dt.month.isin([9, 10]))]
        if w.empty:
            raise ValueError(f"No Biden topline rows for {year}")
        return float((w["approve"] - w["disapprove"]).mean())

    a = approval.copy()
    a["start"] = pd.to_datetime(a["Start Date"], errors="coerce")
    window = a[(a["start"].dt.year == year) & (a["start"].dt.month.isin([9, 10]))]
    if window.empty:
        window = a[(a["start"].dt.year == year) & (a["start"].dt.month.isin([8, 9, 10]))]
    if window.empty:
        raise ValueError(f"No approval polls found for {year}")
    return float((window["Approving"] - window["Disapproving"]).mean())


def cpi_yoy_at_year(cpi: pd.DataFrame, year: int) -> float:
    """CPI YoY inflation reading for `year` (annual end-of-year value)."""
    c = cpi.copy()
    c["date"] = pd.to_datetime(c["date"])
    c["y"] = c["date"].dt.year
    sub = c[c["y"] == year]
    if sub.empty:
        raise ValueError(f"No CPI reading for {year}")
    return float(sub["cpi"].iloc[-1])


def pres_party_retire_pct(retire: pd.DataFrame, year: int, pres_party: str) -> float | None:
    """Share of midterm-year House retirements coming from the president's
    party (e.g. 0.67 for 2018 = 37 R retirements / 55 total). Higher = worse
    for the president's party. Returns None if the retirements file does not
    cover `year` (it begins in 1996)."""
    sub = retire[retire["year"] == year]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return float(row["retirement_rate_D"] if pres_party == "D" else row["retirement_rate_R"])


def unrate_at_year(unrate: pd.DataFrame, year: int, month: int = 10) -> float:
    """Civilian unemployment rate at October of `year` (closest available)."""
    u = unrate.copy()
    u["date"] = pd.to_datetime(u["date"])
    target = pd.Timestamp(year=year, month=month, day=1)
    same_year = u[u["date"].dt.year == year]
    if not same_year.empty:
        idx = (same_year["date"] - target).abs().idxmin()
        return float(same_year.loc[idx, "unrate"])
    # fall back to latest available before target
    return float(u[u["date"] <= target].iloc[-1]["unrate"])


# ── Forecast inputs for 2026 ─────────────────────────────────────────────

def forecast_inputs() -> dict:
    """Pull the latest 2026 inputs: Trump net approval, latest CPI, latest
    unrate, and (informational) latest national gas price."""
    ta = pd.read_csv(TRUMP_APPROVAL)
    ta["date"] = pd.to_datetime(ta["date"])
    last30 = ta.sort_values("date").tail(30)
    net = float((last30["approve"] - last30["disapprove"]).mean())
    asof_approval = ta["date"].max()

    cpi = pd.read_csv(CPI)
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi_latest = cpi.iloc[-1]

    unrate = pd.read_csv(UNRATE)
    unrate["date"] = pd.to_datetime(unrate["date"])
    unrate_latest = unrate.iloc[-1]

    gas = pd.read_csv(GAS)
    gas["date"] = pd.to_datetime(gas["date"])
    gas_latest = gas.iloc[-1]

    return {
        "net_approval": round(net, 2),
        "approval_asof": asof_approval.strftime("%Y-%m-%d"),
        "cpi_yoy": round(float(cpi_latest["cpi"]), 2),
        "cpi_asof": cpi_latest["date"].strftime("%Y-%m-%d"),
        "unrate": round(float(unrate_latest["unrate"]), 2),
        "unrate_asof": unrate_latest["date"].strftime("%Y-%m-%d"),
        "gas_price": round(float(gas_latest["gas_price"]), 3),
        "gas_asof": gas_latest["date"].strftime("%Y-%m-%d"),
    }


# ── Modeling ─────────────────────────────────────────────────────────────

# SPECS values are (feature_cols, required_non_null_cols). The second list is
# what fit_all_specs uses to subset the training frame — it lets specs that
# don't literally include a feature still inherit its NaN-pattern so they
# train on the same subset (used for apples-to-apples comparisons).
SPECS: dict[str, tuple[list[str], list[str]]] = {
    "approval_only":          (["net_approval"],                              ["net_approval"]),
    "macro_only":             (["cpi_yoy", "unrate"],                         ["cpi_yoy", "unrate"]),
    "approval_plus_macro":    (["net_approval", "cpi_yoy", "unrate"],         ["net_approval", "cpi_yoy", "unrate"]),
    "approval_plus_exposure": (["net_approval", "pres_party_seats_t_minus_2"], ["net_approval", "pres_party_seats_t_minus_2"]),
    "exposure_only":          (["pres_party_seats_t_minus_2"],                ["pres_party_seats_t_minus_2"]),
    # Retirement specs use the post-1996 subset (n=7). approval_only_post96
    # uses ONLY net_approval as a feature but also requires retire_pct to be
    # non-null, so it trains on the same n=7 subset as the retire specs for
    # apples-to-apples LOOCV.
    "approval_only_post96":   (["net_approval"],                              ["net_approval", "pres_party_retire_pct"]),
    "retire_only":            (["pres_party_retire_pct"],                     ["pres_party_retire_pct"]),
    "approval_plus_retire":   (["net_approval", "pres_party_retire_pct"],     ["net_approval", "pres_party_retire_pct"]),
}


def _new_estimator(estimator: str):
    """Estimator factory. 'ols' -> LinearRegression; 'huber' -> HuberRegressor
    with epsilon=1.35 (the sklearn default, ~95% efficiency under normal
    errors, robust to leverage from a few high-residual midterms like 1994
    and 2010)."""
    if estimator == "ols":
        return LinearRegression()
    if estimator == "huber":
        return HuberRegressor(epsilon=1.35, max_iter=500)
    raise ValueError(f"unknown estimator: {estimator}")


def loocv_mae(X: np.ndarray, y: np.ndarray, estimator: str = "ols") -> float:
    loo = LeaveOneOut()
    preds, truths = [], []
    for train_idx, test_idx in loo.split(X):
        m = _new_estimator(estimator).fit(X[train_idx], y[train_idx])
        preds.append(m.predict(X[test_idx])[0])
        truths.append(y[test_idx][0])
    return float(mean_absolute_error(truths, preds))


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42


def prediction_interval(model, X_train: np.ndarray, y_train: np.ndarray,
                         x_new: np.ndarray, alpha: float = 0.05,
                         n_boot: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                         estimator: str = "ols"
                         ) -> tuple[float, float, float]:
    """Point + 95% prediction interval via residual bootstrap.

    Refits the estimator on (X_train, y_train + resampled_residuals) `n_boot`
    times to propagate coefficient uncertainty, then adds an independent
    residual draw to each bootstrap prediction to make it a PI (not a CI) on
    the new point. With n=12 the textbook z=1.96 formula understates tail
    risk, so we take empirical 2.5 / 97.5 percentiles instead.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    in_sample = model.predict(X_train)
    resid = y_train - in_sample
    preds = []
    for _ in range(n_boot):
        boot_resid = rng.choice(resid, size=n, replace=True)
        y_boot = in_sample + boot_resid
        try:
            m_boot = _new_estimator(estimator).fit(X_train, y_boot)
        except ValueError:
            # HuberRegressor occasionally fails to converge on a particular
            # bootstrap draw with small n. Skip that draw rather than fall
            # back to OLS (which would mix estimators and bias the PI).
            continue
        new_error = rng.choice(resid)
        preds.append(float(m_boot.predict(x_new[None, :])[0]) + new_error)
    if len(preds) < n_boot * 0.5:
        raise RuntimeError(
            f"Bootstrap PI: only {len(preds)}/{n_boot} draws converged for "
            f"estimator={estimator}; results unreliable"
        )
    arr = np.asarray(preds)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    point = float(model.predict(x_new[None, :])[0])
    return point, lo, hi


def fit_all_specs(df: pd.DataFrame, estimator: str = "ols"):
    """Fit each candidate spec. Per-spec: drop rows where any required
    feature column is NaN before fitting (so retirement specs train on the
    post-1996 subset). Returns {name: {model, X, y, cols, n_train, loocv_mae}}.
    """
    fitted = {}
    for name, (cols, required) in SPECS.items():
        sub = df.dropna(subset=list(set(required) | {"pres_party_seat_change"}))
        X = sub[cols].to_numpy()
        y = sub["pres_party_seat_change"].to_numpy()
        mae = loocv_mae(X, y, estimator=estimator)
        m = _new_estimator(estimator).fit(X, y)
        fitted[name] = {
            "model": m, "X": X, "y": y, "cols": cols,
            "n_train": int(len(sub)), "loocv_mae": mae,
            "estimator": estimator,
        }
    return fitted


def predict_with_inputs(fitted: dict, spec_name: str, feature_inputs: dict):
    """Run a single-row prediction for `spec_name` using `feature_inputs`."""
    item = fitted[spec_name]
    x_new = np.array([feature_inputs[c] for c in item["cols"]])
    point, lo, hi = prediction_interval(
        item["model"], item["X"], item["y"], x_new, estimator=item["estimator"]
    )
    return point, lo, hi


def apply_deltas(base_inputs: dict, deltas: dict) -> dict:
    """Return a copy of `base_inputs` with `deltas` (key -> additive delta) applied."""
    out = dict(base_inputs)
    for k, dv in deltas.items():
        if dv is None:
            continue
        if k not in out:
            continue
        out[k] = round(float(out[k]) + dv, 4)
    return out


def print_forecast_block(label: str, inputs: dict, fitted: dict,
                          best_name: str, show_all_specs: bool):
    """Print one forecast section (baseline OR scenario)."""
    feat = {
        "net_approval": inputs["net_approval"],
        "cpi_yoy": inputs["cpi_yoy"],
        "unrate": inputs["unrate"],
        "pres_party_seats_t_minus_2": inputs["pres_party_seats_t_minus_2"],
        "pres_party_retire_pct": inputs["pres_party_retire_pct"],
    }
    print(f"\n=== {label} ===")
    print(f"Inputs: net_approval={inputs['net_approval']:+.2f}  "
          f"cpi_yoy={inputs['cpi_yoy']:.2f}  unrate={inputs['unrate']:.2f}  "
          f"seats_t_minus_2={inputs['pres_party_seats_t_minus_2']}  "
          f"retire_pct={inputs['pres_party_retire_pct']:.2f}  "
          f"gas_price=${inputs.get('gas_price', 0):.3f} (informational)")
    point, lo, hi = predict_with_inputs(fitted, best_name, feat)
    print(f"Selected spec ({best_name}): R net seat change = {point:+.1f}   "
          f"95% PI [{lo:+.1f}, {hi:+.1f}]")
    if show_all_specs:
        print("All specs (for context):")
        for nm in fitted:
            p, l, h = predict_with_inputs(fitted, nm, feat)
            print(f"  {nm:<28} {p:+7.1f}  [{l:+.1f}, {h:+.1f}]")
    return point, lo, hi


def main(args):
    house = pd.read_csv(HOUSE, low_memory=False)
    chamber = pd.read_csv(CHAMBER)
    approval = pd.read_csv(APPROVAL)
    cpi = pd.read_csv(CPI)
    unrate = pd.read_csv(UNRATE)
    retire = pd.read_csv(RETIREMENTS)

    target = build_target(house, chamber)
    pres_party_by_year = dict(zip(target["year"], target["pres_party"]))
    feat_rows = []
    for y in MIDTERMS:
        feat_rows.append({
            "year": y,
            "net_approval": net_approval_at_midterm(approval, y),
            "cpi_yoy": cpi_yoy_at_year(cpi, y),
            "unrate": unrate_at_year(unrate, y),
            "pres_party_retire_pct": pres_party_retire_pct(
                retire, y, pres_party_by_year[y]
            ),  # None pre-1996 -> NaN in df
        })
    df = target.merge(pd.DataFrame(feat_rows), on="year")

    # Seat exposure for 2026: president's-party (R) seats coming out of the
    # 2024 election. Computed with the same winners_per_year logic used for
    # the training target so the feature is internally consistent.
    seats_2024 = winners_per_year(house, FORECAST_YEAR - 2)
    pres_party_2026 = "R"  # Trump (R) is in office for the 2026 midterm
    pres_party_seats_2024 = int(
        seats_2024.get("REPUBLICAN" if pres_party_2026 == "R" else "DEMOCRAT", 0)
    )
    retire_2026 = pres_party_retire_pct(retire, FORECAST_YEAR, pres_party_2026)

    if not args.scenario_only:
        print("Training frame (n=%d):" % len(df))
        print(df.to_string(index=False))
        print()

    # Fit every spec with both OLS and Huber, then merge into a single dict
    # keyed by "<spec>__<estimator>" so we can compare side-by-side.
    fitted_ols = fit_all_specs(df, estimator="ols")
    fitted_huber = fit_all_specs(df, estimator="huber")
    fitted = {f"{n}__ols": v for n, v in fitted_ols.items()}
    fitted.update({f"{n}__huber": v for n, v in fitted_huber.items()})

    if not args.scenario_only:
        print(f"{'spec':<30} {'est':<6} {'n':>3}  {'LOOCV MAE':>9}")
        for name, item in fitted.items():
            base = name.rsplit("__", 1)[0]
            est = item["estimator"]
            print(f"  {base:<28} {est:<6} {item['n_train']:>3}  {item['loocv_mae']:>6.2f}")
    best_name = min(fitted, key=lambda k: fitted[k]["loocv_mae"])
    if not args.scenario_only:
        bm = fitted[best_name]
        print(f"\nSelected: {best_name}  features={bm['cols']}  n={bm['n_train']}")
        coefs = dict(zip(bm["cols"], [round(float(c), 3) for c in bm["model"].coef_]))
        print(f"Coefficients: {coefs}")
        print(f"Intercept:    {round(float(bm['model'].intercept_), 3)}")

    inputs = forecast_inputs()
    inputs["pres_party_seats_t_minus_2"] = pres_party_seats_2024
    inputs["pres_party_retire_pct"] = retire_2026

    # Baseline
    point, lo, hi = print_forecast_block(
        "Baseline (current 2026 inputs)", inputs, fitted, best_name, args.all_specs
    )

    # Scenario
    deltas = {
        "net_approval": args.approval_delta,
        "cpi_yoy": args.cpi_delta,
        "unrate": args.unrate_delta,
    }
    has_real_delta = any(v is not None for v in deltas.values())
    has_gas_delta = args.gas_delta is not None
    if has_real_delta or has_gas_delta:
        scen_inputs = apply_deltas(inputs, deltas)
        if has_gas_delta:
            scen_inputs["gas_price"] = round(scen_inputs["gas_price"] + args.gas_delta, 3)
        label_parts = []
        if args.approval_delta is not None:
            label_parts.append(f"approval delta {args.approval_delta:+g}")
        if args.cpi_delta is not None:
            label_parts.append(f"cpi delta {args.cpi_delta:+g}")
        if args.unrate_delta is not None:
            label_parts.append(f"unrate delta {args.unrate_delta:+g}")
        if has_gas_delta:
            label_parts.append(f"gas delta ${args.gas_delta:+.2f}")
        s_point, s_lo, s_hi = print_forecast_block(
            "Scenario: " + ", ".join(label_parts),
            scen_inputs, fitted, best_name, args.all_specs,
        )
        print(f"\nDelta vs baseline (selected spec): {s_point - point:+.1f} seats")
        if has_gas_delta and not has_real_delta:
            print("\nNote: gas_price is NOT a feature in any candidate spec.")
            print("Changing gas alone does not move the prediction from this model.")
            print("In approval-only fundamentals models, gas effects are absorbed by the approval coefficient.")
        elif has_gas_delta:
            print("\nNote: gas_price is informational only; it is not a feature in the model.")
            print("The delta above comes entirely from the non-gas overrides.")

    historical_mean = float(fitted[best_name]["y"].mean())
    print(f"\nHistorical midterm-penalty baseline:  {historical_mean:+.1f}  "
          f"(over winner's n={fitted[best_name]['n_train']} training set)")

    gb = latest_generic_ballot_reference()
    print(f"\nGeneric ballot reference (not used as feature):")
    print(f"  D-R margin (7d avg through {gb['asof']}): {gb['margin_d_minus_r_7d_avg']:+.2f}")
    print(f"  D-R margin (latest day): {gb['margin_d_minus_r_today']:+.2f}")
    implied = -5 * gb["margin_d_minus_r_7d_avg"]
    print(f"  Heuristic implied R seat change (5 seats/pt): {implied:+.0f}  vs model: {point:+.1f}")

    if not args.no_save:
        spec_results = {n: {"features": item["cols"],
                             "estimator": item["estimator"],
                             "n_train": item["n_train"],
                             "loocv_mae_seats": round(item["loocv_mae"], 2)}
                         for n, item in fitted.items()}
        bm = fitted[best_name]
        out = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "forecast_year": FORECAST_YEAR,
            "presidents_party": "R",
            "selected_model": best_name,
            "selected_estimator": bm["estimator"],
            "features_used": bm["cols"],
            "coefficients": dict(zip(bm["cols"],
                                     [round(float(c), 4) for c in bm["model"].coef_])),
            "intercept": round(float(bm["model"].intercept_), 4),
            "training_n": bm["n_train"],
            "training_years": MIDTERMS,
            "loocv_results": spec_results,
            "inputs_2026": inputs,
            "forecast": {
                "point_seat_change": round(point, 1),
                "ci95_low": round(lo, 1),
                "ci95_high": round(hi, 1),
                "historical_midterm_mean": round(historical_mean, 1),
            },
            "generic_ballot_2026_reference": gb,
            "training_frame": df.round(3).to_dict(orient="records"),
        }
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="2026 House midterm baseline forecast + what-if scenarios.",
        epilog="Examples:\n"
               "  python predict_house.py\n"
               "  python predict_house.py --approval-delta=10\n"
               "  python predict_house.py --gas-delta=-0.50\n"
               "  python predict_house.py --approval-delta=10 --unrate-delta=-1 --all-specs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--approval-delta", type=float, default=None,
                   help="Delta on Trump net approval (e.g. +10 = approval 10 pts higher than current)")
    p.add_argument("--cpi-delta", type=float, default=None,
                   help="Delta on CPI YoY (percentage points)")
    p.add_argument("--unrate-delta", type=float, default=None,
                   help="Delta on unemployment rate (percentage points)")
    p.add_argument("--gas-delta", type=float, default=None,
                   help="Delta on gas price ($/gal). Informational only — gas is not a feature.")
    p.add_argument("--all-specs", action="store_true",
                   help="Print forecasts from all candidate specs, not just the LOOCV winner")
    p.add_argument("--scenario-only", action="store_true",
                   help="Skip training-frame and LOOCV printout; show only forecast(s)")
    p.add_argument("--no-save", action="store_true",
                   help="Do not overwrite forecast_2026_house.json")
    main(p.parse_args())

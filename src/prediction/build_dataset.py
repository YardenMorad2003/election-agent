import pandas as pd

DATA_PATH = "src/prediction_data"


# ─────────────────────────────────────────────
# 🗳️ TARGET: Democratic vote share
# ─────────────────────────────────────────────
def load_presidential_data():
    df = pd.read_csv(f"{DATA_PATH}/presidents/1976-2020-president.csv")

    df = df[df["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])]

    grouped = df.groupby(["year", "party_simplified"])["candidatevotes"].sum().reset_index()
    pivot = grouped.pivot(index="year", columns="party_simplified", values="candidatevotes").reset_index()

    pivot["total_votes"] = pivot["DEMOCRAT"] + pivot["REPUBLICAN"]
    pivot["dem_vote_share"] = pivot["DEMOCRAT"] / pivot["total_votes"]

    return pivot[["year", "dem_vote_share"]]


# ─────────────────────────────────────────────
# 📊 Helper: Convert time series → yearly avg
# ─────────────────────────────────────────────
def yearly_average(df, date_col, value_col, new_name):
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    return df.groupby("year")[value_col].mean().reset_index().rename(columns={value_col: new_name})


# ─────────────────────────────────────────────
# 📊 Economic data
# ─────────────────────────────────────────────
def load_fred_data():
    cpi = pd.read_csv(f"{DATA_PATH}/FRED_Data/fred_cpi.csv")
    unrate = pd.read_csv(f"{DATA_PATH}/FRED_Data/fred_unrate.csv")
    #sp500 = pd.read_csv(f"{DATA_PATH}/FRED_Data/fred_sp500.csv")
    sentiment = pd.read_csv(f"{DATA_PATH}/FRED_Data/fred_umcsent.csv")

    cpi_y = yearly_average(cpi, "date", "cpi", "cpi")
    unrate_y = yearly_average(unrate, "date", "unrate", "unemployment")
    #sp500_y = yearly_average(sp500, "date", "sp500", "sp500")
    sentiment_y = yearly_average(sentiment, "date", "umcsent", "sentiment")

    df = cpi_y.merge(unrate_y, on="year", how="inner")
    #df = df.merge(sp500_y, on="year", how="inner")
    df = df.merge(sentiment_y, on="year", how="inner")

    return df


# ─────────────────────────────────────────────
# ⛽ Gas prices
# ─────────────────────────────────────────────
def load_gas_prices():
    gas = pd.read_csv(f"{DATA_PATH}/gas_prices.csv")
    return yearly_average(gas, "date", "gas_price", "gas_price")


# ─────────────────────────────────────────────
# 🧠 Approval (LAST before election)
# ─────────────────────────────────────────────
def load_approval():
    df = pd.read_csv(f"{DATA_PATH}/Approval_Data/pres_approval_data.csv")

    df["Start Date"] = pd.to_datetime(df["Start Date"])
    df["year"] = df["Start Date"].dt.year

    # take last observation per year
    df = df.sort_values("Start Date").groupby("year").tail(1)

    df = df.rename(columns={"Approving": "approval"})

    return df[["year", "approval"]]


# ─────────────────────────────────────────────
# 🗳️ Generic ballot
# ─────────────────────────────────────────────
def load_generic_ballot():
    df = pd.read_csv(f"{DATA_PATH}/generic_topline_historical.csv")

    df["modeldate"] = pd.to_datetime(df["modeldate"])
    df["year"] = df["modeldate"].dt.year

    df["generic_dem"] = df["dem_estimate"] - df["rep_estimate"]

    df = df.groupby("year")["generic_dem"].mean().reset_index()

    return df


# ─────────────────────────────────────────────
# 🧱 BUILD FINAL DATASET
# ─────────────────────────────────────────────
def build_dataset():
    y = load_presidential_data()

    fred = load_fred_data()
    gas = load_gas_prices()
    approval = load_approval()
    generic = load_generic_ballot()

    df = y.merge(fred, on="year", how="left")
    df = df.merge(gas, on="year", how="left")
    df = df.merge(approval, on="year", how="left")
    df = df.merge(generic, on="year", how="left")

    df = df.sort_values("year")
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print("\nColumns:", df.columns)
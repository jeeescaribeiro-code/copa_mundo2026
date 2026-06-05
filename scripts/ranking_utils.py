from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RANKING_DIR = BASE_DIR / "ranking"

DEFAULT_RANK = 100.0
DEFAULT_POINTS = 1000.0

TEAM_NAME_TO_CODE = {
    "Algeria": "ALG",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bosnia": "BIH",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Cabo Verde": "CPV",
    "Colombia": "COL",
    "Cote d'Ivoire": "CIV",
    "Côte d'Ivoire": "CIV",
    "Croatia": "CRO",
    "Curacao": "CUW",
    "Curaçao": "CUW",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "DR Congo": "COD",
    "Congo DR": "COD",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "IR Iran": "IRN",
    "Iraq": "IRQ",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "KSA",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Korea Republic": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Uruguay": "URU",
    "USA": "USA",
    "United States": "USA",
    "Uzbekistan": "UZB",
}


def normalize_team_code(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if len(text) == 3 and text.isupper():
        return text
    return TEAM_NAME_TO_CODE.get(text)


def load_historical_rankings(ranking_dir=RANKING_DIR):
    files = sorted(Path(ranking_dir).glob("fifa_ranking-*.csv"))
    if not files:
        return pd.DataFrame(columns=["country_abrv", "rank", "total_points", "rank_date"])

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if {"country_abrv", "rank", "total_points", "rank_date"}.issubset(frame.columns):
            frames.append(frame[["country_abrv", "rank", "total_points", "rank_date"]].copy())

    if not frames:
        return pd.DataFrame(columns=["country_abrv", "rank", "total_points", "rank_date"])

    rankings = pd.concat(frames, ignore_index=True)
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["total_points"] = pd.to_numeric(rankings["total_points"], errors="coerce")
    rankings = rankings.dropna(subset=["country_abrv", "rank", "total_points", "rank_date"])
    rankings = rankings.sort_values(["country_abrv", "rank_date"])
    return rankings.drop_duplicates(["country_abrv", "rank_date"], keep="last")


def ranking_lookup(rankings, code, match_date):
    if not code or rankings.empty or pd.isna(match_date):
        return DEFAULT_RANK, DEFAULT_POINTS, 0

    team_rankings = rankings[rankings["country_abrv"] == code]
    if team_rankings.empty:
        return DEFAULT_RANK, DEFAULT_POINTS, 0

    available = team_rankings[team_rankings["rank_date"] <= match_date]
    if available.empty:
        return DEFAULT_RANK, DEFAULT_POINTS, 0

    last = available.iloc[-1]
    return float(last["rank"]), float(last["total_points"]), 1


def add_ranking_features(matches, rankings=None):
    rankings = load_historical_rankings() if rankings is None else rankings
    df = matches.copy()

    if "match_date" in df.columns:
        dates = pd.to_datetime(df["match_date"], errors="coerce")
    elif "data" in df.columns:
        dates = pd.to_datetime(df["data"], errors="coerce")
    else:
        dates = pd.to_datetime(df["year"].astype(str) + "-01-01", errors="coerce")

    home_codes = (
        df["home_team_code"].apply(normalize_team_code)
        if "home_team_code" in df.columns
        else df["home_team"].apply(normalize_team_code)
    )
    away_codes = (
        df["away_team_code"].apply(normalize_team_code)
        if "away_team_code" in df.columns
        else df["away_team"].apply(normalize_team_code)
    )

    home_values = [ranking_lookup(rankings, code, date) for code, date in zip(home_codes, dates)]
    away_values = [ranking_lookup(rankings, code, date) for code, date in zip(away_codes, dates)]

    df[["ranking_home", "ranking_points_home", "ranking_available_home"]] = pd.DataFrame(home_values, index=df.index)
    df[["ranking_away", "ranking_points_away", "ranking_available_away"]] = pd.DataFrame(away_values, index=df.index)
    df["diff_ranking"] = df["ranking_away"] - df["ranking_home"]
    df["diff_ranking_points"] = df["ranking_points_home"] - df["ranking_points_away"]
    df["diff_ranking_available"] = df["ranking_available_home"] - df["ranking_available_away"]
    return df

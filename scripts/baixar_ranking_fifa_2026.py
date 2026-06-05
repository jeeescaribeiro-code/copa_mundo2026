import csv
import json
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "ranking" / "fifa_ranking-2026-04-01.csv"
URL = "https://api.fifa.com/api/v3/rankings/?gender=1&count=300"


def team_name(item):
    names = item.get("TeamName") or []
    for name in names:
        if name.get("Locale") == "en-GB":
            return name.get("Description")
    return names[0].get("Description") if names else ""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for item in payload.get("Results", []):
        rows.append({
            "rank": item.get("Rank"),
            "country_full": team_name(item),
            "country_abrv": item.get("IdCountry"),
            "total_points": item.get("DecimalTotalPoints"),
            "previous_points": item.get("DecimalPrevPoints"),
            "rank_change": item.get("RankingMovement"),
            "confederation": item.get("ConfederationName"),
            "rank_date": (item.get("PubDate") or "2026-04-01")[:10],
        })

    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ranking FIFA salvo em: {OUT}")
    print(f"Linhas: {len(rows)}")


if __name__ == "__main__":
    main()

"""
Tennis ATP API - Flask Backend
Serves tennis match data and analytics from ETL pipeline outputs.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load data from pipeline outputs
DATA_DIR = Path("pipeline_output")

def load_data():
    """Load dimension and fact tables from parquet files."""
    global dim_players, dim_tournaments, fact_matches
    
    dim_players = pd.read_parquet(DATA_DIR / "dim_players.parquet")
    dim_tournaments = pd.read_parquet(DATA_DIR / "dim_tournaments.parquet")
    
    # Load partitioned fact_matches
    fact_parts = []
    fact_dir = DATA_DIR / "fact_matches"
    if fact_dir.exists():
        for year_dir in fact_dir.iterdir():
            if year_dir.is_dir():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        parquet_file = month_dir / "data.parquet"
                        if parquet_file.exists():
                            df = pd.read_parquet(parquet_file)
                            # Extract year/month from path
                            year = int(year_dir.name.split("=")[1])
                            month = int(month_dir.name.split("=")[1])
                            df["year"] = year
                            df["month"] = month
                            fact_parts.append(df)
    
    fact_matches = pd.concat(fact_parts, ignore_index=True) if fact_parts else pd.DataFrame()
    
    # Build enriched view
    global fact_plus
    fact_plus = fact_matches.merge(
        dim_players.rename(columns={"player_name": "p1_name", "player_id": "p1_player_id"}),
        left_on="p1_id", right_on="p1_player_id", how="left"
    ).merge(
        dim_players.rename(columns={"player_name": "p2_name", "player_id": "p2_player_id"}),
        left_on="p2_id", right_on="p2_player_id", how="left"
    ).merge(
        dim_players.rename(columns={"player_name": "winner_name", "player_id": "w_player_id"}),
        left_on="winner_id", right_on="w_player_id", how="left"
    ).merge(
        dim_tournaments[["tournament_id", "Tournament", "Surface", "Series"]],
        on="tournament_id", how="left"
    )

# ---------- API Routes ----------

@app.route("/")
def home():
    """API documentation."""
    return jsonify({
        "name": "Tennis ATP API",
        "version": "1.0.0",
        "endpoints": {
            "/api/players": "List all players",
            "/api/players/<name>/stats": "Player career statistics",
            "/api/h2h/<player1>/<player2>": "Head-to-head record",
            "/api/tournaments": "List all tournaments",
            "/api/tournaments/<name>": "Tournament details",
            "/api/health": "Health check",
            "/api/dq": "Data quality report"
        }
    })


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "players": len(dim_players),
            "tournaments": len(dim_tournaments),
            "matches": len(fact_matches)
        }
    })


@app.route("/api/dq")
def dq_report():
    """Return latest DQ report."""
    dq_path = DATA_DIR / "dq_report.json"
    if dq_path.exists():
        with open(dq_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "DQ report not found"}), 404


@app.route("/api/players")
def list_players():
    """List all players with optional search."""
    search = request.args.get("q", "").lower()
    limit = int(request.args.get("limit", 50))
    
    players = dim_players["player_name"].tolist()
    
    if search:
        players = [p for p in players if search in p.lower()]
    
    return jsonify({
        "count": len(players[:limit]),
        "total": len(players),
        "players": players[:limit]
    })


@app.route("/api/players/<name>/stats")
def player_stats(name):
    """Get player career statistics."""
    # URL decode the name (spaces become %20)
    player_name = name.replace("%20", " ")
    
    # Find exact or partial match
    matches = dim_players[dim_players["player_name"].str.contains(player_name, case=False, na=False)]
    
    if len(matches) == 0:
        return jsonify({"error": f"Player '{player_name}' not found"}), 404
    
    # Use first match
    player_name = matches.iloc[0]["player_name"]
    
    # Get all matches for this player
    p1_matches = fact_plus[fact_plus["p1_name"] == player_name].copy()
    p1_matches["is_win"] = p1_matches["winner_id"] == p1_matches["p1_id"]
    
    p2_matches = fact_plus[fact_plus["p2_name"] == player_name].copy()
    p2_matches["is_win"] = p2_matches["winner_id"] == p2_matches["p2_id"]
    
    all_matches = pd.concat([p1_matches, p2_matches])
    
    if len(all_matches) == 0:
        return jsonify({"error": "No matches found for player"}), 404
    
    wins = int(all_matches["is_win"].sum())
    losses = len(all_matches) - wins
    
    # Surface breakdown
    surface_stats = {}
    for surface in ["Hard", "Clay", "Grass"]:
        s_matches = all_matches[all_matches["Surface"] == surface]
        if len(s_matches) > 0:
            s_wins = int(s_matches["is_win"].sum())
            surface_stats[surface.lower()] = {
                "matches": len(s_matches),
                "wins": s_wins,
                "losses": len(s_matches) - s_wins,
                "win_rate": round(s_wins / len(s_matches) * 100, 1)
            }
    
    return jsonify({
        "player": player_name,
        "career": {
            "total_matches": len(all_matches),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(all_matches) * 100, 1)
        },
        "by_surface": surface_stats,
        "generated_at": datetime.now().isoformat()
    })


@app.route("/api/h2h/<player1>/<player2>")
def head_to_head(player1, player2):
    """Get head-to-head record between two players."""
    # URL decode
    p1 = player1.replace("%20", " ")
    p2 = player2.replace("%20", " ")
    
    # Find matches
    h2h = fact_plus[
        ((fact_plus["p1_name"].str.contains(p1, case=False, na=False)) & 
         (fact_plus["p2_name"].str.contains(p2, case=False, na=False))) |
        ((fact_plus["p1_name"].str.contains(p2, case=False, na=False)) & 
         (fact_plus["p2_name"].str.contains(p1, case=False, na=False)))
    ].sort_values("Date", ascending=False)
    
    if len(h2h) == 0:
        return jsonify({"error": "No head-to-head matches found"}), 404
    
    # Get actual player names from first match
    actual_p1 = h2h.iloc[0]["p1_name"]
    actual_p2 = h2h.iloc[0]["p2_name"]
    
    p1_wins = len(h2h[h2h["winner_name"] == actual_p1])
    p2_wins = len(h2h[h2h["winner_name"] == actual_p2])
    
    # Recent matches
    recent = h2h.head(10)[["Date", "Tournament", "Round", "Surface", "winner_name", "Score"]].copy()
    recent["Date"] = recent["Date"].astype(str).str[:10]
    
    return jsonify({
        "player_1": actual_p1,
        "player_2": actual_p2,
        "head_to_head": {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "total": len(h2h)
        },
        "recent_matches": recent.to_dict(orient="records"),
        "generated_at": datetime.now().isoformat()
    })


@app.route("/api/tournaments")
def list_tournaments():
    """List all tournaments."""
    limit = int(request.args.get("limit", 50))
    surface = request.args.get("surface")
    
    tournaments = dim_tournaments.copy()
    
    if surface:
        tournaments = tournaments[tournaments["Surface"].str.lower() == surface.lower()]
    
    return jsonify({
        "count": len(tournaments[:limit]),
        "total": len(tournaments),
        "tournaments": tournaments[["Tournament", "Series", "Surface"]].head(limit).to_dict(orient="records")
    })


@app.route("/api/tournaments/<name>")
def tournament_details(name):
    """Get tournament details and recent winners."""
    tournament_name = name.replace("%20", " ")
    
    matches = fact_plus[fact_plus["Tournament"].str.contains(tournament_name, case=False, na=False)]
    
    if len(matches) == 0:
        return jsonify({"error": "Tournament not found"}), 404
    
    # Get finals
    finals = matches[matches["Round"] == "The Final"].sort_values("Date", ascending=False)
    
    recent_winners = finals.head(10)[["Date", "winner_name", "Score"]].copy()
    recent_winners["Date"] = recent_winners["Date"].astype(str).str[:10]
    recent_winners.columns = ["year", "champion", "score"]
    
    return jsonify({
        "tournament": matches.iloc[0]["Tournament"],
        "surface": matches.iloc[0]["Surface"],
        "series": matches.iloc[0]["Series"],
        "total_matches": len(matches),
        "recent_champions": recent_winners.to_dict(orient="records"),
        "generated_at": datetime.now().isoformat()
    })


# ---------- Main ----------

if __name__ == "__main__":
    print("Loading data from pipeline outputs...")
    load_data()
    print(f"Loaded: {len(dim_players)} players, {len(dim_tournaments)} tournaments, {len(fact_matches)} matches")
    print("\nStarting Tennis ATP API on http://localhost:5000")
    app.run(debug=True, port=5000)

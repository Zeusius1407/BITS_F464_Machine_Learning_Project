"""Shared constants for the IPL pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Franchise renames -> one canonical name per franchise.
# Deccan Chargers (2008-12) and Sunrisers Hyderabad (2013-) are legally different
# franchises, so they are kept separate by default. Set MERGE_DECCAN_INTO_SRH = True
# to treat them as one "Hyderabad" side instead.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
MERGE_DECCAN_INTO_SRH = False

TEAM_ABBR = {
    "Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Deccan Chargers": "DCH",
    "Gujarat Lions": "GL", "Gujarat Titans": "GT", "Kochi Tuskers Kerala": "KTK",
    "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG", "Mumbai Indians": "MI",
    "Pune Warriors": "PWI", "Punjab Kings": "PBKS", "Rajasthan Royals": "RR",
    "Rising Pune Supergiant": "RPS", "Royal Challengers Bengaluru": "RCB",
    "Sunrisers Hyderabad": "SRH",
}

# Venue clean-up: first the ", City" suffix is stripped, then these aliases are applied.
VENUE_ALIASES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",                       # renamed 2019
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",          # rebuilt + renamed
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",  # same ground, Pune
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",                  # Abu Dhabi, renamed
    "Nehru Stadium": "Jawaharlal Nehru Stadium, Kochi",
}
KNOWN_CITY_SUFFIXES = {
    "Delhi", "Guwahati", "Lucknow", "Mumbai", "Visakhapatnam", "Kolkata", "Dharamsala",
    "Bengaluru", "Chennai", "Mullanpur", "Pune", "Ahmedabad", "Chandigarh", "Hyderabad",
    "Jaipur", "Jamtha", "Abu Dhabi", "Mohali",
}

# Phases (overs are 0-indexed in deliveries.csv: over 0 = 1st over)
PHASES = {"pp": range(0, 6), "mid": range(6, 15), "death": range(15, 20)}

# Dismissals not credited to the bowler / not counted as a batter "out"
NON_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field"}
NOT_AN_OUT = {"retired hurt"}

SNAPSHOT_OVERS = (6, 10, 15)      # innings state is captured after these many overs
FORM_WINDOWS = (5, 10)            # last-k matches for team form
IMPACT_PLAYER_FROM = 2023

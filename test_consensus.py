import os
import sys

# Ensure backend root is in PYTHONPATH
sys.path.append(r'C:\Users\hkatongole\Downloads\plusone-backend-main\plusone-backend-main')

from database import get_connection
from ml.consensus_engine import run_consensus
import json

def test_consensus():
    conn = get_connection()
    cur = conn.cursor()
    
    # Premier League = 9070
    # Let's find an upcoming match
    cur.execute('''
        SELECT m.id, m.home_team_id, m.away_team_id, ht.name as home_name, at.name as away_name, m.match_date 
        FROM matches m 
        JOIN teams ht ON m.home_team_id = ht.id 
        JOIN teams at ON m.away_team_id = at.id 
        WHERE m.league_id = 9070 
        AND m.match_date >= CURRENT_DATE 
        ORDER BY m.match_date ASC LIMIT 1
    ''')
    
    fx = cur.fetchone()
    cur.close()
    conn.close()
    
    if not fx:
        print("No upcoming Premier League fixtures found.")
        return
        
    print(f"Testing consensus for: {fx['home_name']} vs {fx['away_name']} (Match ID: {fx['id']})")
    print("-" * 50)
    
    result = run_consensus(
        home_team_id=fx['home_team_id'], 
        away_team_id=fx['away_team_id'], 
        league_id=9070, 
        season_id=None
    )
    
    print(json.dumps(result, indent=2))
    
if __name__ == '__main__':
    test_consensus()

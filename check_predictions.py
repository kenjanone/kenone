import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = "postgresql://postgres.efsvjpgwfyuieixmowym:.suriah1231@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

def check_db():
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cur = conn.cursor()
        
        print("\n--- Evaluated Predictions in prediction_log ---")
        cur.execute("""
            SELECT id, match_id, home_team, away_team, predicted, actual, correct, created_at 
            FROM prediction_log 
            WHERE correct IS NOT NULL 
            ORDER BY created_at DESC LIMIT 10
        """)
        rows = cur.fetchall()
        if not rows:
            print("No evaluated predictions found.")
        else:
            for r in rows:
                print(f"Log ID: {r['id']} | Match ID: {r['match_id']} | {r['home_team']} vs {r['away_team']} | Pred: {r['predicted']} | Actual: {r['actual']} | Correct: {r['correct']} | Created: {r['created_at']}")

        print("\n--- Un-evaluated Predictions in prediction_log ---")
        cur.execute("""
            SELECT id, match_id, home_team, away_team, predicted, actual, correct, created_at 
            FROM prediction_log 
            WHERE correct IS NULL 
            ORDER BY created_at DESC LIMIT 10
        """)
        rows = cur.fetchall()
        if not rows:
            print("No un-evaluated predictions found.")
        else:
            for r in rows:
                print(f"Log ID: {r['id']} | Match ID: {r['match_id']} | {r['home_team']} vs {r['away_team']} | Pred: {r['predicted']} | Actual: {r['actual']} | Correct: {r['correct']} | Created: {r['created_at']}")
                if r['match_id'] is not None:
                    # check if there's a match for it
                    cur.execute("SELECT id, home_score, away_score, status FROM matches WHERE id = %s", (r['match_id'],))
                    match = cur.fetchone()
                    if match:
                        score_str = f"{match['home_score']}-{match['away_score']} ({match['status']})" if match['home_score'] is not None else f"UNPLAYED ({match['status']})"
                        print(f"  -> Found Match ID {match['id']}: Score is {score_str}")
                    else:
                        print(f"  -> Match ID {r['match_id']} NOT FOUND in matches table!")
                else:
                    print("  -> ERROR: match_id is NULL.")

        print("\n--- Summary Check ---")
        cur.execute("SELECT COUNT(*) as c FROM prediction_log WHERE match_id IS NULL")
        null_matches = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM prediction_log WHERE match_id IS NOT NULL AND correct IS NULL")
        has_matches_waiting = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM prediction_log WHERE match_id IS NOT NULL AND correct IS NOT NULL")
        evaluated = cur.fetchone()['c']
        
        print(f"Predictions missing match_id: {null_matches}")
        print(f"Predictions with match_id (waiting evaluation): {has_matches_waiting}")
        print(f"Predictions evaluated (correct IS NOT NULL): {evaluated}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying db: {e}")

if __name__ == '__main__':
    check_db()

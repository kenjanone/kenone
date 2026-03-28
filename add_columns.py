import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def add_enrichment_columns():
    if not DB_URL:
        print("Error: DATABASE_URL not set in environment.")
        return

    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # Check existing columns
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='prediction_log';
        """)
        existing_cols = {row[0] for row in cur.fetchall()}

        col_defs = {
            "enrichment_predicted_outcome": "TEXT",
            "legacy_predicted_outcome": "TEXT",
            "dc_predicted_outcome": "TEXT",
            "ml_predicted_outcome": "TEXT"
        }

        added = []
        for col_name, col_type in col_defs.items():
            if col_name not in existing_cols:
                print(f"Adding column '{col_name}' to prediction_log...")
                cur.execute(f"ALTER TABLE prediction_log ADD COLUMN {col_name} {col_type};")
                added.append(col_name)

        if added:
            conn.commit()
            print(f"Successfully added columns: {', '.join(added)}")
        else:
            print("All columns already exist.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error updating database schema: {e}")

if __name__ == "__main__":
    add_enrichment_columns()

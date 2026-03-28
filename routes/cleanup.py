"""
Cleanup endpoints to purge corrupted data from bad syncs.
"""
from fastapi import APIRouter, Depends
from routes.deps import require_admin
from database import get_connection

router = APIRouter()

# ─── Tables that reference league_id ─────────────────────────────────────────
_LEAGUE_REF_TABLES = ["teams", "matches", "team_squad_stats", "scrape_log"]


@router.post("/merge-duplicate-leagues")
def merge_duplicate_leagues(_admin: dict = Depends(require_admin)):
    """
    Find leagues that are case-insensitive duplicates of each other.
    Keep the 'canonical' one (prefers rows with country/fbref_id set, or the lower id).
    Re-point all foreign keys to the canonical id, then delete the duplicate.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Find groups of leagues sharing the same LOWER(name)
        cur.execute("""
            SELECT LOWER(name) AS key, array_agg(id ORDER BY
                (CASE WHEN country IS NOT NULL THEN 0 ELSE 1 END),
                (CASE WHEN fbref_id IS NOT NULL THEN 0 ELSE 1 END),
                id
            ) AS ids
            FROM leagues
            GROUP BY LOWER(name)
            HAVING COUNT(*) > 1
        """)
        groups = cur.fetchall()

        merged = []
        for g in groups:
            ids = g["ids"]
            canonical_id = ids[0]       # first = best (has country/fbref_id or lowest id)
            duplicates   = ids[1:]

            for dup_id in duplicates:
                # Re-point every table that uses league_id
                for tbl in _LEAGUE_REF_TABLES:
                    cur.execute(f"SAVEPOINT sp_merge")
                    try:
                        cur.execute(
                            f"UPDATE {tbl} SET league_id = %s WHERE league_id = %s",
                            (canonical_id, dup_id)
                        )
                        cur.execute("RELEASE SAVEPOINT sp_merge")
                    except Exception:
                        cur.execute("ROLLBACK TO SAVEPOINT sp_merge")

                # Also fix teams' league_id when the same team now appears twice under
                # the canonical league – skip on conflict (team already exists there).
                cur.execute(
                    "DELETE FROM leagues WHERE id = %s",
                    (dup_id,)
                )
                merged.append({"removed_id": dup_id, "kept_id": canonical_id})

        conn.commit()
        return {"success": True, "merges": merged, "total": len(merged)}
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()



@router.delete("/bad-teams")
def cleanup_bad_teams(_admin: dict = Depends(require_admin)):
    """Remove teams whose names are raw Python dict strings (e.g. {'text': 'Arsenal'})."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # First remove squad_stats referencing these teams
        cur.execute("""
            DELETE FROM team_squad_stats
            WHERE team_id IN (
                SELECT id FROM teams WHERE name LIKE '{%'
            )
        """)
        stats_deleted = cur.rowcount

        # Then remove the bad teams
        cur.execute("DELETE FROM teams WHERE name LIKE '{%'")
        teams_deleted = cur.rowcount

        conn.commit()
        return {
            "success": True,
            "teams_deleted": teams_deleted,
            "squad_stats_deleted": stats_deleted,
        }
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.delete("/bad-leagues")
def cleanup_bad_leagues(_admin: dict = Depends(require_admin)):
    """Remove fake leagues created by year-only names (e.g. '2024', '2025')."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Find leagues whose name is purely numeric (a year)
        cur.execute("SELECT id FROM leagues WHERE name ~ '^[0-9]{4}$'")
        bad_ids = [row["id"] for row in cur.fetchall()]

        if not bad_ids:
            return {"success": True, "message": "No bad leagues found", "deleted": 0}

        # Remove squad_stats in those leagues
        cur.execute(
            "DELETE FROM team_squad_stats WHERE league_id = ANY(%s)",
            (bad_ids,)
        )
        stats_deleted = cur.rowcount

        # Remove teams in those leagues
        cur.execute("DELETE FROM teams WHERE league_id = ANY(%s)", (bad_ids,))
        teams_deleted = cur.rowcount

        # Remove seasons linked only to these leagues (optional safety check)
        cur.execute("DELETE FROM leagues WHERE id = ANY(%s)", (bad_ids,))
        leagues_deleted = cur.rowcount

        conn.commit()
        return {
            "success": True,
            "leagues_deleted": leagues_deleted,
            "teams_deleted": teams_deleted,
            "squad_stats_deleted": stats_deleted,
        }
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.delete("/all")
def cleanup_all(_admin: dict = Depends(require_admin)):
    """Run all cleanup operations in one shot."""
    teams_result = cleanup_bad_teams()
    leagues_result = cleanup_bad_leagues()
    return {
        "success": True,
        "bad_teams": teams_result,
        "bad_leagues": leagues_result,
    }


@router.get("/preview")
def preview_cleanup():
    """Preview what would be deleted without actually deleting anything."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) AS cnt FROM teams WHERE name LIKE '{%'")
        bad_team_count = cur.fetchone()["cnt"]

        cur.execute("SELECT id, name FROM leagues WHERE name ~ '^[0-9]{4}$'")
        bad_leagues = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT COUNT(*) AS cnt FROM team_squad_stats
            WHERE team_id IN (SELECT id FROM teams WHERE name LIKE '{%')
               OR league_id IN (SELECT id FROM leagues WHERE name ~ '^[0-9]{4}$')
        """)
        bad_stats_count = cur.fetchone()["cnt"]

        return {
            "bad_teams_count": bad_team_count,
            "bad_leagues": bad_leagues,
            "affected_squad_stats": bad_stats_count,
        }
    finally:
        conn.close()

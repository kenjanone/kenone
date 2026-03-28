-- migration for standalone data enrichment sync (odds, injuries, clubelo)
-- This creates independent tables so we don't disrupt the core prediction logic

CREATE TABLE IF NOT EXISTS match_odds (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    b365_home_win DECIMAL(5,2),
    b365_draw DECIMAL(5,2),
    b365_away_win DECIMAL(5,2),
    raw_data JSONB,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id)
);

CREATE TABLE IF NOT EXISTS player_injuries (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    player_name VARCHAR(255) NOT NULL,
    injury_type VARCHAR(255),
    return_date VARCHAR(255),
    raw_data JSONB,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, player_name)
);

CREATE TABLE IF NOT EXISTS team_clubelo (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    elo_date DATE NOT NULL,
    elo DECIMAL(8,2),
    raw_data JSONB,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, elo_date)
);

-- Index for speedy lookups
CREATE INDEX IF NOT EXISTS idx_match_odds_match_id ON match_odds(match_id);
CREATE INDEX IF NOT EXISTS idx_player_injuries_team ON player_injuries(team_id);
CREATE INDEX IF NOT EXISTS idx_team_clubelo_team ON team_clubelo(team_id);

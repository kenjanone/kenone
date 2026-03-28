import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv


from routes import leagues, teams, matches, standings, squad_stats, player_stats, sync, sync_enrichment, health, auth, cleanup, predictions, venue_stats, prediction_log, markets, performance, feedback


load_dotenv()

# ─── Numpy → psycopg2 type adapters ──────────────────────────────────────────
# Prevents "schema np does not exist" errors when numpy float64/int64 values
# are passed directly to psycopg2 inserts across the entire app.
import numpy as np
import psycopg2.extensions

psycopg2.extensions.register_adapter(np.float64, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.float32, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.int64,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.int32,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.bool_,   lambda x: psycopg2.extensions.AsIs(bool(x)))


# ─── HTTPS redirect middleware ─────────────────────────────────────────────────
# Railway/Render terminates TLS at its edge proxy and forwards plain HTTP to the
# app container. The original scheme is in X-Forwarded-Proto.
# We redirect any non-HTTPS request to HTTPS so browsers never hit mixed-content.
# Localhost is excluded so local development still works on http://.

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        proto = request.headers.get("x-forwarded-proto", "")
        host  = request.headers.get("host", "")
        is_local = host.startswith("localhost") or host.startswith("127.0.0.1")
        if proto == "http" and not is_local:
            https_url = str(request.url).replace("http://", "https://", 1)
            return RedirectResponse(url=https_url, status_code=301)
        return await call_next(request)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Analytics API",
    description="Production API for football data scraped from FBref",
    version="1.0.0",
    # Railway terminates SSL at its proxy and sends plain HTTP to the app.
    # Without this, FastAPI redirects /api/leagues → http://…/api/leagues/ (trailing slash),
    # which browsers block as Mixed Content (page is HTTPS, redirect is HTTP).
    redirect_slashes=False,
)

# Apply HTTPS redirect before CORS so redirects carry the right headers
app.add_middleware(HTTPSRedirectMiddleware)


# ─── CORS ─────────────────────────────────────────────────────────────────────
# IMPORTANT: FastAPI's CORSMiddleware does NOT support glob/wildcard patterns
# like "https://*.vercel.app" in allow_origins — it does exact string matching.
# Use allow_origin_regex for real subdomain matching.

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4000",
    "https://localhost:5173",
    "https://football-analytics-eight.vercel.app",
    "https://plusone-frontend-mu.vercel.app",
]

# Regex covers:
#   - localhost on any port over http or https           (dev)
#   - Any *.vercel.app deployment (including previews)   (Vercel)
#   - Any *.onrender.com service                         (Render)
#   - Any *.railway.app service (including *.up.railway) (Railway)
CORS_ORIGIN_REGEX = (
    r"https?://localhost(:\d+)?"
    r"|https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.vercel\.app"
    r"|https://[a-z0-9-]+\.onrender\.com"
    r"|https://[a-z0-9-]+(?:\.up)?\.railway\.app"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
    max_age=600,   # cache preflight for 10 min
)


# Register all route modules
app.include_router(health.router,       prefix="/api",             tags=["Health"])
app.include_router(leagues.router,      prefix="/api/leagues",     tags=["Leagues"])
app.include_router(teams.router,        prefix="/api/teams",       tags=["Teams"])
app.include_router(matches.router,      prefix="/api/matches",     tags=["Matches"])
app.include_router(standings.router,    prefix="/api/standings",   tags=["Standings"])
app.include_router(squad_stats.router,  prefix="/api/squad-stats", tags=["Squad Stats"])
app.include_router(venue_stats.router,  prefix="/api/venue-stats", tags=["Venue Stats"])
app.include_router(player_stats.router, prefix="/api/players",     tags=["Players"])
app.include_router(sync.router,         prefix="/api/sync",        tags=["Sync"])
app.include_router(sync_enrichment.router, prefix="/api/sync",        tags=["Sync Validation"])
app.include_router(cleanup.router,      prefix="/api/cleanup",     tags=["Cleanup"])
app.include_router(auth.router,                                    tags=["Auth"])
app.include_router(predictions.router,    prefix="/api/predictions",    tags=["Predictions"])
app.include_router(prediction_log.router, prefix="/api/prediction-log", tags=["Prediction Log"])
app.include_router(markets.router,        prefix="/api/markets",         tags=["Markets"])
app.include_router(performance.router,    prefix="/api/performance",      tags=["Performance"])
app.include_router(feedback.router,        prefix="/api/feedback",         tags=["Feedback"])


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


-- ============================================================
-- Auth Migration — PlusOne Backend
-- Run ONCE in the Supabase SQL Editor
-- ============================================================

-- ─── 1. users table ──────────────────────────────────────────────────────────
-- Stores registered admin/user accounts.
-- Passwords are stored as bcrypt hashes — never plain text.

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user'   -- 'user' | 'admin'
                  CHECK (role IN ('user', 'admin')),
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);


-- ─── 2. Seed your first admin account ────────────────────────────────────────
-- IMPORTANT: Do NOT use a plain-text password here.
-- Generate a bcrypt hash in Python first:
--
--   python3 -c "from passlib.context import CryptContext; \
--               c = CryptContext(schemes=['bcrypt']); \
--               print(c.hash('your_password_here'))"
--
-- Then paste the hash below and run this INSERT:
--
-- INSERT INTO users (email, password_hash, role)
-- VALUES (
--     'admin@yourapp.com',
--     '$2b$12$REPLACE_THIS_WITH_YOUR_BCRYPT_HASH',
--     'admin'
-- )
-- ON CONFLICT (email) DO NOTHING;

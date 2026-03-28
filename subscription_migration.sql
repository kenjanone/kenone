-- PlusOne Subscription Migration
-- Run this in Supabase SQL Editor (safe to run multiple times)

-- 1. Add phone column if missing
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone TEXT;

-- 2. Add subscription plan tracking
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS trial_expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days');
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_expires_at TIMESTAMPTZ;

-- 3. Payment reference fields
ALTER TABLE users ADD COLUMN IF NOT EXISTS payment_ref TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS payment_amount TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS payment_method TEXT;

-- 4. Add plan check constraint (safe way to add without breaking if already exists)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'users_plan_check'
  ) THEN
    ALTER TABLE users ADD CONSTRAINT users_plan_check
      CHECK (plan IN ('trial','basic','pro','elite','admin'));
  END IF;
END $$;

-- 5. Add feedback reply columns to user_feedback table
ALTER TABLE user_feedback ADD COLUMN IF NOT EXISTS reply_text TEXT;
ALTER TABLE user_feedback ADD COLUMN IF NOT EXISTS replied_at TIMESTAMPTZ;
ALTER TABLE user_feedback ADD COLUMN IF NOT EXISTS replied_by TEXT;

-- 6. Existing admins: set is_active=true and give them 'admin' plan
UPDATE users SET is_active = true, plan = 'admin' WHERE role = 'admin';

-- Verify
SELECT id, email, role, plan, is_active, trial_expires_at, subscription_expires_at
FROM users ORDER BY id;

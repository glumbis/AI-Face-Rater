-- The leaderboard and community average. Run this once in the Supabase dashboard: SQL Editor > New query > paste > Run.
-- Running it again is safe. Only a nickname, a score and the per-region scores are stored, never a picture.
--
-- The website talks to the database only through the functions below, so visitors can't read or change the table
-- directly (row level security is on and the public key has no rights on the table). Each browser has a random secret
-- token; only its SHA-256 is stored, so the token proves which entries are yours without an account.

create table if not exists public.entries (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(name) between 2 and 20 and name ~ '^[[:alnum:] _.-]+$'),
  score numeric(3, 1) not null check (score between 0 and 10),
  reference text not null check (reference in ('boy', 'girl')),
  regions jsonb check (regions is null or (jsonb_typeof(regions) = 'object' and pg_column_size(regions) < 400)),
  owner_hash text not null,
  consent_version smallint not null,
  created_at timestamptz not null default now(),
  -- One entry per browser and model face: submitting again replaces it
  unique (owner_hash, reference)
);

alter table public.entries enable row level security;
revoke all on public.entries from anon, authenticated;

-- Adds or replaces this browser's entry for that model face
create or replace function public.submit_score(p_token text, p_name text, p_score numeric, p_reference text,
                                               p_regions jsonb, p_consent_version smallint)
returns void language sql security definer set search_path = '' as $$
  insert into public.entries (name, score, reference, regions, owner_hash, consent_version)
  select trim(p_name), round(p_score, 1), p_reference, p_regions,
         encode(sha256(convert_to(p_token, 'UTF8')), 'hex'), p_consent_version
  where char_length(p_token) >= 32  -- the website makes 64-character random tokens
  on conflict (owner_hash, reference) do update
    set name = excluded.name, score = excluded.score, regions = excluded.regions,
        consent_version = excluded.consent_version, created_at = now();
$$;

-- The best scores for a model face (never the owner hash)
create or replace function public.top_scores(p_reference text, p_limit int default 20)
returns table (name text, score numeric, created_at timestamptz) language sql stable security definer set search_path = '' as $$
  select e.name, e.score, e.created_at from public.entries e
  where e.reference = p_reference
  order by e.score desc, e.created_at asc
  limit least(greatest(p_limit, 1), 50);
$$;

-- How a score compares with everyone who shared theirs: how many, their average, the share of them below this score,
-- and the average of each region
create or replace function public.community_stats(p_reference text, p_score numeric)
returns table (players bigint, average numeric, below numeric, regions jsonb)
language sql stable security definer set search_path = '' as $$
  select count(*),
         round(avg(e.score), 1),
         round(coalesce(avg((e.score < p_score)::int), 0), 2),
         (select jsonb_object_agg(r.key, round(r.avg, 1)) from (
            select kv.key, avg(kv.value::numeric) as avg
            from public.entries x, jsonb_each_text(x.regions) kv
            where x.reference = p_reference and kv.value ~ '^[0-9.]+$'
            group by kv.key) r)
  from public.entries e
  where e.reference = p_reference;
$$;

-- Deletes every entry from this browser; returns how many were removed
create or replace function public.delete_my_entries(p_token text)
returns int language sql security definer set search_path = '' as $$
  with gone as (
    delete from public.entries where char_length(p_token) >= 32 and owner_hash = encode(sha256(convert_to(p_token, 'UTF8')), 'hex') returning 1
  )
  select count(*)::int from gone;
$$;

revoke execute on function public.submit_score, public.top_scores, public.community_stats, public.delete_my_entries from public;
grant execute on function public.submit_score, public.top_scores, public.community_stats, public.delete_my_entries to anon;

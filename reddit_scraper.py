#!/usr/bin/env python
"""
Enhanced Reddit scraper
────────────────────────────────────────────────────────
• Accepts username OR full profile URL
• Interactive prompt if no user specified
• Flag  --per-user-db  to store each target in its own DB
• Adds  target_user  column so downstream code can filter easily

Examples
========
python reddit_scraper.py spez
python reddit_scraper.py https://www.reddit.com/user/kojied/ --limit 500
python reddit_scraper.py --per-user-db          # prompts for profile
"""

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    select,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session
import praw

# ───────────────────────────────
# 0.  Env & credential validation
# ───────────────────────────────
load_dotenv()
CID = os.getenv("REDDIT_CLIENT_ID")
CSECRET = os.getenv("REDDIT_CLIENT_SECRET")
UAGENT = os.getenv("REDDIT_USER_AGENT", "reddit-scraper/0.3 (by u/unknown)")

if not (CID and CSECRET):
    sys.exit("❌  Missing Reddit keys.  Add them to a .env file.")

# ───────────────────────────────
# 1.  SQLAlchemy ORM
# ───────────────────────────────
class Base(DeclarativeBase):
    metadata = MetaData()


class RedditItem(Base):
    __tablename__ = "reddit_items"

    id = Column(String, primary_key=True)           # thing ID (t3_, t1_)
    target_user = Column(String, nullable=False)    # profile we scraped
    item_type = Column(String, nullable=False)      # submission | comment
    subreddit = Column(String, nullable=False)
    author = Column(String, nullable=False)
    created_utc = Column(DateTime, nullable=False)
    title = Column(String)                          # submissions only
    body = Column(Text)
    score = Column(Integer)
    permalink = Column(String)


def get_engine(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    return engine


# ───────────────────────────────
# 2.  Reddit helpers
# ───────────────────────────────
def reddit_client():
    return praw.Reddit(
        client_id=CID,
        client_secret=CSECRET,
        user_agent=UAGENT,
    )


USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


def parse_username(raw: str) -> str:
    """
    Accepts:
      • 'spez'
      • 'u/spez'
      • 'https://www.reddit.com/user/spez/'  (trailing slash optional)

    Returns -> 'spez'
    """
    raw = raw.strip()

    # If it's a URL, extract /user/<name> segment
    if raw.lower().startswith("http"):
        path = urlparse(raw).path.strip("/")        # e.g. 'user/spez'
        parts = path.split("/")
        if len(parts) >= 2 and parts[0].lower() in {"user", "u"}:
            raw = parts[1]
        else:
            raise ValueError("❌  URL must look like …/user/<name>/")

    # Handle optional 'u/' prefix
    if raw.lower().startswith("u/"):
        raw = raw[2:]

    if not USERNAME_RE.fullmatch(raw):
        raise ValueError(f"❌  Invalid Reddit username: '{raw}'")

    return raw


def fetch_activity(client, username, limit):
    redditor = client.redditor(username)
    return (
        list(redditor.submissions.new(limit=limit)),
        list(redditor.comments.new(limit=limit)),
    )


def upsert(engine, target_user, submissions, comments):
    ids = [s.id for s in submissions] + [c.id for c in comments]

    with Session(engine) as ses:
        existing = {
            r[0]
            for r in ses.execute(
                select(RedditItem.id).where(RedditItem.id.in_(ids))
            )
        }
        rows = []

        def to_dt(ts):
            return dt.datetime.fromtimestamp(ts, dt.timezone.utc)

        for s in submissions:
            if s.id in existing:
                continue
            rows.append(
                RedditItem(
                    id=s.id,
                    target_user=target_user,
                    item_type="submission",
                    subreddit=str(s.subreddit),
                    author=str(s.author or ""),
                    created_utc=to_dt(s.created_utc),
                    title=s.title,
                    body=s.selftext or "",
                    score=s.score,
                    permalink=f"https://reddit.com{s.permalink}",
                )
            )

        for c in comments:
            if c.id in existing:
                continue
            rows.append(
                RedditItem(
                    id=c.id,
                    target_user=target_user,
                    item_type="comment",
                    subreddit=str(c.subreddit),
                    author=str(c.author or ""),
                    created_utc=to_dt(c.created_utc),
                    title=None,
                    body=c.body,
                    score=c.score,
                    permalink=f"https://reddit.com{c.permalink}",
                )
            )

        ses.add_all(rows)
        ses.commit()
        return len(rows)


# ───────────────────────────────
# 3.  CLI
# ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Scrape Reddit user activity")
    ap.add_argument(
        "profile",
        nargs="?",
        help="Username or full profile URL (omit to enter interactively)",
    )
    ap.add_argument("--limit", type=int, default=1000, help="Items per feed")
    ap.add_argument(
        "--db",
        default="reddit_cache.db",
        help="SQLite file (ignored if --per-user-db supplied)",
    )
    ap.add_argument(
        "--per-user-db",
        action="store_true",
        help="Create data/<username>.db instead of a shared DB",
    )
    args = ap.parse_args()

    # interactive fallback
    raw_profile = args.profile or input("Reddit profile (username or URL): ").strip()
    target = parse_username(raw_profile)

    db_path = (
        Path("data") / f"{target}.db"
        if args.per_user_db
        else Path(args.db)
    )
    db_path.parent.mkdir(exist_ok=True)

    eng = get_engine(db_path)
    reddit = reddit_client()

    print(f"⏳  Fetching u/{target} (limit={args.limit}) …")
    subs, coms = fetch_activity(reddit, target, args.limit)
    added = upsert(eng, target, subs, coms)

    with Session(eng) as ses:
        total_rows = ses.scalar(select(func.count()).select_from(RedditItem))

    print(
        f"✅  Added {added} new rows for u/{target} in '{db_path}'. "
        f"(Total rows in table: {total_rows})"
    )


if __name__ == "__main__":
    main()

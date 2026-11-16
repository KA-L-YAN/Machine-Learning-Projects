#!/usr/bin/env python3
"""
Unsave every saved POST (photos, carousels, videos, reels) from your own
Instagram account. Saved audio/music is NOT touched -- Instagram keeps that
in a separate list that this script never reads.

Usage:
    pip install instagrapi
    python unsave_all_posts.py                 # prompts for username + password
    python unsave_all_posts.py -u myname       # prompts only for password
    python unsave_all_posts.py --dry-run       # just list what would be unsaved

Notes:
  * Uses Instagram's unofficial private API (instagrapi). Automating your
    account is against Instagram's ToS and very fast bulk actions can get
    the account temporarily rate-limited, so the script sleeps between
    requests. Increase --delay if you get "feedback_required" / 429 errors.
  * Your session is cached in a local file so you don't log in again
    (and trigger security checks) every run. Delete it when you're done.
"""

import argparse
import getpass
import os
import random
import sys
import time

try:
    from instagrapi import Client
    from instagrapi.exceptions import (
        ChallengeRequired,
        FeedbackRequired,
        LoginRequired,
        PleaseWaitFewMinutes,
        TwoFactorRequired,
    )
except ImportError:
    sys.exit("instagrapi is not installed. Run:  pip install instagrapi")


def login(username: str, password: str, session_file: str) -> Client:
    cl = Client()
    cl.delay_range = [1, 3]

    if os.path.exists(session_file):
        try:
            cl.load_settings(session_file)
            cl.login(username, password)
            cl.get_timeline_feed()  # verify the session is still valid
            print("Logged in using saved session.")
            return cl
        except LoginRequired:
            print("Saved session expired, logging in fresh...")
            old = cl.get_settings()
            cl = Client()
            cl.set_uuids(old.get("uuids", {}))
            cl.delay_range = [1, 3]

    try:
        cl.login(username, password)
    except TwoFactorRequired:
        code = input("Enter your 2FA code: ").strip()
        cl.login(username, password, verification_code=code)
    except ChallengeRequired:
        sys.exit(
            "Instagram wants a security check. Open the Instagram app, approve "
            "the login ('This was me'), then run the script again."
        )

    cl.dump_settings(session_file)
    print("Logged in.")
    return cl


def iter_saved_posts(cl: Client):
    """Yield (media_id, shortcode) for every item in the 'Saved posts' feed."""
    max_id = None
    while True:
        params = {"include_igtv_preview": "false"}
        if max_id:
            params["max_id"] = max_id
        resp = cl.private_request("feed/saved/posts/", params=params)
        for item in resp.get("items", []):
            media = item.get("media") or {}
            if media.get("id"):
                yield media["id"], media.get("code", "?")
        if not resp.get("more_available") or not resp.get("next_max_id"):
            break
        max_id = resp["next_max_id"]


def main():
    ap = argparse.ArgumentParser(description="Unsave all saved Instagram posts.")
    ap.add_argument("-u", "--username", help="Instagram username")
    ap.add_argument("--dry-run", action="store_true", help="List posts without unsaving")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Base seconds to wait between unsaves (default 2)")
    ap.add_argument("--session-file", default="ig_session.json",
                    help="Where to cache the login session")
    args = ap.parse_args()

    username = args.username or input("Instagram username: ").strip()
    password = getpass.getpass("Instagram password: ")  # not echoed to screen

    cl = login(username, password, args.session_file)

    print("Fetching saved posts...")
    saved = list(iter_saved_posts(cl))  # collect first so pagination isn't disturbed
    print(f"Found {len(saved)} saved post(s).")
    if not saved:
        return

    if args.dry_run:
        for media_id, code in saved:
            print(f"  would unsave https://www.instagram.com/p/{code}/")
        return

    confirm = input(f"Unsave all {len(saved)} posts? Type 'yes' to continue: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    done = failed = 0
    for i, (media_id, code) in enumerate(saved, 1):
        try:
            cl.media_unsave(media_id)
            done += 1
            print(f"[{i}/{len(saved)}] unsaved {code}")
        except (FeedbackRequired, PleaseWaitFewMinutes) as e:
            print(f"Rate-limited by Instagram ({e.__class__.__name__}). "
                  f"Stopped after {done} unsaves; wait a while and re-run.")
            break
        except Exception as e:  # keep going on one-off errors
            failed += 1
            print(f"[{i}/{len(saved)}] failed {code}: {e}")
        time.sleep(args.delay + random.uniform(0, args.delay))

    print(f"\nDone. Unsaved: {done}, failed: {failed}.")
    print("Re-run the script if anything is left (Instagram sometimes pages lazily).")


if __name__ == "__main__":
    main()

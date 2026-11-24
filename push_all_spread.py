import subprocess
import os
import sys
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def push_all_spread(batch_size=30):
    print("Fetching remote state...")
    subprocess.run(["git", "fetch", "origin", "main"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get all unpushed commits in chronological order
    ahead = subprocess.check_output(
        ["git", "rev-list", "--reverse", "origin/main..HEAD"]
    ).decode().splitlines()

    if not ahead:
        print("Nothing to push - all commits are already on GitHub!")
        return

    total = len(ahead)
    days_needed = (total + batch_size - 1) // batch_size
    print(f"Found {total} unpushed commits.")
    print(f"Will spread as {batch_size}/day across {days_needed} days, then push all.\n")

    today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    # Step 1: Redate all commits
    print("=== STEP 1: Redating commits ===")
    current_parent = subprocess.check_output(
        ["git", "rev-parse", "origin/main"]
    ).decode().strip()

    new_tip = current_parent
    for i, commit_hash in enumerate(ahead):
        day_offset = i // batch_size
        minute_offset = i % batch_size
        date = today + timedelta(days=day_offset)
        date_str = date.strftime(f"%Y-%m-%dT09:{minute_offset:02d}:00")

        tree_hash = subprocess.check_output(
            ["git", "rev-parse", f"{commit_hash}^{{tree}}"]
        ).decode().strip()

        msg = subprocess.check_output(
            ["git", "log", "--format=%B", "-1", commit_hash]
        ).decode().strip()

        author_name  = subprocess.check_output(["git", "log", "--format=%an", "-1", commit_hash]).decode().strip()
        author_email = subprocess.check_output(["git", "log", "--format=%ae", "-1", commit_hash]).decode().strip()

        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"]     = author_name
        env["GIT_AUTHOR_EMAIL"]    = author_email
        env["GIT_AUTHOR_DATE"]     = date_str
        env["GIT_COMMITTER_NAME"]  = author_name
        env["GIT_COMMITTER_EMAIL"] = author_email
        env["GIT_COMMITTER_DATE"]  = date_str

        new_hash = subprocess.check_output(
            ["git", "commit-tree", tree_hash, "-p", current_parent, "-m", msg],
            env=env
        ).decode().strip()

        current_parent = new_hash
        new_tip = new_hash

        day_label = date.strftime("%b %d")
        if (i + 1) % 30 == 0 or i == total - 1:
            print(f"  Redated {i+1}/{total} commits... (up to {day_label})")

    # Reset local branch to redated history
    print("\nUpdating local branch...")
    subprocess.run(["git", "reset", "--hard", new_tip],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Redate complete!\n")

    # Step 2: Push all at once in batches (to avoid timeout)
    print("=== STEP 2: Pushing all commits to GitHub ===")
    pushed_total = 0
    batch_num = 0

    while True:
        subprocess.run(["git", "fetch", "origin", "main"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        remaining = subprocess.check_output(
            ["git", "rev-list", "--reverse", "origin/main..HEAD"]
        ).decode().splitlines()

        if not remaining:
            break

        batch_num += 1
        batch = remaining[:batch_size]
        target = batch[-1]

        day_date = subprocess.check_output(
            ["git", "log", "--format=%ad", "--date=short", "-1", target]
        ).decode().strip()

        print(f"  Pushing batch {batch_num} ({len(batch)} commits, dated {day_date})...")
        res = subprocess.run(
            ["git", "push", "origin", f"{target}:refs/heads/main"],
            capture_output=True
        )

        if res.returncode != 0:
            print(f"  Push failed: {res.stderr.decode()}")
            break

        pushed_total += len(batch)
        print(f"  Pushed! Total pushed so far: {pushed_total}/{total}")

    print(f"\n=== ALL DONE! ===")
    print(f"Pushed {pushed_total} commits across {batch_num} days.")
    print(f"Your GitHub graph will show 30 contributions/day from today onwards!")

if __name__ == "__main__":
    push_all_spread(batch_size=30)

import subprocess
import os
from datetime import datetime, timedelta

def redate_all_unpushed(batch_size=30):
    # Fetch latest remote state
    print("Fetching remote state...")
    subprocess.run(["git", "fetch", "origin", "main"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get all unpushed commits in chronological order
    ahead = subprocess.check_output(
        ["git", "rev-list", "--reverse", "origin/main..HEAD"]
    ).decode().splitlines()

    if not ahead:
        print("No unpushed commits found. Everything is already on GitHub!")
        return

    total = len(ahead)
    print(f"Found {total} unpushed commits to redate.")
    print(f"Spreading them as {batch_size} commits/day starting TODAY...\n")

    today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    # Get base commit (origin/main HEAD)
    current_parent = subprocess.check_output(
        ["git", "rev-parse", "origin/main"]
    ).decode().strip()

    # Rebuild each commit with a new date
    for i, commit_hash in enumerate(ahead):
        day_offset = i // batch_size
        minute_offset = i % batch_size  # spread within the day (9:00, 9:01, ..., 9:29)
        date = today + timedelta(days=day_offset)
        date_str = date.strftime(f"%Y-%m-%dT09:{minute_offset:02d}:00")

        # Get tree hash
        tree_hash = subprocess.check_output(
            ["git", "rev-parse", f"{commit_hash}^{{tree}}"]
        ).decode().strip()

        # Get commit message
        msg = subprocess.check_output(
            ["git", "log", "--format=%B", "-1", commit_hash]
        ).decode().strip()

        # Get author info
        author_name  = subprocess.check_output(["git", "log", "--format=%an", "-1", commit_hash]).decode().strip()
        author_email = subprocess.check_output(["git", "log", "--format=%ae", "-1", commit_hash]).decode().strip()

        # Build new commit with overridden dates
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

        day_label = (today + timedelta(days=day_offset)).strftime("%b %d")
        print(f"[{i+1:4d}/{total}] {commit_hash[:7]} -> {new_hash[:7]}  ({day_label})")

    # Reset local main branch to the new commit chain
    print(f"\nUpdating local branch to redated history...")
    subprocess.run(["git", "reset", "--hard", current_parent], check=True)

    days_needed = (total + batch_size - 1) // batch_size
    print(f"\n✅ Done! All {total} commits have been redated.")
    print(f"   Today ({today.strftime('%b %d')}): 30 commits ready to push")
    print(f"   Total days needed: {days_needed} days")
    print(f"\nNow run:  python push_daily.py")
    print("Run that command once every day to push today's 30 commits!")

if __name__ == "__main__":
    redate_all_unpushed(batch_size=30)

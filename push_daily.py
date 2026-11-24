import subprocess
import sys

def push_daily_batch(batch_size=30):
    # Fetch existing commits on remote main if any
    try:
        remote_head = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], stderr=subprocess.DEVNULL
        ).decode().strip()
        # Find how many commits we are ahead
        ahead_commits = subprocess.check_output(
            ["git", "rev-list", "--reverse", "origin/main..HEAD"]
        ).decode().splitlines()
    except Exception:
        # Remote doesn't have main yet, start from root
        ahead_commits = subprocess.check_output(
            ["git", "rev-list", "--reverse", "HEAD"]
        ).decode().splitlines()

    total_ahead = len(ahead_commits)
    if total_ahead == 0:
        print("Everything is up to date! No unpushed commits.")
        return

    print(f"Total unpushed commits remaining: {total_ahead}")
    batch = ahead_commits[:batch_size]
    target_commit = batch[-1]
    batch_count = len(batch)

    print(f"Pushing today's batch of {batch_count} commits up to commit {target_commit[:7]}...")
    res = subprocess.run(["git", "push", "origin", f"{target_commit}:refs/heads/main"])
    if res.returncode == 0:
        remaining = total_ahead - batch_count
        print(f"\nSuccessfully pushed {batch_count} commits today!")
        print(f"Remaining commits for future daily batches: {remaining}")
    else:
        print(f"Push failed with exit code {res.returncode}")

if __name__ == "__main__":
    size = 30
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            pass
    push_daily_batch(size)

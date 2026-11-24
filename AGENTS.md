# Standing Constraint: Daily Git Commit Spreading (30 Commits/Day)

## Permanent Rule for All Projects (New or Existing)

Whenever the user opens, creates, or works on **any** repository/project:

1. **Commit Spreading Rule**:
   - Commits must **never** be pushed all on a single day.
   - All commits must be spread across days at a rate of **30 commits per day**, starting from the current date.
   - For every commit, both `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` must match the designated day and staggered times (e.g., 09:00 - 18:00) so that GitHub's contribution graph accurately turns green with 30 contributions per day.

2. **Standard Scripts**:
   Ensure the following utility scripts are present and functional in every project root:
   - `push_daily.py` (pushes today's 30 commits)
   - `redate_commits.py` (redates unpushed commits starting from today at 30/day)
   - `push_all_spread.py` (one-shot: redates unpushed commits across days and pushes all batches to remote)

3. **Platform Safeguards**:
   - Use `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` and ASCII labels (`[DONE]`, `->`) instead of raw Unicode emojis or symbols to avoid `cp1252` encoding crashes on Windows PowerShell.
   - Author / GitHub user credentials:
     - User: **Xaelvoryx**
     - Git Email: `gobinathcse2023@ksrce.ac.in` (or secondary `czeanovats@gmail.com`)

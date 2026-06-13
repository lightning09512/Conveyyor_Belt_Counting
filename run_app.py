import subprocess
import sys
import time
from conveyor_counter.app import main


def auto_commit():
    try:
        # Check if there are unstaged or staged changes
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if res.returncode == 0:
            changes = res.stdout.strip()
            if changes:
                print("Detecting code changes. Automatically committing...")
                # Add all tracked changes and non-ignored untracked files
                subprocess.run(["git", "add", "."])
                commit_msg = f"auto-commit: code changes at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                subprocess.run(["git", "commit", "-m", commit_msg])
                print("Auto-committed successfully!")
    except Exception as e:
        print(f"Auto-commit warning: {e}", file=sys.stderr)


if __name__ == "__main__":
    auto_commit()
    main()

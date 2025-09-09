#!/usr/bin/env python3
# scripts/hooks/check_issue_reference.py

import subprocess
import re
import sys

def check_issue_reference():
    """Ensure branch name contains issue number."""
    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True
        )
        branch = result.stdout.strip()
        
        # Skip if on main
        if branch in ["main", "master", "develop"]:
            return 0
        
        # Check for issue number in branch name
        pattern = r"^(feature|bugfix|perf|security|docs|chore)/\d+-"
        if not re.match(pattern, branch):
            print(f"❌ Error: Branch '{branch}' doesn't follow naming convention")
            print("   Expected: <type>/<issue>-<description>")
            print("   Example: feature/123-add-neural-engine")
            return 1
        
        # Extract issue number
        issue_match = re.search(r"/(\d+)-", branch)
        if issue_match:
            issue_number = issue_match.group(1)
            print(f"✅ Branch linked to issue #{issue_number}")
        
        return 0
        
    except Exception as e:
        print(f"Warning: Could not check branch name: {e}")
        return 0

if __name__ == "__main__":
    sys.exit(check_issue_reference())
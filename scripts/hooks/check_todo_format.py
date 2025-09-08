#!/usr/bin/env python3
# scripts/hooks/check_todo_format.py

import sys
import re

def check_todo_format(filenames):
    """Ensure TODOs follow format: TODO(username): description (#issue)"""
    pattern = re.compile(r'#\s*(TODO|FIXME|XXX|HACK|NOTE)')
    correct_pattern = re.compile(r'#\s*(TODO|FIXME)\([a-zA-Z0-9_-]+\):\s+.+\s+\(#\d+\)')
    
    errors = []
    
    for filename in filenames:
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    match = pattern.search(line)
                    if match:
                        if not correct_pattern.search(line):
                            errors.append(f"{filename}:{line_num}: {line.strip()}")
        except (UnicodeDecodeError, IOError):
            # Skip files that can't be read as text
            continue
    
    if errors:
        print("❌ Invalid TODO format found:")
        print("   Expected: TODO(username): description (#issue)")
        for error in errors:
            print(f"   {error}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(check_todo_format(sys.argv[1:]))
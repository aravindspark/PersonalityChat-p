import os
import regex
import sys


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python get_scores.py <filename>")
        sys.exit(1)

    fname = sys.argv[1]
    file = os.path.join(os.getcwd(), fname)
    # pattern = r'^\|\s[A-Za-z]*\s+\|\s+\d\.?\d+?\s+\|\s+\d\.?\d{2}?\s+\|'
    pattern = r'^\|\s[A-Za-z]*\s+\|\s+([0-9]\.?[0-9]*?)\s+\|\s+([0-9]\.?[0-9]*?)\s+\|'
    scores = regex.compile(pattern)

    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        match_pattern = scores.match(line)
        if match_pattern:
            avg = match_pattern.group(1)
            std = match_pattern.group(2)
            print(f"& ${avg}\pm{std}$ ", end=' ')

    print()


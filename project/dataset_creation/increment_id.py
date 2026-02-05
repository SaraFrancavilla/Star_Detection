#!/usr/bin/env python3
import sys
import random

def increment_id(current_id):
    if len(current_id) != 3 or not current_id.isupper():
        raise ValueError(f"Invalid ID: '{current_id}' (must be 3 uppercase letters)")

    # Convert letters to numbers 0..25
    a, b, c = [ord(ch) - ord('A') for ch in current_id]

    # Increment with carry
    c += 1
    if c > 25:
        c = 0
        b += 1
    if b > 25:
        b = 0
        a += 1
    if a > 25:
        a = 0  # optional wrap-around after ZZZ

    # Convert back to letters
    return f"{chr(a + ord('A'))}{chr(b + ord('A'))}{chr(c + ord('A'))}"


def generate_angles():
    """Generate RA, DE, ROLL with desired distributions"""
    ra = round(random.uniform(5, 355), 2)          # avoid edges
    de = round(max(min(random.gauss(0, 25), 89), -89), 2)  # Gaussian, clamped
    roll = round((random.uniform(0, 360)), 2)  # offset from RA
    return ra, de, roll

if __name__ == "__main__":
    # Decide which function to call based on argument
    if len(sys.argv) < 2:
        print("Usage: python3 increment_and_angles.py [increment_id|angles] [CURRENT_ID]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "increment_id":
        if len(sys.argv) != 3:
            print("Usage: python3 increment_and_angles.py increment_id CURRENT_ID")
            sys.exit(1)
        current_id = sys.argv[2]
        print(increment_id(current_id))

    elif cmd == "angles":
        ra, de, roll = generate_angles()
        print(f"{ra} {de} {roll}")

    else:
        print("Unknown command. Use 'increment_id' or 'angles'.")
        sys.exit(1)
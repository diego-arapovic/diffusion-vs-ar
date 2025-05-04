#!/usr/bin/env python3
"""
Usage:
    python3 generate_longer_cd.py <input_file> >> <output_file>
"""

import json
import random
import sys
import math


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def choose_op_and_number(original_last: int):
    """Pick an operator once, then sample only the number until a valid result emerges."""
    op = random.choice(["+", "-", "*", "/"])

    # If division is hopeless (prime or 0), fall back to another operator once
    if op == "/" and (original_last == 0 or is_prime(original_last)):
        op = random.choice(["+", "-", "*"])

    if op == "+":
        if original_last >= 100:
            return -1, op, -1
        n = random.randint(1, 100-original_last)
        return n, op, n + original_last

    if op == "*":
        if original_last >= 22:
            return -1, op, -1
        n = random.randint(1, 100//original_last)
        return n, op, n * original_last

    if op == "-":
        n = random.randint(1, 100)
        return n, op, original_last - n
    
    n = random.randint(1, original_last)
    while original_last % n != 0:
        n = random.randint(2, original_last-1)
    return n, op, original_last // n


def process_line(line: str) -> str:
    record = json.loads(line)
    input_numbers = [int(x) for x in record["input"].split(",")]
    original_last = input_numbers[-1]

    new_val = -1
    while new_val == -1:
        new_val, operator, computed = choose_op_and_number(original_last)

    insertion_index = random.randint(1, len(input_numbers) - 1)
    input_numbers.insert(insertion_index, new_val)

    if operator in ["+", "*"]:
        if random.choice([True, False]):
            new_equation = f"{new_val}{operator}{original_last}={computed}"
        else:
            new_equation = f"{original_last}{operator}{new_val}={computed}"
    elif operator == "-":
        if original_last - new_val >= 0:
            new_equation = f"{original_last}{operator}{new_val}={computed}"
        else:
            new_equation = f"{new_val}{operator}{original_last}={-computed}"
    elif operator == "/":
        new_equation = f"{original_last}{operator}{new_val}={computed}"

    record["output"] = record["output"].rstrip(",") + "," + new_equation

    if computed >= 0:
        input_numbers[-1] = computed
    else:
        input_numbers[-1] = -computed
    record["input"] = ",".join(map(str, input_numbers))

    return json.dumps(record)


def main():
    input_file = sys.argv[1]
    with open(input_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                print(process_line(line))


if __name__ == "__main__":
    main()

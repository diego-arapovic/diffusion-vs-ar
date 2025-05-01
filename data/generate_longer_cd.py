#!/usr/bin/env python3
"""
Usage:
    python3 generate_longer_cd.py <input_file> >> <output_file>

Description:
    Reads a JSONL file where each line is a JSON object with "input" and "output" fields.
    Inserts a random number into the "input" list, modifies the last value using + or *,
    updates the "input" and "output" accordingly, and prints each modified record as JSON.
"""

import json
import random
import sys

def process_line(line):
    record = json.loads(line)
    
    input_numbers = [int(x) for x in record["input"].split(",")]
    
    original_last = input_numbers[-1]
    
    new_val = random.randint(0, 100)
    
    insertion_index = random.randint(1, len(input_numbers) - 1)
    input_numbers.insert(insertion_index, new_val)
    
    operator = random.choice(["+", "*"])
    if operator == "+":
        computed = new_val + original_last
    else:
        computed = new_val * original_last
    
    new_equation = f"{new_val}{operator}{original_last}={computed}"
    record["output"] = record["output"].rstrip(",") + "," + new_equation
    
    input_numbers[-1] = computed
    record["input"] = ",".join(str(num) for num in input_numbers)
    
    return json.dumps(record)

def main():    
    input_file = sys.argv[1]
    with open(input_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                updated_line = process_line(line)
                print(updated_line)

if __name__ == "__main__":
    main()

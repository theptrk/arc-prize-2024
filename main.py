import json

import os

print(os.listdir('./'))

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Reading the JSON files
training_challenges_file = './arc-agi_training_challenges.json'
training_solutions_file = './arc-agi_training_solutions.json'

tc_data = read_json_file(training_challenges_file)
ts_data = read_json_file(training_solutions_file)

# Printing the contents of the JSON files
print("Contents of tc:")
print(json.dumps(tc_data, indent=4))

print("\nContents of ts")
print(json.dumps(ts_data, indent=4))


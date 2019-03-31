import csv

with open('lyrics.csv') as f:
    a = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
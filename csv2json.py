# code to convert csv data to json file

import csv
import json

# open the file
with open('./arxiv_abstracts.csv', 'r', encoding='utf-8') as f:
    # create a csv reader object
    reader = csv.reader(f)
    # read the first row
    row = next(reader)
    # create a list to store the data
    data = []
    # iterate over the rows
    i = 0
    for row in reader:
        # create a dictionary to store the data
        d = {
            'id': i,
            # 'title': row[1],
            'text': row[2]
        }
        i += 1
        # append the dictionary to the list
        data.append(d)

# write the data to a json file
with open('./arxiv_abstracts.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

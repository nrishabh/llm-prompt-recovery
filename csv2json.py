# code to convert csv data to json file

import csv
import json

# open the file
# with open('./arxiv_abstracts.csv', 'r', encoding='utf-8') as f:
# with open('./song_lyrics.csv', 'r', encoding='utf-8') as f:
# with open('./Resume.csv', 'r', encoding='utf-8') as f:
# with open('./recipes_data.csv', 'r', encoding='utf-8') as f:
with open('./gemma100.csv', 'r', encoding='utf-8') as f:
    # create a csv reader object
    reader = csv.reader(f)
    # read the first row
    row = next(reader)
    # create a list to store the data
    data = []
    # id = 0
    for i, row in enumerate(reader):
        # create a dictionary to store the data
        # Songs Dataset
        # d = {
        #     'id': i,
        #     'text': row[6]
        # }
        # Arxiv Abstracts Dataset
        # d = {
        #     'id': i,
        #     'text': row[2]
        # }
        # Resume Dataset
        # d = {
        #     'id': i,
        #     'text': row[1]
        # }
        # Recipe Dataset
        # if len(row[2].split(' ')) > 100 and len(row[2].split(' ')) < 1500:
        #     id +=1
        #     d = {
        #         'id': id,
        #         'text': row[2]
        #     }
        # # append the dictionary to the list
        #     data.append(d)
        # if id == 10000:
        #     break

        # Prompts Dataset
        d = {
            'id': i,
            'prompt': row[1]
        }
        data.append(d)

# write the data to a json file
with open('./prompts100.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

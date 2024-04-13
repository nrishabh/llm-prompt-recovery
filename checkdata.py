# python code to print the first row of data from a csv file

import csv

# open the file
with open('./arxiv_abstracts.csv') as f:
    # create a csv reader object
    reader = csv.reader(f)
    # read the first row
    row = next(reader)

    # print the first 5 rows

    for i in range(5):
        row = next(reader)
        print(row[0])
        print(row[2])
        print(len(row[1].split(' ')))
        print('-------------------\n')
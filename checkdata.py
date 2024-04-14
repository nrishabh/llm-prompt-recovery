# python code to print the first row of data from a csv file

import csv

# open the file
# with open('./arxiv_abstracts.csv', 'r', encoding='utf-8') as f:
# with open('./song_lyrics.csv', 'r', encoding='utf-8') as f:
# with open('./resume.csv', 'r', encoding='utf-8') as f:
# with open('./recipes_data.csv', 'r', encoding='utf-8') as f:
with open('./gemma100.csv', 'r', encoding='utf-8') as f:
    # create a csv reader object
    reader = csv.reader(f)
    # read the first row
    row = next(reader)
    print(row)

    print('-------------------\n')
    print('-------------------\n')
    print('-------------------\n')

    # print the header names of the csv file

    # print the first 5 rows
    for i in range(5):
        row = next(reader)
        # print(row[0])
        print(row[1])
        print(len(row[1].split(' ')))
        print('-------------------\n')

    # count how many texts have length more than 500 and less than 1000
    # count = 0
    # for i, row in enumerate(reader):
    #     # if len(row[2].split(' ')) > 100 and len(row[2].split(' ')) < 1500:
    #     if len(row[1].split(' ')) < 30:
    #         count += 1
    #     # if i % 1000 == 0:
    #     #     print(i, count)
    #     # if count == 10000:
    #     #     break
    # print(count)

    # get max length of texts
    # max_len = 0
    # for i, row in enumerate(reader):
    #     print(row[1])
    #     print(len(row[1].split(' ')))
    #     if len(row[1].split(' ')) > max_len:
    #         # print(row[1])
    #         # print(len(row[1].split(' ')))
    #         max_len = len(row[1].split(' '))
    #     # if i % 1000 == 0:
    #     #     print(i, max_len)
    # print(f"max_len: {max_len}")
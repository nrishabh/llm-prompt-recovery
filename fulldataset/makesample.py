import json

f=open('../ogtext/stdjson/shakespeare.json')
shakespeares=json.load(f)
f.close()
fulldataset=[]
n=0
for entry in shakespeares:
    thisdata={'id':n, 'input_text': entry['text'], 'prompt': "Explain to me like I'm five", 'output_text': entry['text']}
    fulldataset.append(thisdata)
    n+=1

output='./sample_full.json'
with open(output, "w") as outfile:
    json.dump(fulldataset, outfile, ensure_ascii=False)
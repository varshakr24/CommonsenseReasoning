import json
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
#filename = "Downloads/t.jsonl"
predfile = "model.10.bin.test"
csstrfile = "commongen.test.cs_str.txt"
preds = []
inp = []
with open(predfile,'r') as o:
	for line in o:
                this = []
                for word in line.strip().split(' '):
                    this.append(stemmer.stem(word))
                preds.append(this)

with open(csstrfile,'r') as o:
	for line in o:
                concepts = line.strip('\n').split('#')
                for i,c in enumerate(concepts):
                    concepts[i] = stemmer.stem(concepts[i][:-2])
                inp.append(concepts)
#print(len(inp),len(preds))
total = 0
missed = 0
for i, item in enumerate(inp):
        flag = False
        total +=1
        sent = preds[i]
        for c in item:
            if sent.count(c)>1:
                flag=True
        if flag==True:
            missed+=1
#        print(item, preds[i],flag)
print(missed/total)

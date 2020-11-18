import json
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
#from nltk.tag import pos_tag
#from nltk.tag.stanford import StanfordNERTagger
#

import spacy
nlp = spacy.load('en_core_web_sm')
 

stemmer = SnowballStemmer("english")
predfile = "Row1.test"
csstrfile = "commongen.test.cs_str.txt"
pred_words = []
pred_pos = []
inp_words = []
inp_pos = []


def match_pos(pos):
    if pos.startswith('J'):
        return "A"
    elif pos.startswith('V'):
        return "V"
    elif pos.startswith('N'):
        return "N"
    elif pos.startswith('R'):
        return "A"
    else:
        return None

with open(predfile,'r') as o:
	for line in o:
                stem_w = []
                pos_w = []
                doc = nlp(line)
                tokenized_line = word_tokenize(line)
                for i, word in enumerate(tokenized_line):
                    stem_w.append(stemmer.stem(word))
                for ent in doc:
                    pos_w.append(match_pos(ent.pos_))
                pred_words.append(stem_w)
                pred_pos.append(pos_w)


with open(csstrfile,'r') as o:
	for line in o:
                stem_w = []
                pos_w = []
                concepts = line.strip('\n').split('#')
                for c in concepts:
                    stem_w.append(stemmer.stem(c[:-2]))
                    pos_w.append(c[-1])
                inp_words.append(stem_w)
                inp_pos.append(pos_w)


total = 0
mismatch = 0

concept_matching_ind = []
for i, sample in enumerate(inp_words):
#    if i >= len(pred_words):
#        break
    flag = False
    total += 1
    for j, concept in enumerate(sample):
        matching_index = [ind for ind, x in enumerate(pred_words[i]) if x == concept]
        if matching_index:
            for ind in matching_index:
                if pred_pos[i][ind] != inp_pos[i][j]:
                    print(sample, concept, inp_pos[i][j])
                    print(pred_words[i])
                    print(pred_pos[i][ind])
                    mismatch += 1
                    flag = True
                    break
        if flag:
            break

print(mismatch/total)

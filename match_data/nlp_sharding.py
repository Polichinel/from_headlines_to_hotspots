import pandas as pd
import spacy
import sys
from functools import partial

nlp = spacy.load('en_core_web_sm')

try:
    file_id=sys.argv[1]
except:
    raise ValueError("You need to give a 0--213 integer")


try:
    if sys.argv[2] == 'test': test=True
except:
    test=False

def none_to_str(x):
    if x is None:
        return ''
    else:
        return x


def named_entities(row):
    if row.cur_id % 100 == 0:
        print (row.cur_id, end=",")
    parsed = nlp(none_to_str(row.title) + ' ' +
                 none_to_str(row.snippet) + ' ' +
                 none_to_str(row.body))
    return list(set([(ent.text,ent.label_) for ent in parsed.ents]))

fname = f"shards/shard_{file_id}"
print (fname)

df = pd.read_parquet(fname)

if test:
    df=df.iloc[1:402]

df['ner']=df.apply(named_entities,1)
df.to_parquet(f"shards_out/shard_{file_id}.parquet")

print("\n")

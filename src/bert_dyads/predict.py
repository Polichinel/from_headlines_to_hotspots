print("Starting script")

import pandas as pd

print("Loaded pandas")

import datasets
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import evaluate

print("Loaded torch and huggingface")

from scipy.special import softmax
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

print("Loaded transformer utils")

from torch.utils.data import DataLoader
from torch.optim import AdamW

import click

print("Done. If curious, these prints are to debug speed of spyder, that is underwhelming")


@click.command()
@click.option('--seed',default=421337, help='MonteCarlo Iteration ID.')
@click.option('--shard_id',default=0, help='Data Shard')


def read_opt(seed, shard_id):
    """Read options for training."""
    print(f"{seed=}, {shard_id=}")
    return seed, shard_id


seed,shard_id = read_opt(standalone_mode=False)
#torch.use_deterministic_algorithms(True)

# Convenience area
np.random.seed(seed)
torch.manual_seed(seed)

path = '/mimer/NOBACKUP/groups/uppstore2019113/'
huggingface_name='/mimer/NOBACKUP/groups/uppstore2019113/hf_cache/conflibert'

np.random.seed(seed=seed)
torch.manual_seed(seed)

try:
    del model
except:
    pass

try:
    del trainer
except:
    pass

torch.cuda.empty_cache()
mps_device = torch.device("cuda")
path = path.strip(' ').rstrip('/')

tokenizer = AutoTokenizer.from_pretrained(huggingface_name)

model_object = torch.load(f'{path}/out_bert/conflibert_{seed}_restart_cos_29.pt')
model = model_object['model']

shard = pd.read_parquet(f'{path}/in_shards/ner_shard_{shard_id}.parquet')
shard['text']=shard.title+' '+shard.snippet+' '+shard.body
shard.loc[shard.text.isna(),'text']=shard.title
shard.loc[shard.text.isna(),'text']='[UNK]'

shard = shard.reset_index()

out_data = []
for i in range(10000):
    
    if i%250==0: print(i,end=',')
    
    row = shard.iloc[i]
    #print (row.text)
    art = tokenizer(row.text, return_tensors='pt',
               padding="max_length", truncation=True, max_length=500)
    art = {k: v.to(mps_device) for k, v in art.items()}
    
    with torch.no_grad():
        outputs_stage_1 = model.bert(**art)
        bert_embed_pool = outputs_stage_1[-1]
        logits = model.classifier(bert_embed_pool)

    bert_embed_max = outputs_stage_1.last_hidden_state.max(1).values.cpu()
    bert_embed_mean = outputs_stage_1.last_hidden_state.max(1).values.cpu()
    
    bert_embed_pool = bert_embed_pool.cpu()
    
    probas = torch.nn.functional.softmax(logits, dim=-1)
    top5 = torch.topk(probas,5).indices.cpu().tolist()[0]
    top5_val = torch.topk(probas,5).values.cpu().tolist()[0]
    top5_iter = iter(top5_val)
    top5 = [(model_object['id2class'][i], next(top5_iter)) for i in top5]
    
    out_data += [{'id':row['index'],
     'an':row.an,
     'dyad_id':row.dyad_id, 
     'date':row.date_end, 
     'ged_id':row.ged_id, 
     'pred_dyad': top5[0],
     'pred_proba': top5_val[0],
     'text':row.text,
     'bert_embed_max':bert_embed_max,
     'bert_embed_mean': bert_embed_mean,
     'bert_embed_pool':bert_embed_pool,
     'k5_dyad': top5,
     'k5_proba': top5_val}]
    
torch.save(out_data, f'/mimer/NOBACKUP/groups/uppstore2019113/preds_conflibert/shard_{shard_id}_{seed}.pt')

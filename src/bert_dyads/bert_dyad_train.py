
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import evaluate
from scipy.special import softmax

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

import wandb

import click


@click.command()
@click.option('--seed',default=421337, help='MonteCarlo Iteration ID.')
@click.option('--grouping_var',default='dyad_id', help='Grouping Var')
@click.option('--huggingface_name',default='conflibert', help='Model')
@click.option('--snippet', default=False, help='Tokenize the full article or just head/snippet (lead)?')
@click.option('--data_path', default='out_data/pos_collect.parquet', help='Data Path')
@click.option('--num_epochs', default=30, help='Number of epochs?')
@click.option('--scheduler', default='restart_cos', help='Scheduler? {cos,lin,restart_cos}')
@click.option('--learning_rate', default=5e-5, help='Learning Rate?')
@click.option('--aie', default=20, help='How many articles for a dyad in epoch?')



def read_opt(seed, grouping_var, huggingface_name, snippet, data_path, num_epochs, scheduler, learning_rate, aie):
    """Read options for training."""
    print(f"{seed=}, {grouping_var=}, {huggingface_name=}, {snippet=}, {num_epochs=}, {scheduler=}, {learning_rate=}, {aie=}")
    return seed, grouping_var, huggingface_name, snippet, data_path, num_epochs, scheduler, learning_rate, aie


seed,grouping_var,huggingface_name,snippet,data_path,num_epochs,scheduler,learning_rate,aie = read_opt(standalone_mode=False)
#torch.use_deterministic_algorithms(True)

# Convenience area
np.random.seed(seed)
torch.manual_seed(seed)

fraction_in_train = .8

if huggingface_name.lower()=='bert': huggingface_name='/mimer/NOBACKUP/groups/uppstore2019113/hf_cache/bert-base-uncased'
if huggingface_name.lower()=='conflibert': huggingface_name='/mimer/NOBACKUP/groups/uppstore2019113/hf_cache/conflibert'
if huggingface_name.lower()=='roberta': huggingface_name='/mimer/NOBACKUP/groups/uppstore2019113/hf_cache/roberta'
if huggingface_name.lower()=='distilbert': huggingface_name='/mimer/NOBACKUP/groups/uppstore2019113/hf_cache/distilbert'

#Give me a model name to use for file names, tracking etc.
base_mod_name = huggingface_name.split('/')[-1].split('-')[0].lower()


df_rel = pd.read_parquet(data_path)

wandb.init(
    # set the wandb project where this run will be logged
    project="escalation_bert",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "huggingface_model": base_mod_name,
    "obs_per_epoch":aie,
    "epochs": num_epochs,
    "snippet_only": snippet,
    "seed": seed
    }
)

#Dataprep
if snippet:
    df_rel['text']=df_rel.title+' '+df_rel.snippet
    base_mod_name = 'snip_'+base_mod_name
    print("SNIPPET ONLY")
else:
    df_rel['text']=df_rel.title+' '+df_rel.snippet+' '+df_rel.body


cntr=df_rel[['title','dyad_name']].groupby('dyad_name').count()
cntr = cntr[cntr.title<30].reset_index()
govs = list(cntr[cntr.dyad_name.str.contains('Government')].dyad_name)
non_govs = list(cntr[~cntr.dyad_name.str.contains('Government')].dyad_name)
    
    
df_rel_sm = df_rel[['dyad_name','text','dyad_id']]
df_rel_sm.loc[df_rel_sm.dyad_name.isin(govs),'dyad_name'] = 'Other Governmental Conflict'
df_rel_sm.loc[df_rel_sm.dyad_name.isin(non_govs),'dyad_name'] = 'Other Nonstate Conflict'
df_rel_sm.loc[df_rel_sm.dyad_name == 'Other Governmental Conflict', 'dyad_id'] = 10000
df_rel_sm.loc[df_rel_sm.dyad_name == 'Other Nonstate Conflict', 'dyad_id'] = 10001
df_rel_sm['dyad_id']=df_rel_sm['dyad_id'].astype('int')


# Initialization
np.random.seed(seed=seed)
torch.manual_seed(seed)

# Clean up any weights stored on GPU and make sure retraining doesn't happen on top 
# Of already trained data.

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


tokenizer = AutoTokenizer.from_pretrained(huggingface_name)

def preprocess_function(examples):
    """
    Tokenize text using the chosen Huggingface tokenizer
    The tokenizer is a superglobal.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)

def df_to_torch(df, shuffle=True, balance=False):
    """
    Given a Pandas Dataframe containing processed UCDP data, 
    return a torch DataLoader that has been processed accordingly to 
    :shuffle - randomize the data, for training purposes. Do not use with RNN/LSTM models
    :balance - balance the data to include reasonable numbers of per-class texts
    returns a torch Dataloader with the same data
    """    
    
    if balance:
        df=df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), aie))).sample(frac=1)
        
    print(df.head())
        
    r_t = Dataset.from_pandas(df)
    r_t = r_t.map(preprocess_function, batched=True)
    try:
        r_t = r_t.remove_columns(["text"])
    except:
        pass
    
    try:
        r_t = r_t.remove_columns(["__index_level_0__"])
    except:
        pass

    
    try:
        r_t = r_t.rename_column("label", "labels")
    except:
        pass
    r_t.set_format("torch")
    r_t = DataLoader(r_t, shuffle=shuffle, batch_size=8)
    return r_t


def alt_get_predictions(dataloader):
    """
    Given a dataloader and a superglobal trained torch huggingface module
    Predicts logits from
    """
    gather_logits = torch.empty(0,device=mps_device)
    gather_labels = torch.empty(0,device=mps_device)

    progress_bar_x = tqdm(range(int(len(dataloader.dataset)/dataloader.batch_size)))

    for batch in dataloader:
            batch = {k: v.to(mps_device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits

            gather_logits = torch.cat([gather_logits,logits],dim=0)
            gather_labels = torch.cat([gather_labels,batch["labels"]],dim=0)

            progress_bar_x.update(1)
            
    logits = gather_logits.cpu()
    labels = np.array(gather_labels.cpu())
    predictions = torch.argmax(logits, dim=-1)
    
    
    return labels, logits, predictions.cpu()

def alt_evaluate(dataloader):
    metrics =[
              evaluate.load("f1"), 
              evaluate.load("recall"),
              evaluate.load("precision"),
              evaluate.load("accuracy")
             ]
    #evaluate.combine is ridiculously buggy

    #metric = evaluate.load("accuracy")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(mps_device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        for metric in metrics:
            metric.add_batch(predictions=predictions, references=batch["labels"])#, average="micro")

    results = []
    for metric in metrics:
        if metric.name!='accuracy':
            results += [metric.compute(average="macro")]

        else:
            res = metric.compute()
            results += [res]

    return results

rel = df_rel_sm[['text','dyad_name']].rename(columns={'dyad_name':'label'})
rel['text']=rel.text.str.lower()
classes = [class_ for class_ in list(rel.label.unique())]
class2id = {class_:id for id, class_ in enumerate(classes)}
id2class = {id:class_ for class_, id in class2id.items()}
print (f"Data loaded with total size: {rel.shape[0]}")
rel['label']=rel['label'].apply(lambda x:class2id[x])
rel['label']

r_train = rel.sample(frac=fraction_in_train)
r_test = rel.loc[~rel.index.isin(r_train.index)].sample(10000)

test_dataloader = df_to_torch(r_test, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_name, id2label=id2class, label2id=class2id, ignore_mismatched_sizes=True
)

# Set up the experiment for re-training and send to GPU

optimizer = AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * int(1000/8)+2

model.to(mps_device)

for epoch in range(num_epochs):
    
    train_dataloader = df_to_torch(r_train, shuffle=True, balance=True)
    
    lr_scheduler = None
    
    if scheduler.lower().strip() == 'cos':
         lr_scheduler = get_scheduler(
         name="cosine_with_restarts", 
         optimizer=optimizer, 
         num_warmup_steps=10,
         num_training_steps=int(train_dataloader.dataset.num_rows/8)+2
    )
            
    if scheduler.lower().strip() == 'restart_cos':
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            #name="cosine_with_restarts", 
            optimizer=optimizer, 
            num_warmup_steps=10,
            num_cycles = 5,
            num_training_steps=int(train_dataloader.dataset.num_rows/8)+2
        )
        
    if scheduler.lower().strip() == 'lin':
        lr_scheduler = get_scheduler(name="linear", 
                                     optimizer=optimizer, 
                                     num_warmup_steps=0, 
                                     num_training_steps=int(train_dataloader.dataset.num_rows/8)+2
                                    )
        
    if scheduler is None: raise ValueError("No valid scheduler selected!")
    
    eval_res_cur = alt_evaluate(test_dataloader)
    # List of dicts to dict. Ugly as hell, but works.
    eval_res_cur = {list(i.keys())[0]:i[list(i.keys())[0]] for i in eval_res_cur}
    eval_res_cur['epoch']=epoch
    print(eval_res_cur)

    wandb.log(eval_res_cur)

    print (f"Epoch {epoch}:")
    model.train()
    progress_bar = tqdm(range(int(train_dataloader.dataset.num_rows/8)))
    
    for batch in train_dataloader:
        # Train
        # print (lr_scheduler.get_lr())
        batch = {k: v.to(mps_device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        lr_scheduler.step()
        
    if epoch%5==0 and epoch>1 and eval_res_cur['recall']>0.5:
        torch.save({'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'class2id': class2id,
            'id2class': id2class,
            'name': base_mod_name,
            'scheduler': scheduler,
            }, f'/mimer/NOBACKUP/groups/uppstore2019113/out_bert/checkpoint_s_{base_mod_name}_{seed}_{scheduler}_{epoch}.pt')

        
eval_res_final = alt_evaluate(test_dataloader)
# List of dicts to dict. Ugly as hell, but works.
eval_res_final = {list(i.keys())[0]:i[list(i.keys())[0]] for i in eval_res_final}

eval_res_final['epoch']=num_epochs
wandb.log(eval_res_final)

wandb.finish()


#if snippet:  base_mod_name = 'snippet_'+base_mod_name
torch.save({'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'class2id': class2id,
            'id2class': id2class,
            'name': base_mod_name,
            'scheduler': scheduler,
            }, f'/mimer/NOBACKUP/groups/uppstore2019113/out_bert/s_{base_mod_name}_{seed}_{scheduler}_{epoch}.pt')

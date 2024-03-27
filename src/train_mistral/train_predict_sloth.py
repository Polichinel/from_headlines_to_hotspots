from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
from jinja2 import Template
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
import re
from itertools import chain


max_seq_length = 2048

LAST_TRAINED_MONTH = 468

import click

print("Done import. If curious, these prints are to debug speed of module spider, that is underwhelming")


@click.command()
@click.option('--month_cur',default=500, help='MonteCarlo Iteration ID.')


def read_opt(month_cur):
    """Read options for training."""
    print(f"{month_cur=}")
    return month_cur


month_cur = read_opt(standalone_mode=False)

if month_cur < LAST_TRAINED_MONTH:
    raise ValueError("No in-sample inference!")

    
ESCALATION = {-5:'conflict termination',
              -4:'extreme de-escalation',
              -3:'substantial de-escalation',
              -2:'decrease',
              -1:'small reduction', 
               0:'no change',
               1:'small increase',
               2:'escalation',
               3:'substantial escalation',
               4:'extreme escalation',
               5:'conflict onset'}
keywords = ', '.join(list(ESCALATION.values()))

trained_model = '/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_trained_16/'
load_in_4bit = True

data_path='/mimer/NOBACKUP/groups/uppstore2019113/monthly'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = trained_model, # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
print(f'{tokenizer.add_bos_token=}, {tokenizer.add_eos_token=}')


train_data = torch.load(f'{data_path}/month_{month_cur-2}.pt')
train_data += torch.load(f'{data_path}/month_{month_cur-1}.pt')

test_data = torch.load(f'{data_path}/month_{month_cur}.pt')


def train_pos_event(event):

    event_prompt = f"""You are an assistant tasked with forecasting the risk of armed conflict between {event['dyad_name']}. Using the news article below describing a battle, estimate the risk of conflict escalation or de-escalation in the current month, three months from now and six months from now. Answer using only these keywords : {keywords}.
    """ + '''.\nAnswer as a JSON with the following format:
    {
    "fatalities": int,
    "escalation current month": str,
    "escalation next month": str,
    "escalation three months": str,
    "escalation six months": str
    }
    ### Article:
    ''' + event['text']
    
    pos_answer = "{"+f"""
        "fatalities": {int(event['event_best'])},
        "escalation current month": {ESCALATION[int(event['escalation_rolling3_nowcasting'])]},
        "escalation next month": {ESCALATION[int(event['escalation_rolling3_lead1'])]},
        "escalation three months": {ESCALATION[int(event['escalation_rolling3_lead3'])]},
        "escalation six months": {ESCALATION[int(event['escalation_rolling3_lead6'])]}
        """+"}"
    return event_prompt, pos_answer


def train_context_rag_event(event):
    rag_negs = ''
    size_x = 1+int(750/(len(event['rag_negatives'])+0.00000001))
    if size_x>1000: raise ValueError("No Rag")
    
    #print(size_x)
    for art in event['rag_negatives']:
        rag_negs += " ".join(art.replace("\n"," ").split(" ")[:size_x]) + " "

    event_prompt = f"""You are an assistant tasked with forecasting the risk of armed conflict between {event['dyad_name']}. Using the news article below describing the behavior of the groups and their context, estimate the risk of conflict escalation or de-escalation in the current month, three months from now and six months from now. Answer using only these keywords : {keywords}.
    """ + '''.\nAnswer as a JSON with the following format:
    {
    "escalation current month": str,
    "escalation next month": str,
    "escalation three months": str,
    "escalation six months": str
    }
    ### Article:
    ''' + rag_negs
    
    pos_answer = "{"+f"""
        "escalation current month": {ESCALATION[int(event['escalation_rolling3_nowcasting'])]},
        "escalation next month": {ESCALATION[int(event['escalation_rolling3_lead1'])]},
        "escalation three months": {ESCALATION[int(event['escalation_rolling3_lead6'])]},
        "escalation six months": {ESCALATION[int(event['escalation_rolling3_lead6'])]}
        """+"}"
    return event_prompt, pos_answer
        

def train_template(event_prompt, pos_answer):
    data = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "messages" : [
            {"role": "user", "content": event_prompt},
            {"role": "assistant", "content": pos_answer}
        ]
        }
    template = Template(tokenizer.chat_template)
    rendered_template = template.render(data)
    return rendered_template


def predict_template(event_prompt):
    data = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "messages" : [
            {"role": "user", "content": event_prompt}
        ]
        }
    template = Template(tokenizer.chat_template)
    rendered_template = template.render(data)
    return rendered_template



train_dataset = []
for event in tqdm(train_data):
    train_dataset += [{"text":train_template(*train_pos_event(event))}]
    try:
        train_dataset += [{"text":train_template(*train_context_rag_event(event))}]
    except ValueError:
        pass


train_dataset = Dataset.from_list(train_dataset)

run = wandb.init(
    project='simon-mihai-views', 
    job_type="training", 
    anonymous="allow"
)


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        fp16_full_eval = True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        #max_steps=60,
        output_dir = f"/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_out_month_{month_cur}/",
        save_strategy = "steps",
        save_steps = 20,
        optim = "adamw_8bit",
        seed = 9520,
    ),
)

trainer.train(resume_from_checkpoint = False)

def test_pos_event(event):
    event_prompt = f"""You are an assistant tasked with forecasting the risk of armed conflict between {event['dyad_name']}. Using the news article below describing a battle, estimate the risk of conflict escalation or de-escalation in the current month, three months from now and six months from now. Answer using only these keywords : {keywords}.
    """ + '''.\nAnswer as a JSON with the following format:
    {
    "fatalities": int,
    "escalation current month": str,
    "escalation next month": str,
    "escalation three months": str,
    "escalation six months": str
    }
    ### Article:
    ''' + event['text']
    
    return event_prompt


def test_context_rag_event(event, column='rag_negatives'):
    rag_negs = ''
    size_x = 1+int(15000/(len(event[column])+0.00000001))
    if size_x>16000: raise ValueError("No Rag")
    
    #print(size_x)
    for art in event[column]:
        rag_negs += " ".join(art.replace("\n"," ").split(" ")[:size_x]) + " "

    event_prompt = f'''Now, I will give you another article, containing context to the battle event above. 
    Answer the same question as above, in the same JSON format as above. 
  
    ### Article:
    ''' + rag_negs
    
    return event_prompt

def test_first_template(event_prompt):
    data = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "messages" : [
            {"role": "user", "content": event_prompt}
        ]
        }
    template = Template(tokenizer.chat_template)
    rendered_template = template.render(data)
    return rendered_template



def json_clean(text_out):
    text_out = (text_out[::-1].strip().split('{')[0]+'{')[::-1]
    return text_out.split('}')[0]+'}'.strip()

FastLanguageModel.for_inference(model)


event_set = []
for event in tqdm(test_data):
        render0 = test_first_template(test_pos_event(event))
        # [3:] is to get rid of the BOS, since this is appended to render0 in stage 2.
        try:
            render1 = test_first_template(test_context_rag_event(event))[3:]
        except:
            render1 = render0


        logging.set_verbosity(logging.CRITICAL)
        pipe2 = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=70,
                        return_full_text=False
                       )

        result0 = pipe2(render0)
        text_out = json_clean(result0[0]['generated_text'])

        result1 = pipe2(render0 + text_out + '</s>' + render1)
        text_out1 = json_clean(result1[0]['generated_text'])

        event['pred_no_rag']=text_out
        event['pred_rag']=text_out1
        event_set += [event]
        
torch.save(event_set, f'{data_path}/step_trained_month_{month_cur}.pt')


dyads_event_set = []
for dyad_id in tqdm(set(i['dyad_id'] for i in test_data)):
    subset_d = [j for j in test_data if dyad_id == j['dyad_id']]
    
    xdf = pd.DataFrame(subset_d).sort_values('jdate')
    
    subset_ragp = xdf.rag_positives
    subset_ragp = list(set(chain.from_iterable(subset_ragp)))
    
    subset_ragn = xdf.rag_negatives
    subset_ragn = list(set(chain.from_iterable(subset_ragn)))
    
    subset_text = list(xdf.text)
    
    dyad_event = {"text":' '.join(subset_text),
              "rag_positives":[' '.join(subset_ragp)],
              "rag_negatives":[' '.join(subset_ragn)],
              "dyad_id":dyad_id, "dyad_name":xdf.loc[0].dyad_name,
              "escalation_rolling3_nowcasting": xdf.loc[0].escalation_rolling3_nowcasting,
              "escalation_rolling3_lead1": xdf.loc[0].escalation_rolling3_lead1,
              "escalation_rolling3_lead3": xdf.loc[0].escalation_rolling3_lead3,
              "escalation_rolling3_lead6": xdf.loc[0].escalation_rolling3_lead6
             }
    
    render0 = test_first_template(test_pos_event(dyad_event))
    # [3:] is to get rid of the BOS, since this is appended to render0 in stage 2.
    try:
            render1 = test_first_template(test_context_rag_event(dyad_event))[3:]
    except:
            render1 = render0


    logging.set_verbosity(logging.CRITICAL)
    pipe2 = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=70,
                        return_full_text=False
                       )

    result0 = pipe2(render0)
    text_out = json_clean(result0[0]['generated_text'])

    result1 = pipe2(render0 + text_out + '</s>' + render1)
    text_out1 = json_clean(result1[0]['generated_text'])

    dyad_event['pred_no_rag']=text_out
    dyad_event['pred_rag']=text_out1
    dyads_event_set += [dyad_event]

torch.save(event_set, f'{data_path}/dyad_sets_{month_cur}.pt')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
from jinja2 import Template
from datasets import Dataset

from tqdm import tqdm
from unsloth import FastLanguageModel
import re

max_seq_length = 2048


LAST_TRAINED_MONTH = 468
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

trained_model = '/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_out/checkpoint-3900/'
load_in_4bit = True

data_path='/mimer/NOBACKUP/groups/uppstore2019113/monthly'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = trained_model, # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)


model.eval()

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
print(f'{tokenizer.add_bos_token=}, {tokenizer.add_eos_token=}')


def pos_event(event):
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


def context_rag_event(event):
    rag_negs = ''
    size_x = 1+int(750/(len(event['rag_negatives'])+0.00000001))
    if size_x>1000: raise ValueError("No Rag")
    
    #print(size_x)
    for art in event['rag_negatives']:
        rag_negs += " ".join(art.replace("\n"," ").split(" ")[:size_x]) + " "

    event_prompt = f'''Now, I will give you another article, containing context to the battle event above. 
    Answer the same question as above, in the same JSON format as above. 
  
    ### Article:
    ''' + rag_negs
    
    return event_prompt

def first_template(event_prompt):
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


for mid in range(505,517):
    print(mid)
    event_set = []
    test_data_month = torch.load(f'{data_path}/month_{mid}.pt')
    
    for event in tqdm(test_data_month):
        render0 = first_template(pos_event(event))
        # [3:] is to get rid of the BOS, since this is appended to render0 in stage 2.
        try:
            render1 = first_template(context_rag_event(event))[3:]
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
    torch.save(event_set, f'{data_path}/out_month_{mid}.pt')
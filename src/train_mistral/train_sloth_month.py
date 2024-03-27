from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
from jinja2 import Template
from datasets import Dataset

from tqdm import tqdm
from unsloth import FastLanguageModel

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

data_path='/mimer/NOBACKUP/groups/uppstore2019113/monthly'
month_id_start=422
month_id_end=468

in_data_month = []
for mid in range(month_id_start, month_id_end+1):
    print(mid, end=' ')
    in_data_month += torch.load(f'{data_path}/month_{mid}.pt')
print(' ')

test_data_month = []
for mid in range(month_id_end+1, month_id_end+12):
    print(mid, end=' ')
    test_data_month += torch.load(f'{data_path}/month_{mid}.pt')
print(' ')

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

base_model = "/mimer/NOBACKUP/groups/uppstore2019113/mistral"
new_model = "/mimer/NOBACKUP/groups/uppstore2019113/mistral7b-views-train_lora"
new_model_full = "/mimer/NOBACKUP/groups/uppstore2019113/mistral7b-train-tiny"
check_point = "/mimer/NOBACKUP/groups/uppstore2019113/mistral7b-views-train_checkpoints"

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
print(f'{tokenizer.add_bos_token=}, {tokenizer.add_eos_token=}')

#Mistral doesn't have true system prompts.

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
    
    pos_answer = "{"+f"""
        "fatalities": {int(event['event_best'])},
        "escalation current month": {ESCALATION[int(event['escalation_rolling3_nowcasting'])]},
        "escalation next month": {ESCALATION[int(event['escalation_rolling3_lead1'])]},
        "escalation three months": {ESCALATION[int(event['escalation_rolling3_lead6'])]},
        "escalation six months": {ESCALATION[int(event['escalation_rolling3_lead6'])]}
        """+"}"
    return event_prompt, pos_answer


def context_rag_event(event):
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


def prompt_template(event_prompt, pos_answer):
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

dataset = []
test_dataset = []

for event in tqdm(in_data_month):
    dataset += [{"text":prompt_template(*pos_event(event))}]
    try:
        dataset += [{"text":prompt_template(*context_rag_event(event))}]
    except ValueError:
        pass
    

for event in tqdm(test_data_month):
    test_dataset += [{"text":prompt_template(*pos_event(event))}]
    try:
        test_dataset += [{"text":prompt_template(*context_rag_event(event))}]
    except ValueError:
        pass

dataset = Dataset.from_list(dataset)
test_dataset = Dataset.from_list(test_dataset)

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
    train_dataset = dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        eval_accumulation_steps = 4,
        evaluation_strategy = "steps",
        eval_steps = 500,
        fp16_full_eval = True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        #max_steps=60,
        output_dir = "/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_out/",
        save_strategy = "steps",
        save_steps = 100,
        optim = "adamw_8bit",
        seed = 9520,
    ),
)

trainer.train(resume_from_checkpoint = True)

wandb.finish()
trainer.model.save_pretrained("/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_trained/")
model.save_pretrained_merged("/mimer/NOBACKUP/groups/uppstore2019113/mistral7b_4bit_trained_16/", 
                             tokenizer, save_method = "merged_16bit",)
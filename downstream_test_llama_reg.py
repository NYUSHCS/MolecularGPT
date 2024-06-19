import os
import sys
import time
import fire
# import gradio as gr
import torch
from datasets import load_dataset
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompter import Prompter
import numpy as np
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

from sklearn.metrics import (r2_score,
                             roc_auc_score)
import pandas as pd

def main(
    CLI: bool = False,
    protein: bool = False,
    load_8bit: bool = True,
    base_model: str = " ",  # path of llama2-7b-chat
    lora_weights: str = "", # path of lora weight ./ckpts
    prompt_template: str = "",  
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    path: str = "",  # test dataset path    e.g. ./esol/0-shot/
    shot: int = 0,   # k-shot inference  
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            #device_map="auto",
            device_map={"": 0},
            attn_implementation = "flash_attention_2"
        )
        
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0},
            )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    
    label_ignore = [-100]
    raw_label = {1: "Yes", 0: "No", 'invalid': label_ignore}
    label_y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[1])) # Not include CLS or other tokens
    label_n = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[0]))
     # input a list so that they can be concatenated in collator
    label_dict = {1: label_y, 0: label_n, 'invalid': label_ignore}        
    
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,truncation=True,max_length=4096,padding=False,return_tensors=None)

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 4096
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            # 'Only output Yes or No!!!'+data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
            # 'Only output Yes or No!!!'+ data_point["instruction"], data_point["input"]
        )

        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

            
        tokenized_user_prompt["labels"] = tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
 
        return tokenized_user_prompt 

    def evaluate(
        instruction,
        input=None,
        output=None,
        temperature=0.1,
        repetition_penalty=1,
        max_new_tokens=128,
        **kwargs,
    ):

        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt)

        inputs['labels'] = tokenizer(str(output))['input_ids']
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(device)
        
        do_sample=False

        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            logprobs = True,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            
            text = tokenizer.decode(generation_output[0][0])
            
            re=prompter.get_response(text)

            re=tokenizer(re)
            
            re=tokenizer.decode(re['input_ids'][1:-1])
            
        return float(output), float(re)

    DATASETS = [
        'esol',
        'freesolv',
        'lipo',
                ]

    for dataset in DATASETS:
        path = os.path.join(path, dataset)
        data = []
        data_score = []
        for f in os.listdir(path):
            path1 = os.path.join(path, f)
            
            data.append(f.split(".")[0])
            raw_datasets_val = load_dataset("json", data_files=path1)
            val_data = raw_datasets_val["train"]
            
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            labels_list = []
            predict_list = []
            for i, batch in enumerate(val_data):
                
                output, re = evaluate(batch['instruction'], batch['input'], batch['output'], temperature=0.1, repetition_penalty=1, max_new_tokens=128)
                # print(f'output={output} re={re}')
                labels_list.append(output)
                predict_list.append(re)
            score = (torch.tensor(labels_list)-torch.tensor(predict_list)).std()
            data_score.append(float(score))
            
            df = pd.DataFrame({'dataset':data,'score':data_score})
            df.to_csv(f'./cache/{dataset}_{shot}_new.csv')
            


if __name__ == "__main__":
    fire.Fire(main)

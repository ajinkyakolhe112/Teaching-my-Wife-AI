# PYTORCH MODEL
import transformers

model_name = "openai-community/gpt2"
tokenizer  = transformers.AutoTokenizer.from_pretrained(model_name)
model      = transformers.AutoModelForCausalLM.from_pretrained(model_name) 

# PEFT MODEL
import peft

lora_config = peft.LoraConfig(r=16)
peft_model  = peft.get_peft_model(model, lora_config)

import torchinfo
import torch

batch_size  = 1
seq_length  = 10
input_size  = (batch_size, seq_length)
dummy_input = torch.randint(low = 0, high = tokenizer.vocab_size, size = input_size)

torchinfo.summary(model=peft_model, input_data=dummy_input)
torchinfo.summary(model=model,      input_data=dummy_input)
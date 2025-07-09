from transformers import AutoModelForSeq2SeqLM

model_name_or_path     = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

import peft
peft_config = peft.LoraConfig(
    task_type=peft.TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = peft.get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282


from transformers import AutoModelForSeq2SeqLM

# ORIGINAL MODEL UNCHANGED
model     = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# PEFT MODEL - SEPERATE FROM ORIGINAL MODEL
peft_model_id = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM"
config        = peft.PeftConfig.from_pretrained(peft_model_id)
model         = peft.PeftModel.from_pretrained(model, peft_model_id)

model = model.to(device)
model.eval()
inputs = tokenizer("Tweet text : @HondaCustSvc Your customer service has been horrible during the recall process. I will never purchase a Honda again. Label :", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
# 'complaint'

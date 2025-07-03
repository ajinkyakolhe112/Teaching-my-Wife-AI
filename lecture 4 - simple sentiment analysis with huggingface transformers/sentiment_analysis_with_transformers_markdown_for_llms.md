# Converted from sentiment_analysis_with_transformers.ipynb - Markdown format optimized for LLM readability

## Pipeline

```
import transformers
```

```
from transformers import pipeline

classifier = pipeline(task = "sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.



    config.json:   0%|          | 0.00/629 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]


    Device set to use cuda:0





    [{'label': 'POSITIVE', 'score': 0.9598046541213989},
     {'label': 'NEGATIVE', 'score': 0.9994558691978455}]



```
classifier.task
```




    'sentiment-analysis'



```
classifier.model
```




    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): DistilBertSdpaAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )



```
classifier.model.device
```




    device(type='cuda', index=0)



```
# classifier is a complex variable. It has multiple variables inside it. 

class Complex_Variable:
    def __init__(self):
        self.var1 = 0
        self.var2 = 0
        self.var3 = 0
complex_variable = Complex_Variable()
print(vars(complex_variable))
print(complex_variable.var1)

## Finding variables saved inside classifier. 
## 
vars(classifier)
```

```
classifier.tokenizer
```




    DistilBertTokenizerFast(name_or_path='distilbert/distilbert-base-uncased-finetuned-sst-2-english', vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True, added_tokens_decoder={
    	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }
    )



## Pipeline components
**Tokenizer**
-> Total number of vocabulary. 
-> Words in vocabulary are mapped word to numeric_id 

```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/629 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]


```
raw_inputs = [
    "I hate this so so much!",
]
input_tokenized = tokenizer(raw_inputs, return_tensors="pt") # , padding=True, truncation=True, return_tensors="pt")
print(input_tokenized)
```

    {'input_ids': tensor([[ 101, 1045, 5223, 2023, 2061, 2061, 2172,  999,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}


```
input_tokenized['input_ids'].shape
# Shape of 9 words to 9 token_ids
```




    torch.Size([1, 9])



```
token_ids = input_tokenized['input_ids']
"""
"token_ids" might be more intuitive in some ways, 
"input_ids" has become the standard terminology in the transformer ecosystem, 
particularly in the Hugging Face Transformers library. 
It's one of those cases where historical convention and technical specificity have led to a specific naming choice 
that might not be immediately obvious to newcomers.
"""
```

## Pipeline component - Model

```
# Classifier model is made of 3 sequential models
# ARCHITECTURE OF CLASSIFIER MODEL
classifier.model.distilbert
classifier.model.pre_classifier
classifier.model.dropout
classifier.model.classifier
```

```
# Take token_ids and returns a vector which contains meaning of the word. 

classifier.model.distilbert
# It is called as "embedding". (Also called hidden state)

print(classifier.model.distilbert( **input_tokenized.to("cuda")))
```

    BaseModelOutput(last_hidden_state=tensor([[[-0.2934,  0.6605, -0.1290,  ..., -0.1458, -0.9237,  0.0349],
             [-0.2303,  0.9075, -0.0971,  ..., -0.3591, -0.5514,  0.2912],
             [-0.1149,  0.8132, -0.0493,  ..., -0.2271, -0.7429,  0.1508],
             ...,
             [-0.2813,  0.6866, -0.1427,  ..., -0.3270, -0.8705,  0.1477],
             [-0.0420,  0.8612, -0.2456,  ..., -0.2251, -0.7963, -0.0410],
             [ 0.0545,  0.2883, -0.1800,  ..., -0.3559, -0.8834, -0.0389]]],
           device='cuda:0', grad_fn=<NativeLayerNormBackward0>), hidden_states=None, attentions=None)


```
classifier.model ( **input_tokenized.to("cuda"))
```




    SequenceClassifierOutput(loss=None, logits=tensor([[ 3.8815, -3.1521]], device='cuda:0', grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)



```
classifier.model.distilbert( **input_tokenized.to("cuda")).last_hidden_state.shape
```




    torch.Size([1, 9, 768])



```
classifier.model
```

```
from torchinfo import summary
import torch

WORDS_IN_SENTENCE = 100
dummy_input = {
    'input_ids':      torch.randint(0, 30522, (1, WORDS_IN_SENTENCE), dtype=torch.long, device=classifier.device),
    'attention_mask': torch.ones((1, WORDS_IN_SENTENCE), dtype=torch.long, device=classifier.device)
}

summary(classifier.model, input_data=dummy_input, verbose=1);
```

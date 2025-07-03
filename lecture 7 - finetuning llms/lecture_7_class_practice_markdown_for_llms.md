# Converted from lecture_7_class_practice.ipynb - Markdown format optimized for LLM readability

```python
import torch
```

```python
word_meaning = torch.randn((512))
```

```python
sentence = torch.randn((25,512))
```

```python
# TOTAL VOCABULARY SIZE = 1024
model = torch.nn.Sequential(
    torch.nn.Linear(in_features = 25*512 , out_features = 1024),
    torch.nn.Softmax(dim=1)
)
```

```python
import torchsummary

torchsummary.summary( model, input_size = (1, 25 * 512))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1              [-1, 1, 1024]      13,108,224
    ================================================================
    Total params: 13,108,224
    Trainable params: 13,108,224
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.05
    Forward/backward pass size (MB): 0.01
    Params size (MB): 50.00
    Estimated Total Size (MB): 50.06
    ----------------------------------------------------------------


```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
model     = transformers.AutoModel.from_pretrained("openai-community/gpt2")
```

```python
model
```




    GPT2Model(
      (wte): Embedding(50257, 768)
      (wpe): Embedding(1024, 768)
      (drop): Dropout(p=0.1, inplace=False)
      (h): ModuleList(
        (0-11): 12 x GPT2Block(
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): GPT2Attention(
            (c_attn): Conv1D(nf=2304, nx=768)
            (c_proj): Conv1D(nf=768, nx=768)
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (resid_dropout): Dropout(p=0.1, inplace=False)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): GPT2MLP(
            (c_fc): Conv1D(nf=3072, nx=768)
            (c_proj): Conv1D(nf=768, nx=3072)
            (act): NewGELUActivation()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )



```python

```

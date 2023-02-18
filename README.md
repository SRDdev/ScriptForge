<img src="https://imgs.search.brave.com/KGt2X1WFgdGVhaSWe_pqVk3yucJuxKvp4ENxA58K5DM/rs:fit:960:539:1/g:ce/aHR0cHM6Ly93cHRh/dmVybi5jb20vd3At/Y29udGVudC91cGxv/YWRzLzIwMjAvMTAv/R29vZ2xlLVBvZGNh/c3RzLWxvZ28uanBn" style="height:200px;">

# ScriptGPT
ScriptGPT is a GPT model built to generate amazing Youtube/Podcast/Film Scripts. ScriptGPT is a PyTorch implementation of the GPT (Generative Pre-trained Transformer) language model.

The current version is a smaller version based on only single podcast episode.

## Introduction to GPT
GPT (Generative Pre-trained Transformer) is a language model developed by OpenAI. It is based on the transformer architecture, which was introduced in the paper "Attention is All You Need" by Google researchers. The key idea behind GPT is to pre-train a deep neural network on a large dataset, and then fine-tune it on a specific task, such as language translation or question answering.

GPT's architecture consists of an encoder and a decoder, both of which are made up of multiple layers of self-attention and feed-forward neural network. The encoder takes in the input sequence and produces a representation of it, while the decoder generates the output sequence based on the representation.

GPT-2, an updated version of GPT, was trained on a dataset of over 40 GB of text data, and is able to generate human-like text, complete tasks such as translation and summarization, and even create original content.

GPTLite is a smaller version which is built for fine-tuning and is trained on the Dataset, which is still powerful enough to generate human-like text, but with less computational resources required.

## Dataset
The model is currently in beta stage and is only trained on single podcast episode.As the dataset for this is not available, I am building the dataset myself.
The youtube video on which this model is trained is linked below.

[Link](https://youtu.be/I-dlPuqFguo)

## Inference
__Currently Unavailable__

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModelForCausalLM.from_pretrained("")
```
_Pipeline_
```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model=model)

text="Just a very quick thing before we get started,"

generator(text, max_length=300, num_return_sequences=5)
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## Citations
```
@citation{ ScriptGPT,
  author = {Shreyas Dixit},
  year = {2023},
  url = {}
}
```

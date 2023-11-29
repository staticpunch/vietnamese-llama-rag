# Description
The main goal of this project is building a transformer model for closed question answering task (answer questions given a context) in Vietnamese, which can further be incorporated into a Retrieval-Augmented Generation (RAG) pipeline. This repo also contains the solution of my team (named Folivora) for the Viettel Hearted AI Challenge (VHAC).
- Instruction finetuned Llama-2-7b model that is suitable for Vietnamese closed QA.
- Basic RAG pipeline.
- Better RAG pipeline (our solution for VHAC).

# Introduction
Our initial motivation for this project was to compete in VHAC -  NLP track. In this track, we were given a corpus of Vietnamese Wikipedia articles and had to build a RAG solution to tackle general questions, whose answers could be found within the corpus. For more information about this competition, please visit: [`https://aihub.ml/competitions/557#participate`](https://aihub.ml/competitions/557). Our approach was utilizing semantic search combined with BM25 for retrieving most relevant contexts to a question, then using a GPT-like model solely trained on closed QA task to give answer to that question. 

# Usage
## Closed QA task only.
You can check our model card here: [`llm4fun/vietrag-7b-v1.0`](https://huggingface.co/llm4fun/vietrag-7b-v1.0)
```py
from transformers import GenerationConfig, TextStreamer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch

question = "<your-question>"
context = "<your-context>"
instruction = 'You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.'
input = f"Dựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\n{context}\n\nQuestion: {question}"
prompt_template = (
    "### System:\n"
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
prompt = prompt_template.format(instruction=instruction, input=input, output='')

torch_dtype = torch.bfloat16
model_id = "llm4fun/vietrag-7b-v1.0"
device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    config=LlamaConfig.from_pretrained(model_id),
    torch_dtype=torch_dtype
)

model = model.eval().to(device)

def generate(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # top_k=20,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )

    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

output = generate(prompt)
```
## RAG
You can start with `basic_rag.ipynb`

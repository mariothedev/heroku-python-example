# import urllib.parse
from transformers import pipeline
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI

app = FastAPI()

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
# tokenizer.decode(tokenizer.eos_token_id)

generator = pipeline("text-generation", model="distilgpt2")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/generate/{word}")
async def generate(word):
    res = generator(
        word,
        max_length=30,
        num_return_sequences=2
    )
    # input_ids = tokenizer.encode(urllib.parse.unquote(word), return_tensors='pt')
    # output = model.generate(input_ids, max_length=10, num_beams=3, no_repeat_ngram_size=2, early_stopping=True)
    # text = tokenizer.decode(output[0], skip_special_tokens=True)


    return {"output": res}
    # return {"output": text}
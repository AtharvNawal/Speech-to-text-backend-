import os
os.environ["USE_TF"] = "0"

from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator
import re

app = FastAPI()

# Allow all origins (or restrict to frontend IP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

# Load multilingual NER model
ner = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
translator = Translator()

def translate_to_english(text: str):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print("Translation error:", e)
        return text

def extract_phone(text: str):
    matches = re.findall(r"\b(?:\d[\s-]?){10,13}\b", text)
    for match in matches:
        normalized = re.sub(r"\D", "", match)
        if 10 <= len(normalized) <= 13:
            return normalized
    return ''

@app.post("/extract")
async def extract_entities(input_text: InputText):
    original = input_text.text
    translated = translate_to_english(original)

    result = ner(translated)
    name_parts = [ent["word"] for ent in result if ent["entity_group"] == "PER"]
    name = ' '.join(name_parts).strip()
    phone = extract_phone(translated)

    print("Original:", original)
    print("Translated:", translated)
    print("Name:", name or "Not found")
    print("Phone:", phone or "Not found")

    return {
        "Original": original,
        "Translated": translated,
        "name": name or "Not found",
        "phone": phone or "Not found"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

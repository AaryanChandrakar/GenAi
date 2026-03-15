import re
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from ollama import Client

app = FastAPI()

# Connect to local Ollama server
client = Client(host="http://localhost:11434")

# System prompt to guide the model output
SYSTEM_PROMPT = """
You are an AI assistant.
Respond in clean plain text.
Do not use markdown formatting.
Do not use bullet points, asterisks, or special symbols.
Write answers in simple sentences and paragraphs.
"""

def clean_output(text: str) -> str:
    """Remove markdown symbols and clean formatting."""
    text = text.replace("\\n", "\n")
    text = re.sub(r"\*\*", "", text)  # remove bold **
    text = re.sub(r"\*", "", text)    # remove *
    text = re.sub(r"\#", "", text)    # remove headings #
    text = re.sub(r"\-", "", text)    # remove list dashes
    return text.strip()

@app.post("/chat", response_class=PlainTextResponse)
def chat(query: str):
    response = client.chat(
        model="gemma:2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    raw_output = response["message"]["content"]
    cleaned_output = clean_output(raw_output)

    return cleaned_output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
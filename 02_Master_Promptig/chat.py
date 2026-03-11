from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "what is 2 + 2 * 0"} #Zero short prompting 
    ]
)

print(response.choices[0].message.content)
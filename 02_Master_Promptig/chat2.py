from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI Assistant who is specialised in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example: 
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10. Funfact you can even multiply 10 with 3 which gives same result.

Input: Why is sky blue?
Output: I am sorry I can only answer questions related to maths.
"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature = 0.5,
    max_tokens = 200,
    messages=[
        # Example of few-short prompting
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": "what is 2 + 2 * 0"} 
        #{"role": "user", "content": "Who is Salman Khan?"}
    ]
)

print(response.choices[0].message.content)
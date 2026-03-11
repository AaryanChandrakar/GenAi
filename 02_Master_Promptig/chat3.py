from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking doen complex problem and then resolve the user query.

For th given user input, analyse the problem and then break down the problem step by step.
Atleast think 5-6 steps on how to solve the prblem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation.

Example: 
Input: How to build a house?
Output: 
1. First we need to find a plot of land.
2. Then we need to get a permit from the local authorities.
3. Then we need to hire a contractor.
4. Then we need to build the house.
5. Then we need to get a certificate of occupancy. 

"""

result = client.chat.completions.create(
    model="gpt-4",
    temperature = 0.5,
    max_tokens = 300,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "How to book a honeymoon trip?"}
    ]

)

print(result.choices[0].message.content)
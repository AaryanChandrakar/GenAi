import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking doen complex problem and then resolve the user query.

For th given user input, analyse the problem and then break down the problem step by step.
Atleast think 5-6 steps on how to solve the prblem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then you finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON format output as per output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query 

Output Format:
{{step: "string, content: "string"}}

Example: 
Input: What is 2 + 2 ?
Output:{{step: "analyse", content:"Alright! The user is interested in maths query and he is asking a basic arithmatic operation.}}
Output:{{step: "think", content:"To perform the addition I must go left to right and add all the operands."}}
Output:{{step: "output", content:"4"}}
Output:{{step: "validate", content: "seems like 4 is correct and for 2 + 2"}}
Output:{{step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers"}}
"""

messages = [
    {"role": "system", "content":system_prompt}
]

query = input(">=")
messages.append({"role": "user", "content":query})

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    if parsed_response.get("step") != "output":
        print(f"🧠: {parsed_response.get('content')}")
        continue

    print(f"🤖: {parsed_response.get('content')}")
    break


# result = client.chat.completions.create(
#     model="gpt-4o",
#     response_format={"type": "json_object"},
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": "What is 3 + 4 * 5"},

#         #
#         {"role": "assistant", "content": json.dumps({"step": "analyse", "content": "The user is asking for the result of a mathematical expression which involves addition and multiplication."})},
#         {"role": "assistant", "content": json.dumps({"step": "think", "content": "To solve the expression 3 + 4 * 5, I need to follow the order of operations, also known as BIDMAS/BODMAS (Brackets, Orders, Division/Multiplication, Addition/Subtraction). Multiplication should be performed before addition."})},
#         {"role": "assistant", "content": json.dumps({"step": "think", "content": "First, I will calculate the multiplication part of the expression: 4 * 5."})},
#         {"role": "assistant", "content": json.dumps({"step": "output", "content": "The result of 4 * 5 is 20."})},
#         {"role": "assistant", "content": json.dumps({"step": "think", "content": "Next, I will perform the addition operation: 3 + 20."})},
#         {"role": "assistant", "content": json.dumps({"step": "output", "content": "The result of 3 + 20 is 23."})},
#         {"role": "assistant", "content": json.dumps({"step": "validate", "content": "The expression was correctly interpreted as following the rules of BIDMAS/BODMAS, and both intermediate and final steps seem accurate. 3 + 4 * 5 results in 23."})},
#         {"role": "assistant", "content": json.dumps({"step": "result", "content": "The result of the expression 3 + 4 * 5, when solved using the order of operations, is 23."})}


#     ]

# )

#print(result.choices[0].message.content)
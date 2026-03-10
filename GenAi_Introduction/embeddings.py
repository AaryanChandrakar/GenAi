from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI()

text = "Eiffel Tower is in Paris and is famous landmark, it is 324 meter tall."

response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

print("Vector Embeddings: ",response.data[0].embedding)

#Temperature: Temperature is a hyperparameter in machine learning that controls the randomness or creativity of the model's output.
# Higher the temperature, more creative the output.
# Lower the temperature, more deterministic the output.

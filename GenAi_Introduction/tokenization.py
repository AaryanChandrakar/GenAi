import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
#Token: A token is the basic unit of text or data that a Large Language Model (LLM) processes, serving as the "building blocks" for understanding and generating language.

#Tokenization: Tokenization in NLP is the fundamental process of breaking down raw text into smaller, meaningful units called tokens, which can be words, subwords, or characters, to prepare the data for machine learning models.

#Vector: A vector is a mathematical representation of data in a multi-dimensional space, where each dimension represents a feature or attribute of the data.

#Vectoration: Vectorization is the process of converting raw data into a vector representation, which can be used for machine learning models.

#Embedding: Embeddings are dense, numerical vector representations of data (text, images, audio, video) that capture semantic meaning and relationships, allowing AI models to measure similarity.

#Vocabulary size: Vocabulary size in an LLM is the total number of unique tokens (words, subwords, or characters) the model can understand and generate.

print("vocab Size", encoder.n_vocab) # 200019

text = "The cat sat on the mat."
token = encoder.encode(text)
print("Tokens: ",token)
print("Tokens count: ",len(token))

my_token = [976, 9059, 10139, 402, 290, 2450, 13]
decoded = encoder.decode(my_token)
print(decoded)
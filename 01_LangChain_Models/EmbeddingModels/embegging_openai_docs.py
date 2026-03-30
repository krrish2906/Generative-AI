from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
documents = [
    "Who is the current Prime Minister of India?",
    "Who is the current President of India?",
    "What is the capital of India?",
]

result = embeddings.embed_documents(documents)
print(str(result))
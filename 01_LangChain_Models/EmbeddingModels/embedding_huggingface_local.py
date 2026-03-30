from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The current Prime Minister of India is Narendra Modi.",
    "The current President of India is Droupadi Murmu.",
    "The capital of India is Delhi.",
]

result = embeddings.embed_documents(documents)
print(str(result))
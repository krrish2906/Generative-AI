from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who is the current Prime Minister of India?")
print(response.content)

# Output:- As of my cut-off knowledge in 2023, the current Prime Minister of India was Narendra Modi.
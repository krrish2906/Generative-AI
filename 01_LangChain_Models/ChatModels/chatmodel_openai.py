from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Place your OpenAI API key in the .env file
model = ChatOpenAI(model="gpt-4")

response = model.invoke("Who is the current Prime Minister of India?")
print(response.content)
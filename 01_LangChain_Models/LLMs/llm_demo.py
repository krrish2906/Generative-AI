from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Place your OpenAI API key in the .env file
llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("Who is the current Prime Minister of India?")
print(result)
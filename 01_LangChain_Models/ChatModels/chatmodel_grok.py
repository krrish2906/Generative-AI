from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Place your Groq API key in the .env file
model = ChatGroq(model="llama-3.3-70b-versatile")

response = model.invoke("Who is the current Prime Minister of India?")
print(response.content)

# Output:- The current Prime Minister of India is Narendra Modi. He has been in office since May 26, 2014.
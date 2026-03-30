from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Place your Google API key in the .env file
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

response = model.invoke("Who is the current Prime Minister of India?")
print(response.content)
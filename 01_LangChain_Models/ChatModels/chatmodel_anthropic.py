from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Place your Anthropic API key in the .env file
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

response = model.invoke("Who is the current Prime Minister of India?")
print(response.content)
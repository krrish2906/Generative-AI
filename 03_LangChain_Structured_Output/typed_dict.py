from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

class Review(TypedDict):
    summary: str
    sentiment: str

model = ChatGroq(model="llama-3.3-70b-versatile")
structured_model = model.with_structured_output(Review)

response = structured_model.invoke(
    """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this. Very bad experience. Didn't like it at all."""
)

print("Response:", response)
print("Type:", type(response))
for key, value in response.items():
    print(f"{key}: {value}", end="\n\n")

# Output:- 
"""
Response: {'sentiment': 'negative', 'summary': 'The product has great hardware, but the software is bloated with too many pre-installed apps and an outdated UI, leading to a very bad experience.'}

Type: <class 'dict'>

sentiment: negative

summary: The product has great hardware, but the software is bloated with too many pre-installed apps and an outdated UI, leading to a very bad experience.     
"""
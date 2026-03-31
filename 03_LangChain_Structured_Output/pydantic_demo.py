from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    
model = ChatGroq(model="llama-3.3-70b-versatile")
structured_model = model.with_structured_output(Review)

response = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

    Pros:
    Insanely powerful processor (great for gaming and productivity)
    Stunning 200MP camera with incredible zoom capabilities
    Long battery life with fast charging
    S-Pen support is unique and useful

    Cons:
    Heavy and bulky design
    Bloatware and unnecessary apps
    Expensive at $1,300
                                 
    Review by Krish Bansal"""
)

print("Response:", response)
print("Type:", type(response))
for key, value in response.model_dump().items():
    print(f"{key}: {value}", end="\n\n")

# Output:-
"""
Response: key_themes=['Samsung Galaxy S24 Ultra', 'powerful processor', 'camera capabilities', 'battery life', 'S-Pen support', 'design', 'bloatware', 'price'] summary="The Samsung Galaxy S24 Ultra is a powerful device with a great camera, long battery life, and useful S-Pen support, but it's heavy, bulky, and expensive with unnecessary bloatware" sentiment='pos' pros=['Insanely powerful processor', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful'] cons=['Heavy and bulky design', 'Bloatware and unnecessary apps', 'Expensive at $1,300'] name='Krish Bansal'

Type: <class '__main__.Review'>

key_themes: ['Samsung Galaxy S24 Ultra', 'powerful processor', 'camera capabilities', 'battery life', 'S-Pen support', 'design', 'bloatware', 'price']

summary: The Samsung Galaxy S24 Ultra is a powerful device with a great camera, long battery life, and useful S-Pen support, but it's heavy, bulky, and expensive with unnecessary bloatware

sentiment: pos

pros: ['Insanely powerful processor', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful']

cons: ['Heavy and bulky design', 'Bloatware and unnecessary apps', 'Expensive at $1,300']

name: Krish Bansal
"""
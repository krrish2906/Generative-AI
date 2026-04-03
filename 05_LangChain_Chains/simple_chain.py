from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

chain = prompt | model | parser
response = chain.invoke({'topic': 'Ramayana'})
print(response)

print("\n" + "=" * 80 + "\n")

chain.get_graph().print_ascii()
chain.get_graph().draw_mermaid_png(output_file_path="simple_chain.png")

# Output:-
"""
1. **The Ramayana is one of the oldest epics in the world**: The Ramayana is an ancient Indian epic poem that dates back to around 500 BCE. It is attributed to the sage Valmiki and is considered one of the most important works of Hindu literature. The epic has been passed down through generations and has had a significant influence on Indian culture and society.

2. **The story of Ramayana is not just limited to India**: The Ramayana has a significant presence in Southeast Asian countries such as Thailand, Indonesia, and Cambodia. The epic has been adapted and modified to suit the local culture and traditions of these countries, resulting in unique and fascinating variations of the story.

3. **Ravana, the antagonist, was a complex character**: Ravana, the king of Lanka, is often portrayed as a one-dimensional villain in the Ramayana. However, in the original text, Ravana is a complex character with a rich backstory and motivations. He is described as a wise and just king who is driven by a desire for power and revenge against the gods.

4. **The Ramayana has a significant impact on Indian art and architecture**: The Ramayana has had a profound influence on Indian art and architecture, with numerous temples, sculptures, and paintings depicting scenes from the epic. The most famous example is the Rama temple in Ayodhya, which is believed to be the birthplace of Lord Rama.

5. **The Ramayana contains valuable lessons on dharma and duty**: The Ramayana is not just a story about good vs. evil; it also contains valuable lessons on dharma (duty) and the importance of fulfilling one's responsibilities. The epic explores themes such as loyalty, duty, and selflessness, making it a timeless and universal tale that continues to inspire and educate people to this day.

================================================================================

     +-------------+       
     | PromptInput |       
     +-------------+       
            *
            *
            *
    +----------------+     
    | PromptTemplate |     
    +----------------+     
            *
            *
            *
      +----------+
      | ChatGroq |
      +----------+
            *
            *
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
            *
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
"""
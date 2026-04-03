from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Based on the following report: {report}, generate a 5 pointer summary',
    input_variables=['report']
)

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
response = chain.invoke({'topic': 'Epic History of India'})
print(response)

print("\n" + "=" * 80 + "\n")

chain.get_graph().print_ascii()
chain.get_graph().draw_mermaid_png(output_file_path="sequential_chain.png")

# Output:-
"""
Here is a 5-pointer summary of the epic history of India:

1. **Ancient India (3300 BCE - 500 CE)**: The history of India began with the Indus Valley Civilization, which was followed by the Vedic Period, marked by the rise of the Aryans and the composition of the Vedas, the oldest and most sacred texts of Hinduism.

2. **Classical and Medieval India (500 BCE - 1500 CE)**: This period saw the rise of powerful empires such as the Mauryan Empire, the Gupta Empire, the Delhi Sultanate, and the Mughal Empire, which made significant contributions to Indian culture, science, and administration.

3. **Modern India (1500 - 1947 CE)**: India was colonized by the British Empire, which had a profound impact on the country's infrastructure, economy, and education system, but also led to the exploitation of India's resources and the suppression of Indian culture.

4. **Key Figures and Events**: Important figures such as Chandragupta Maurya, Ashoka, Akbar, Mahatma Gandhi, and Jawaharlal Nehru played a significant role in shaping Indian history, while events like the Battle of Kalinga, the Invasion of India by Mahmud of Ghazni, and the Indian Independence Movement had a lasting impact on the country.

5. **Contemporary India (1947 CE - present)**: After gaining independence from British rule, India has made significant progress in its economy, technology, and international relations, but continues to face challenges such as poverty, inequality, and environmental degradation, while grappling with the legacy of its colonial past and ongoing debates about caste, religion, and language in Indian society.

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
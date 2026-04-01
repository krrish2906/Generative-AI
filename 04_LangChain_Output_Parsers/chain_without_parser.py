from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
print("Prompt 1:", prompt1, end="\n\n")

response1 = model.invoke(prompt1)
print("Response 1:", response1.content, end="\n\n")

prompt2 = template2.invoke({"text": response1.content})
print("Prompt 2:", prompt2, end="\n\n")

response2 = model.invoke(prompt2)
print("Response 2:", response2.content, end="\n\n")

# Output:-
"""
Prompt 1: text='Write a detailed report on Artificial Intelligence'

Response 1: **Artificial Intelligence (AI): A Comprehensive Report**

**Introduction**

Artificial Intelligence (AI) is a rapidly evolving field of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence, such as learning, problem-solving, and decision-making. AI has made significant advancements in recent years, transforming various industries and revolutionizing the way we live and work.

**History of AI**

The concept of AI dates back to the mid-20th century, when computer scientists began exploring the possibility of creating machines that could think and learn like humans. The term "Artificial Intelligence" was first coined in 1956 by John McCarthy, a prominent computer scientist. Since then, AI has undergone several periods of growth and stagnation, with notable milestones including:

1. **1950s-1960s**: The first AI programs were developed, including the Logical Theorist and the General Problem Solver.
2. **1970s**: The first expert systems were created, which were designed to mimic human decision-making.
3. **1980s**: AI experienced a resurgence, with the development of machine learning and neural networks.
4. **1990s**: AI saw significant advancements in areas like natural language processing and computer vision.
5. **2000s**: The rise of big data and the internet of things (IoT) led to the emergence of new AI applications.

**Types of AI**

There are several types of AI, each with its unique characteristics and applications:

1. **Narrow or Weak AI**: Designed to perform a specific task, such as image recognition, speech recognition, or natural language processing.
2. **General or Strong AI**: Aims to replicate human intelligence and reason across a wide range of tasks.
3. **Superintelligence**: Significantly more intelligent than the best human minds, with the ability to learn and adapt at an exponential rate.
4. **Artificial General Intelligence (AGI)**: A type of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks, similar to human intelligence.

**Applications of AI**

AI has numerous applications across various industries, including:

1. **Healthcare**: AI is used in medical imaging, disease diagnosis, and personalized medicine.
2. **Finance**: AI is applied in risk assessment, portfolio management, and customer service.
3. **Retail**: AI is used in customer service, inventory management, and supply chain optimization.
4. **Transportation**: AI is used in autonomous vehicles, route optimization,

Prompt 2: text='Write a 5 line summary on the following text:\n **Artificial Intelligence (AI): A Comprehensive Report**\n\n**Introduction**\n\nArtificial Intelligence (AI) is a rapidly evolving field of computer science that focuses on creating intelligent machines capable of performing tasks that typically require human intelligence, such as learning, problem-solving, and decision-making. AI has made significant advancements in recent years, transforming various industries and revolutionizing the way we live and work.\n\n**History of AI**\n\nThe concept of AI dates back to the mid-20th century, when computer scientists began exploring the possibility of creating machines that could think and learn like humans. The term "Artificial Intelligence" was first coined in 1956 by John McCarthy, a prominent computer scientist. Since then, AI has undergone several periods of growth and stagnation, with notable milestones including:\n\n1. **1950s-1960s**: The first AI programs were developed, including the Logical Theorist and the General Problem Solver.\n2. **1970s**: The first expert systems were created, which were designed to mimic human decision-making.\n3. **1980s**: AI experienced a resurgence, with the development of machine learning and neural networks.\n4. **1990s**: AI saw significant advancements in areas like natural language processing and computer vision.\n5. **2000s**: The rise of big data and the internet of things (IoT) led to the emergence of new AI applications.\n\n**Types of AI**\n\nThere are several types of AI, each with its unique characteristics and applications:\n\n1. **Narrow or Weak AI**: Designed to perform a specific task, such as image recognition, speech recognition, or natural language processing.\n2. **General or Strong AI**: Aims to replicate human intelligence and reason across a wide range of tasks.\n3. **Superintelligence**: Significantly more intelligent than the best human minds, with the ability to learn and adapt at an exponential rate.\n4. **Artificial General Intelligence (AGI)**: A type of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks, similar to human intelligence.\n\n**Applications of AI**\n\nAI has numerous applications across various industries, including:\n\n1. **Healthcare**: AI is used in medical imaging, disease diagnosis, and personalized medicine.\n2. **Finance**: AI is applied in risk assessment, portfolio management, and customer service.\n3. **Retail**: AI is used in customer service, inventory management, and supply chain optimization.\n4. **Transportation**: AI is used in autonomous vehicles, route optimization,'

Response 2: Here is a 5-line summary of the text:

Artificial Intelligence (AI) is a rapidly evolving field that focuses on creating intelligent machines capable of performing human-like tasks. The concept of AI dates back to the mid-20th century, with significant advancements in recent years transforming various industries. There are several types of AI, including narrow or weak AI, general or strong AI, superintelligence, and artificial general intelligence (AGI). AI has numerous applications across industries such as healthcare, finance, retail, and transportation. The field of AI continues to grow and evolve, with ongoing research and development of new AI technologies.
"""
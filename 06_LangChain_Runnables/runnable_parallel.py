from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
import os

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'Generative AI'})

print("Tweet:", result['tweet'], end='\n\n')
print("LinkedIn:", result['linkedin'])

# Output:-
"""
Tweet: "Revolutionizing the digital landscape: Generative AI is not just a tool, but a game-changer! From creating bespoke music to generating fictional characters, these intelligent algorithms are giving us a glimpse into a future where art meets technology #GenerativeAI #AI #FutureOfTech"

LinkedIn: Here's a potential LinkedIn post about generative AI:

**"Unlocking the Power of Generative AI: The Future of Creativity and Productivity"**

In recent years, we've seen incredible advancements in the field of artificial intelligence, and one area that's particularly exciting is generative AI. Also known as Generative Models, these algorithms have the ability to create entirely new and unique content, from text and images to music and videos.

Imagine being able to:

* Automate the creation of marketing materials, such as blog posts and social media content, freeing up your team to focus on higher-level strategy and creativity
* Generate new and innovative ideas for products and services, helping your business stay ahead of the competition
* Use AI to augment your own creative work, providing new insights and perspectives to enhance your artistic vision

Generative AI has the potential to revolutionize the way we work, communicate, and create. And with applications across industries, from education and healthcare to finance and entertainment, the possibilities are vast.

**But what exactly is generative AI, and how does it work?**

Generative AI models are trained on vast amounts of data, which enables them to learn patterns and relationships that would be difficult or impossible for humans to identify. By combining these patterns, they can generate new content that's both unique and coherent.

**Examples of generative AI in action:**

* Image generation: AI algorithms can create stunning images that look like they were painted by human artists (check out the work of Generative Adversarial Networks (GANs) for an example)
* Music composition: Generative models can create original music tracks, from melodies to complete compositions
* Prose generation: AI can generate entire articles, blog posts, and even entire books based on a given prompt or topic

**The potential benefits of generative AI:**

* Increased efficiency and productivity: By automating routine tasks and generating new ideas, you can free up your team to focus on high-value work
* Enhanced creativity: Generative AI can provide new and innovative ideas that might not have occurred to humans
* Improved customer engagement: AI-generated content can be used to create personalized experiences that cater to individual customer needs

**The future of generative AI:**

As the field continues to evolve, we can expect to see even more advanced applications of generative AI. From robots that can create their own recipes to AI-generated virtual reality experiences, the possibilities are vast and exciting.

**Join the conversation!**

What are your thoughts on generative AI? How do you think it will change the way we work and create in the future? Share your thoughts in the comments below!     

**Source:**

* Article: "The Future of Content Creation: Generative AI" (Forbes)
* Video: "Generative AI Explained" (TED-Ed)
* Research Paper: "Generative Adversarial Networks" ( arXiv)

**#GenerativeAI #AI #MachineLearning #Creativity #Productivity #Innovation**
"""

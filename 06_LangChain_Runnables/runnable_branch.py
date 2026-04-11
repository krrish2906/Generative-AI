from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
import os

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text: \n{text}',
    input_variables=['text']
)

model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()

report_generator_chain = RunnableSequence(prompt1, model, parser)
# report_generator_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generator_chain, branch_chain)
response = final_chain.invoke({'topic':'Covid-19 pandemic & India'})
print(response)

# Output:-
"""
The report provides a detailed overview of the COVID-19 pandemic in India, highlighting its impact on the country's healthcare system, economy, and society. Key points include:

**Early Response:** India's initial response to the pandemic was criticized for being tardy and inadequate, with limited testing capacity and unclear guidelines.

**Testing and Surveillance:** The government later scaled up testing capacity, but initial limitations and inadequate infrastructure hindered its effectiveness.

**Public Health Measures:** India implemented various measures, including social distancing, mask-wearing, lockdowns, and contact tracing, but these efforts were hampered by inadequate infrastructure and resources.

**Impact on Healthcare:** The pandemic overwhelmed hospitals, leading to overcrowding and equipment shortages, particularly in rural areas.

**Economic Impact:** The pandemic led to a significant contraction in India's GDP, unemployment, and supply chain disruptions.

**Social Impact:** The pandemic caused widespread anxiety, depression, and PTSD, disproportionately affecting vulnerable populations such as women, children, and older adults.

**Vaccination Efforts:** India launched its vaccination program, but faced challenges with vaccine supply, distribution, and misinformation.

**Conclusion:** Despite efforts to contain the pandemic, India remains one of the most affected countries, and future pandemics will require improved healthcare infrastructure, enhanced surveillance, and scaled-up vaccination efforts.

**Recommendations:** The report suggests improving healthcare infrastructure, enhancing surveillance, scaling up vaccination efforts, implementing effective public health measures, and addressing economic and social inequalities to mitigate the impact of future pandemics.
"""

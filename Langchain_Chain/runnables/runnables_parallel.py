from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt1 = PromptTemplate(
    template="Generate a Linkdin post about {topic}",
    input_variables=["topic"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'Linkdin':RunnableSequence(prompt1, model, parser),
})

print(chain.invoke({'topic':'AI'}))
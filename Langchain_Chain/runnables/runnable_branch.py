from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Sumaarise the following - {text}",
    input_variables=["text"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 10000,
        RunnableSequence(prompt2, model, parser)
    ),
    RunnablePassthrough()
)


final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

print(final_chain.invoke({'topic':'AI'}))
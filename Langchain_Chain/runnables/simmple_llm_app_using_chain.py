from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = GoogleGenerativeAI(model="gemini-2.0-flash")
prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Suggest a catchy blog title about {topic}."  
)

chain = LLMChain(llm = llm , prompt = prompt)
topic = input("Enter a topic: ")
output = chain.run(topic)
print("Geerated Blog Title: ", output)
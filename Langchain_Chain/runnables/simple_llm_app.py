from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    input_variables = ["topic"],
    template = "Suggest a catchy blog title about {topic}."  
)

topic = input("Enter a topic: ")

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)
print("Geerated Blog Title: ", blog_title)
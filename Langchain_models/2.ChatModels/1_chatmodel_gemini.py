from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7,max_output_tokens=5)

response = model.invoke("Capital of India")

print(response)
print(response.content)

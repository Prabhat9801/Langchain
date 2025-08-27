from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",dimensions=32)

embedded = embeddings.embed_query("What is Langchain?")

print(str(embedded))
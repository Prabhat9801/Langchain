from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",dimensions=32)
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.",
    "LangChain is data-aware and can connect a language model to other sources of data.",]
embedded = embeddings.embed_documents(documents)

print(str(embedded))
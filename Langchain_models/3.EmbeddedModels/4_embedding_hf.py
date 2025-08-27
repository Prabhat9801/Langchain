from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()  # loads HUGGINGFACEHUB_API_TOKEN automatically

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.",
    "LangChain is data-aware and can connect a language model to other sources of data."
]

embedded = embeddings.embed_documents(documents)
print(f"Shape: {len(embedded)} x {len(embedded[0])}")  # should be num_docs x 384

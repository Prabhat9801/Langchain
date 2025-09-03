from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
loader = TextLoader('./Langchain_Chain/runnables/docs.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001", dimensions=32))

retriever = vectorstore.as_retriever()

query = "what are the key takeaway from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

prompt = f"Based on the following text, answer the question: {query}\nDocument:\n\n{retrieved_text}"

answer = llm.predict(prompt)

print("Answer: ", answer)



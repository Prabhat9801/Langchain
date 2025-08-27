from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    # provider="auto",  # let Hugging Face choose the best provider for you
)
model = ChatHuggingFace(llm = llm )

response = model.invoke("who is the president of india?")


print(response.content)
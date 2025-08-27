from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Step 1: Generate prompt with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
prompt_result = gemini.invoke("Generate a prompt for the image generation of dog")
prompt = prompt_result.content
print("Prompt:", prompt)

# Step 2: Generate image using Hugging Face
client = InferenceClient(
    provider="fal-ai",   # You can also use default HF provider if available
    api_key=HF_TOKEN,
)

# Output is a PIL.Image object
image = client.text_to_image(
    prompt,
    model="Qwen/Qwen-Image"
)

# Save the image
image.save("output.png")
print("✅ Image saved as output.png")

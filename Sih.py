from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment
load_dotenv()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", dimensions=32)

# Internship documents with metadata
internships = [
    {
        "id": "I-101",
        "title": "Data Analysis Intern",
        "org": "AgriTech Pvt Ltd",
        "required_education": "UG",
        "skills": "python, excel, pandas",
        "sector": "Agriculture",
        "location": "remote",
        "apply_url": "https://example.com/apply/I-101"
    },
    {
        "id": "I-102",
        "title": "Web Development Intern",
        "org": "Tech4All",
        "required_education": "UG",
        "skills": "html, css, javascript, react",
        "sector": "IT",
        "location": "delhi",
        "apply_url": "https://example.com/apply/I-102"
    },
    {
        "id": "I-103",
        "title": "Field Outreach Intern",
        "org": "RuralAction",
        "required_education": "12th",
        "skills": "communication, mobilization",
        "sector": "Rural Development",
        "location": "bihar",
        "apply_url": "https://example.com/apply/I-103"
    },
    {
        "id": "I-104",
        "title": "Content Writing Intern",
        "org": "EduVision",
        "required_education": "12th",
        "skills": "writing, research, editing",
        "sector": "Education",
        "location": "remote",
        "apply_url": "https://example.com/apply/I-104"
    },
    {
        "id": "I-105",
        "title": "Public Health Intern",
        "org": "HealthFirst",
        "required_education": "UG",
        "skills": "data collection, ms-office",
        "sector": "Healthcare",
        "location": "mumbai",
        "apply_url": "https://example.com/apply/I-105"
    }
]

# Convert internship data into documents for embedding
documents = [
    f"{i['title']} at {i['org']} requires {i['required_education']} education with skills in {i['skills']}. "
    f"Sector: {i['sector']}. Location: {i['location']}. Apply here: {i['apply_url']}"
    for i in internships
]

# User input
education = input("Enter your education (e.g., UG, 12th, BTech): ")
skills = input("Enter your skills (comma separated, e.g., python, html, machine learning): ")
location = input("Enter your preferred location (e.g., remote, delhi, mumbai): ")

# Create a query from user details
query = f"Education: {education}, Skills: {skills}, Location: {location}"

# Generate embeddings
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Calculate similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get top 3–4 recommendations
top_indices = np.argsort(scores)[-4:][::-1]

# Show recommendations in a nice format
print("\n✨ Recommended Internships for You:\n")
for i in top_indices:
    job = internships[i]
    print(f"🔹 {job['title']} at {job['org']}")
    print(f"   📍 Location: {job['location']}")
    print(f"   🎓 Required Education: {job['required_education']}")
    print(f"   🛠 Skills: {job['skills']}")
    print(f"   🔗 Apply Here: {job['apply_url']}")
    print(f"   ✅ Similarity Score: {scores[i]:.4f}\n")

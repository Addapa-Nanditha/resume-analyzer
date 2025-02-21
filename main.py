from fastapi import FastAPI, File, UploadFile
import pdfplumber
from docx import Document
import os
import openai
import re 
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
import requests 
from collections import Counter
import nltk
from nltk.corpus import stopwords
from typing import Dict
from dotenv import load_dotenv
import requests 
from typing import List, Dict, Union



# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Download stopwords if not already present
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# ------------- TEXT CLEANING & NLP PROCESSING ----------------
def clean_text(text: str) -> str:
    """Clean resume text by removing special characters and extra spaces"""
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
    text = text.lower().strip()  # Convert to lowercase & trim spaces
    return text
    
'''def clean_text(text: str) -> str:
    """Clean resume text while preserving important characters like email addresses"""
    
    # First save email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    
    # Replace emails with placeholders
    for i, email in enumerate(emails):
        text = text.replace(email, f"EMAIL_PLACEHOLDER_{i}")
    
    # Clean the text but preserve some important characters
    text = re.sub(r'[^a-zA-Z\s\.\-\+\@]', ' ', text)  # Keep dots, hyphens, plus, @
    text = text.lower().strip()
    
    # Restore emails
    for i, email in enumerate(emails):
        text = text.replace(f"email_placeholder_{i}", email)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text'''


def remove_stopwords(text: str) -> str:
    """Remove common stopwords from resume text"""
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return " ".join(filtered_words)



'''def extract_skills(text: str) -> list:
    """Automatically extract skills from resume text using SkillNER"""
    annotations = skill_extractor.annotate(text)
    detected_skills = [skill["doc_node_value"] for skill in annotations["results"]["full_matches"]]
    
    return list(set(detected_skills))  # Return unique skills

    doc = nlp(text)
    
    # Extract only meaningful nouns and proper nouns
    detected_skills = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    # Filter skills that are present in the predefined skills list
    matched_skills = set(detected_skills).intersection(predefined_skills)
    
    return list(matched_skills)  # Return only relevant skills'''
    




# Load API key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter client
client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def extract_skills(text: str) -> List[str]:
    """
    Extract skills from resume text using OpenRouter API
    
    Args:
        text (str): Cleaned resume text
    Returns:
        List[str]: List of extracted skills or error messages
    """
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    if not OPENROUTER_API_KEY:
        return ["Error: OpenRouter API key not found in environment variables"]
    
    # API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000"  # Replace with your actual URL in production
    }
    
    # Structured prompt for better skill extraction
    prompt = """
    Extract technical and soft skills from the following resume text. 
    Return ONLY a comma-separated list of skills, with no additional text or formatting.
    Focus on:
    - Technical skills (programming languages, tools, platforms)
    - Soft skills (leadership, communication, etc.)
    - Industry-specific skills
    
    Resume text:
    {text}
    """.format(text=text)
    
    # Request body
    data = {
        "model": "gpt-3.5-turbo",  # You can change this to other available models
        "messages": [
            {
                "role": "system",
                "content": "You are a skilled ATS (Applicant Tracking System) that extracts professional skills from resumes."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        # Debug logging
        print("DEBUG: API Response Status Code:", response.status_code)
        print("DEBUG: API Response Headers:", response.headers)
        print("DEBUG: API Response Body:", response.text)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        # Verify response structure
        if not response_data.get('choices'):
            return ["Error: API response missing 'choices' field"]
            
        if not response_data['choices'][0].get('message'):
            return ["Error: API response missing 'message' field"]
            
        # Extract and process skills
        skills_text = response_data['choices'][0]['message']['content'].strip()
        
        # Split skills and clean them
        skills_list = [
            skill.strip()
            for skill in skills_text.split(',')
            if skill.strip()
        ]
        
        return skills_list if skills_list else ["No skills were extracted"]
        
    except requests.exceptions.RequestException as e:
        return [f"Error: API request failed - {str(e)}"]
    except ValueError as e:
        return [f"Error: Failed to parse API response - {str(e)}"]
    except Exception as e:
        return [f"Error: Unexpected error - {str(e)}"]


'''@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)) -> Dict[str, str]:
    """API endpoint to upload a resume and extract text & skills"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    else:
        return {"error": "Unsupported file type. Please upload a PDF or DOCX file."}'''
    
@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)) -> dict:
    """API to upload a resume and extract cleaned text & skills"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract text from file
    if file.filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    else:
        return {"error": "Unsupported file type."}

    # Debugging prints to check output at each step
    print("\n==================== DEBUGGING START ====================")
    print("RAW TEXT EXTRACTED:", raw_text[:500])  # Print first 500 characters
    print("\n--------------------------------------------------------")

    cleaned_text = clean_text(raw_text)
    print("CLEANED TEXT:", cleaned_text[:500])  # Print first 500 characters
    print("\n--------------------------------------------------------")

    text_without_stopwords = remove_stopwords(cleaned_text)
    print("TEXT WITHOUT STOPWORDS:", text_without_stopwords[:500])  # Print first 500 characters
    print("\n--------------------------------------------------------")

    extracted_skills = extract_skills(text_without_stopwords)
    print("EXTRACTED SKILLS:", extracted_skills)
    print("===================== DEBUGGING END =====================\n")

    return {
        "filename": file.filename,
        "cleaned_text": text_without_stopwords,
        "extracted_skills": extracted_skills
    }


'''# Process text using NLP
    cleaned_text = clean_text(raw_text)
    text_without_stopwords = remove_stopwords(cleaned_text)
    extracted_skills = extract_skills(text_without_stopwords)

    return {
        "filename": file.filename,
        "cleaned_text": text_without_stopwords,
        "extracted_skills": extracted_skills
    }'''
    
    


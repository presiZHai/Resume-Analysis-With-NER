import subprocess
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
import os
import re

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Path to the 'data' folder containing the PDF resumes
data_folder = 'data'

# List of PDF file names containing resumes
pdf_files = [
    'CV_Abhilash.pdf',
    'CV_Aleksandr.pdf',
    'CV_Andrea.pdf',
    'CV_Charlotte.pdf',
    'CV_Kanika.pdf', 
    'CV_Manuel.pdf',
    'CV_Mythily.pdf', 
    'CV_Nikhil.pdf', 
    
]

# Full paths to the PDF files
pdf_paths = [os.path.join(data_folder, pdf_file) for pdf_file in pdf_files]

# Extract text from each PDF resume and store it in a list
resumes_text = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]

# Create a DataFrame with columns 'ID' and 'resume_text'
data = pd.DataFrame({'ID': range(1, len(pdf_files) + 1), 'resume_text': resumes_text})

# Path to save the CSV file
output_csv_path = os.path.join(data_folder, 'resumes.csv')

# Save the DataFrame to a CSV file
data.to_csv(output_csv_path, index=False)

# Load data from CSV file
data = pd.read_csv('data/resumes.csv')  

# Suppress the output of the spacy download command
subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Add entity ruler pipeline to spaCy model
entity_ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define patterns as dictionaries
patterns = [
    {"label": "SKILL", "pattern": [{"LOWER": "matplotlib"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "python"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "pandas"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "seaborn"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "sql"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "mysql"}]},
    {"label": "PERSON", "pattern": [{"LOWER": "name"}]}
]

# Add patterns to entity ruler
entity_ruler.add_patterns(patterns)

# Download NLTK resources
nltk.download('punkt')  # Download the 'punkt' tokenizer resource
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove hyperlinks, special characters, and punctuations using regex
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Convert the text to lowercase
    text = text.lower()

    # Tokenize the text using nltk's word_tokenize
    words = word_tokenize(text)

    # Lemmatize the text to its base form for normalization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Remove English stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word not in stop_words]

    # Join the tokens back into a single string
    return ' '.join(filtered_words)

# Clean the 'resume_text' column in the DataFrame
data['cleaned_resume'] = data['resume_text'].apply(clean_text)

# Define options for visualization
options = {'ents': ['PERSON', 'GPE', 'SKILL'], 'colors': {'PERSON': 'orange', 'GPE': 'lightgreen', 'SKILL': 'lightblue'}}

# Visualize named entities in each resume
for resume_text in data['cleaned_resume']:
    doc = nlp(resume_text)
    displacy.render(doc, style="ent", jupyter=True, options=options)

# Define the company requirements
company_requirements = """We are seeking a Data Analyst with experience using Python for data cleaning, data analysis, and exploratory data analysis (EDA). 
                          The ideal candidate will also have the ability to explain complex mathematical concepts to non-mathematicians."""

# Combine the company requirements with stopwords removed
cleaned_company_requirements = clean_text(company_requirements)

# Calculate TF-IDF vectors for the company requirements and resume texts
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['resume_text'])
company_tfidf = tfidf_vectorizer.transform([cleaned_company_requirements])

# Calculate cosine similarity between the company requirements and each resume
similarity_scores = cosine_similarity(company_tfidf, tfidf_matrix).flatten()

# Get the indices of resumes sorted by similarity score
sorted_indices = similarity_scores.argsort()[::-1]

# Display the top 8 most similar resumes
top_n = 8
for i in range(top_n):
    index = sorted_indices[i]
    print(f"Resume ID: {data['ID'][index]}")
    print(f"Similarity Score: {similarity_scores[index]}")
    print(data['resume_text'][index])
    print()

def calculate_similarity(resume_text, required_skills):
    # Process the resume text with the spaCy model
    doc = nlp(resume_text)

    # Extract skills from the resume using the entity ruler
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"]

    # Calculate the number of matching skills with required skills
    matching_skills = [skill for skill in skills if skill in required_skills]
    num_matching_skills = len(matching_skills)

    # Calculate the similarity score
    similarity_score = num_matching_skills / max(len(required_skills), len(skills))

    return similarity_score

# Example usage:
for text in data[['cleaned_resume']].itertuples(index = False):
    resume_text = str(text[0])
    print(resume_text)
    required_skills = ["matplotlib", "python", "pandas", "seaborn", "sql", "mysql"]
    similarity_score = calculate_similarity(resume_text, required_skills)
    print("Similarity Score:", similarity_score)
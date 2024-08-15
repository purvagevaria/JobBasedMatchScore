import gradio as gr
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file.name, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to calculate similarity using TF-IDF vectors
def calculate_tfidf_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
    return similarity_score

# Function to analyze resume and job description
def analyze_resume_and_job_desc(resume_file, job_desc_text):
    resume_text = extract_text_from_pdf(resume_file)
    match_score = calculate_tfidf_similarity(resume_text, job_desc_text)
    suggestions = []
    if match_score < 0.7:
        suggestions = [
            "Highlight more relevant skills.",
            "Add specific experiences that match the job description.",
            "Use keywords from the job description."
        ]
    return {'matchScore': match_score * 100, 'suggestions': suggestions}

# Define Gradio interface
iface = gr.Interface(
    fn=analyze_resume_and_job_desc,
    inputs=[
        gr.File(label="Upload Resume (PDF)"),
        gr.Textbox(lines=10, placeholder="Paste job description here...", label="Job Description")
    ],
    outputs=[
        gr.JSON(label="Results")
    ],
    live=True
)

iface.launch()

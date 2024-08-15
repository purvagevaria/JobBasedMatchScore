import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
import fitz  # PyMuPDF for extracting text from PDF
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file, filetype='pdf')
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Function to compute similarity score using scikit-learn
def compute_match_score(resume_text, job_description):
    # Vectorize the texts
    vectors = vectorizer.fit_transform([resume_text, job_description])
    
    # Compute cosine similarity
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100  # Convert to percentage
    return score

# Function to generate recommendations
def generate_recommendations(score):
    recommendations = []
    if score < 50:
        recommendations.append("Highlight more relevant skills.")
        recommendations.append("Add specific experiences that match the job description.")
    else:
        recommendations.append("Your resume is well-aligned with the job description.")
    return recommendations

# Function to generate a chart
def generate_chart(score):
    fig, ax = plt.subplots()
    ax.bar(['Match Score'], [score], color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score')
    ax.set_title('Match Score')
    
    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to PIL Image
    buf_image = Image.open(buf)
    return buf_image

# Function to handle Gradio interface
def analyze(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)
    match_score = compute_match_score(resume_text, job_description)
    recommendations = generate_recommendations(match_score)
    chart_image = generate_chart(match_score)
    return f"Match Score: {match_score:.2f}%", recommendations, chart_image

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Job Matching System")
        with gr.Row():
            resume_input = gr.File(label="Upload Resume (PDF)", type="binary")
            job_description_input = gr.Textbox(label="Paste Job Description Text", lines=10)
        with gr.Row():
            submit_button = gr.Button("Analyze")
            output = gr.Textbox(label="Match Score")
            recommendations_output = gr.Textbox(label="Recommendations")
            chart_output = gr.Image(type="pil", label="Match Score Chart")
        submit_button.click(analyze, inputs=[resume_input, job_description_input], outputs=[output, recommendations_output, chart_output])
    return demo

if __name__ == "__main__":
    gradio_interface().launch(share=True)

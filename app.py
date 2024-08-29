import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import fitz  
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean and tokenize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, bytes):
        pdf_document = fitz.open(stream=pdf_file, filetype='pdf')
    else:
        pdf_document = fitz.open(pdf_file.name)
    
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    pdf_document.close()
    return text

# Calculate match score using TF-IDF and Cosine Similarity
def compute_match_score(resume_text, job_description):
    resume_text_cleaned = clean_text(resume_text)
    jd_text_cleaned = clean_text(job_description)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text_cleaned, jd_text_cleaned])
    
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    match_score = cosine_sim * 100  # Convert to percentage
    
    return match_score

# Function to generate recommendations based on match score
def generate_recommendations(score):
    recommendations = []
    if score < 60:
        recommendations.append("Consider using more relevant keywords.")
        recommendations.append("Fix typos and variations in your resume.")
    elif score < 80:
        recommendations.append("Your resume is fairly aligned, but some key skills or experiences might be missing.")
    else:
        recommendations.append("Your resume is well-aligned with the job description.")
    
    recommendations_text = "\n".join([f"• {rec}" for rec in recommendations])
    return recommendations_text

# Generate a pie chart for match score
def generate_chart(score):
    score = min(max(score, 0), 100)
    
    labels = ['Match Score', 'Other']
    sizes = [score, 100 - score]
    colors = ['#FFA500', '#E0E0E0']
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops=dict(color="black", fontsize=12),
                                      wedgeprops=dict(width=0.3, edgecolor='black'))
    ax.set_title('Match Score', fontsize=16, fontweight='bold', color='black')
    
    ax.legend(wedges, labels, title="Score Breakdown",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12, title_fontsize='13', shadow=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    buf_image = Image.open(buf)
    return buf_image

# Creating Gradio interface
def analyze(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)
    match_score = compute_match_score(resume_text, job_description)
    recommendations = generate_recommendations(match_score)
    chart_image = generate_chart(match_score)
    
    return f"Match Score: {match_score:.2f}%", recommendations, chart_image

# Gradio interface
def gradio_interface():
    with gr.Blocks(css="""
        #analyze-button {
            background: linear-gradient(45deg, #FF6347, #FFD700);
            color: white;
            border-radius: 8px;
            font-size: 14px;
            padding: 8px 16px;
            border: none;
            width: 120px;
            height: 40px;
        }
        #analyze-button:hover {
            background: linear-gradient(45deg, #FF4500, #FFA500);
        }
        .gradio-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        .gradio-output {
            font-family: sans-serif;
        }
    """) as demo:
        gr.Markdown("## Resume and Job Description Analyzer")
        
        with gr.Row():
            resume_input = gr.File(label="Upload Resume (PDF)")
        
        with gr.Row():
            job_description_input = gr.Textbox(label="Enter Job Description", lines=10)
        
        with gr.Row():
            analyze_button = gr.Button("Analyze", elem_id="analyze-button")
        
        match_score_output = gr.Textbox(label="Match Score", elem_classes="gradio-output")
        recommendations_output = gr.Textbox(label="Recommendations", elem_classes="gradio-output")
        chart_output = gr.Image(label="Match Score Chart")
        
        analyze_button.click(
            fn=analyze, 
            inputs=[resume_input, job_description_input], 
            outputs=[match_score_output, recommendations_output, chart_output]
        )
    
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)

# Launch Gradio interface
if __name__ == "__main__":
    gradio_interface()

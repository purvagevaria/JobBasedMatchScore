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

# Function to generate a pie chart
def generate_chart(score):
    labels = ['Match Score', 'Remaining']
    sizes = [score, 100 - score]
    colors = ['#4CAF50', '#FFC107']
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops=dict(color="black", fontsize=12),
                                      wedgeprops=dict(width=0.3))
    ax.set_title('Match Score', fontsize=16, fontweight='bold', color='black')
    
    # Add a legend
    ax.legend(wedges, labels, title="Score Breakdown",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12, title_fontsize='13', shadow=True)
    
    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
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
    with gr.Blocks(css="""
        #analyze-button {
            background-color: #4CAF50;  /* Green background */
            color: white;               /* White text */
            border-radius: 5px;         /* Rounded corners */
            font-size: 14px;            /* Smaller font size */
            padding: 8px 16px;          /* Padding around the text */
            border: none;               /* Remove border */
        }
        #analyze-button:hover {
            background-color: #45a049;  /* Darker green on hover */
        }
        .gradio-container {
            font-family: Arial, sans-serif; /* Custom font */
        }
    """) as demo:
        gr.Markdown("# Job Matching System")
        with gr.Row():
            resume_input = gr.File(label="Upload Resume (PDF)", type="binary")
            job_description_input = gr.Textbox(label="Paste Job Description Text", lines=10)
        with gr.Row():
            submit_button = gr.Button("Analyze", elem_id="analyze-button")
            output = gr.Textbox(label="Match Score")
            recommendations_output = gr.Textbox(label="Recommendations")
            chart_output = gr.Image(type="pil", label="Match Score Chart")
        
        submit_button.click(analyze, inputs=[resume_input, job_description_input], outputs=[output, recommendations_output, chart_output])
    
    return demo

if __name__ == "__main__":
    gradio_interface().launch(share=True)

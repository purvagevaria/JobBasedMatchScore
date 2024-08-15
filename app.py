import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the PDF file
    pdf_document = fitz.open(stream=pdf_file, filetype='pdf')
    text = ""
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    pdf_document.close()
    return text

# Function to compute fuzzy match score
def compute_fuzzy_match_score(resume_text, job_description):
    fuzzy_score = fuzz.ratio(resume_text, job_description)  # Fuzzy matching score
    return fuzzy_score

# Function to generate recommendations based on fuzzy match score
def generate_fuzzy_recommendations(score):
    recommendations = []
    if score < 50:
        recommendations.append("Consider using more relevant keywords.")
        recommendations.append("Fix typos and variations in your resume.")
    else:
        recommendations.append("Your resume is well-aligned with the job description.")
    
    recommendations_text = "\n".join([f"• {rec}" for rec in recommendations])
    return recommendations_text

# Function to generate a pie chart for fuzzy match score
def generate_chart(fuzzy_score):
    fuzzy_score = min(max(fuzzy_score, 0), 100)
    
    labels = ['Fuzzy Match Score', 'Other']
    sizes = [fuzzy_score, 100 - fuzzy_score]
    colors = ['#FFA500', '#E0E0E0']
    
    if len(labels) != len(sizes):
        raise ValueError("Labels and sizes must be of the same length")
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops=dict(color="black", fontsize=12),
                                      wedgeprops=dict(width=0.3, edgecolor='black'))
    ax.set_title('Fuzzy Match Score', fontsize=16, fontweight='bold', color='black')
    
    ax.legend(wedges, labels, title="Score Breakdown",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12, title_fontsize='13', shadow=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    buf_image = Image.open(buf)
    return buf_image

# Function to handle Gradio interface
def analyze(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)
    fuzzy_score = compute_fuzzy_match_score(resume_text, job_description)
    fuzzy_recommendations = generate_fuzzy_recommendations(fuzzy_score)
    chart_image = generate_chart(fuzzy_score)
    return f"Fuzzy Match Score: {fuzzy_score:.2f}%", fuzzy_recommendations, chart_image

# Gradio interface
def gradio_interface():
    with gr.Blocks(css="""
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Open+Sans:wght@300;400;700&display=swap');

        #analyze-button {
            background: linear-gradient(45deg, #FF6347, #FFD700);
            color: white;
            border-radius: 8px;
            font-size: 14px;
            padding: 8px 16px;
            border: none;
            width: 120px;
            height: 40px;
            font-family: 'Roboto', sans-serif;
        }
        #analyze-button:hover {
            background: linear-gradient(45deg, #FF4500, #FFA500);
        }
        .gradio-container {
            font-family: 'Open Sans', sans-serif;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        #job-description {
            height: 120px;
            overflow-y: auto;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ccc;
            background-color: #FFF8DC;
        }
        .output-box {
            background-color: #fff;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-size: 14px;
            margin-top: 10px;
            font-family: 'Roboto', sans-serif;
        }
        .output-box h2 {
            color: #FF6347;
            font-weight: bold;
        }
        #recommendations-output {
            font-family: 'Open Sans', sans-serif;
            font-size: 16px;
            color: #333333;
            line-height: 1.6;
        }
        #chart-container {
            background-color: #E0FFFF;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
    """) as demo:
        gr.Markdown("# 🎨 Job Matching Dashboard")
        with gr.Row():
            resume_input = gr.File(label="📄 Upload Resume (PDF)", type="binary", elem_id="resume-input")
            job_description_input = gr.Textbox(label="📝 Paste Job Description Text", lines=8, elem_id="job-description")
        with gr.Row():
            submit_button = gr.Button("Analyze", elem_id="analyze-button")
        with gr.Row():
            output = gr.Textbox(label="Fuzzy Match Score", elem_classes="output-box")
            recommendations_output = gr.Textbox(label="Recommendations", elem_classes="output-box", elem_id="recommendations-output")
        with gr.Row():
            chart_output = gr.Image(type="pil", label="Match Score Chart", elem_id="chart-container")
        
        submit_button.click(analyze, inputs=[resume_input, job_description_input], outputs=[output, recommendations_output, chart_output])
    
    return demo


if __name__ == "__main__":
    gradio_interface().launch(share=True)

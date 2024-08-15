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
    
    # Convert recommendations to bullet points
    recommendations_text = "\n".join([f"• {rec}" for rec in recommendations])
    return recommendations_text

# Function to generate a pie chart
def generate_chart(score):
    labels = ['Match Score', 'Remaining']
    sizes = [score, 100 - score]
    colors = ['#FFA500', '#FFFFFF']  # Orange and white for the pie chart
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops=dict(color="black", fontsize=12),
                                      wedgeprops=dict(width=0.3, edgecolor='black'))  # Black borders
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
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Open+Sans:wght@300;400;700&display=swap');

        #analyze-button {
            background: linear-gradient(45deg, #FF6347, #FFD700);  /* Gradient background */
            color: white;               /* White text */
            border-radius: 8px;         /* Rounded corners */
            font-size: 14px;            /* Font size */
            padding: 8px 16px;          /* Padding around the text */
            border: none;               /* Remove border */
            width: 120px;               /* Set button width */
            height: 40px;               /* Set button height */
            font-family: 'Roboto', sans-serif; /* Font family */
        }
        #analyze-button:hover {
            background: linear-gradient(45deg, #FF4500, #FFA500);  /* Darker gradient on hover */
        }
        .gradio-container {
            font-family: 'Open Sans', sans-serif; /* Custom font */
            background-color: #f9f9f9;      /* Light background */
            padding: 20px;                  /* Padding around the container */
            border-radius: 10px;            /* Rounded container corners */
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
        #job-description {
            height: 120px;                 /* Shortened height */
            overflow-y: auto;              /* Enable vertical scrolling */
            padding: 8px;                  /* Reduced padding */
            border-radius: 5px;            /* Rounded corners */
            border: 1px solid #ccc;        /* Light grey border */
            background-color: #FFF8DC;     /* Light color background for contrast */
        }
        .output-box {
            background-color: #fff;        /* White background */
            padding: 10px;                 /* Padding */
            border-radius: 5px;            /* Rounded corners */
            border: 1px solid #ccc;        /* Light grey border */
            font-size: 14px;               /* Font size */
            margin-top: 10px;              /* Space between elements */
            font-family: 'Roboto', sans-serif; /* Font family */
        }
        .output-box h2 {
            color: #FF6347;                /* Title color */
            font-weight: bold;             /* Bold title */
        }
        #recommendations-output {
            font-family: 'Open Sans', sans-serif; /* Custom font */
            font-size: 16px;               /* Larger font size */
            color: #333333;                /* Darker text color */
            line-height: 1.6;              /* Increased line spacing for readability */
        }
        #chart-container {
            background-color: #E0FFFF;     /* Light background for chart */
            border-radius: 8px;            /* Rounded corners */
            padding: 15px;                 /* Padding */
            margin-top: 20px;              /* Space above the chart */
        }
    """) as demo:
        gr.Markdown("# 🎨 Job Matching Dashboard")
        with gr.Row():
            resume_input = gr.File(label="📄 Upload Resume (PDF)", type="binary", elem_id="resume-input")
            job_description_input = gr.Textbox(label="📝 Paste Job Description Text", lines=8, elem_id="job-description")
        with gr.Row():
            submit_button = gr.Button("Analyze", elem_id="analyze-button")
        with gr.Row():
            output = gr.Textbox(label="Match Score", elem_classes="output-box")
            recommendations_output = gr.Textbox(label="Recommendations", elem_classes="output-box", elem_id="recommendations-output")
        with gr.Row():
            chart_output = gr.Image(type="pil", label="Match Score Chart", elem_id="chart-container")
        
        submit_button.click(analyze, inputs=[resume_input, job_description_input], outputs=[output, recommendations_output, chart_output])
    
    return demo


if __name__ == "__main__":
    gradio_interface().launch(share=True)

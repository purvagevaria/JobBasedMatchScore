from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader  # Updated import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)  # Updated to PdfReader
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def calculate_similarity(resume_text, job_desc_text):
    documents = [resume_text, job_desc_text]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_score[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    job_desc = request.form['jobDesc']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract text from the uploaded resume PDF
    resume_text = extract_text_from_pdf(file_path)
    
    # Calculate similarity score
    match_score = calculate_similarity(resume_text, job_desc)

    # Generate suggestions (simplified for demonstration)
    suggestions = []
    if match_score < 0.7:
        suggestions = [
            "Highlight more relevant skills.",
            "Add specific experiences that match the job description.",
            "Use keywords from the job description."
        ]

    return jsonify({'matchScore': match_score * 100, 'suggestions': suggestions})

if __name__ == "__main__":
    app.run(debug=True)

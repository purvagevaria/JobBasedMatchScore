# Job Matching Tool

This project is a web-based application designed to help users evaluate how well their resumes match a job description. It extracts text from a resume in PDF format and compares it to a job description using both fuzzy matching and rule-based recommendation systems. The tool provides a similarity score and actionable suggestions to improve the resume.

## Features

- **Job Description Input:** Enter a job description to compare with the resume.
- **Match Score:** Get a similarity score indicating how well your resume matches the job description using both fuzzy matching and rule-based methods.
- **Recommendations:** Receive actionable suggestions based on the job description to improve your resume.
- **Visualization:** View a pie chart representing the match score visually.

## Dependencies

The project requires the following Python packages:

- `gradio`: For creating the web interface.
- `fuzzywuzzy`: For fuzzy string matching.
- `python-Levenshtein`: (Optional) For faster fuzzywuzzy processing.
- `matplotlib`: For generating pie charts and other plots.
- `Pillow`: For image processing.
- `PyMuPDF`: For extracting text from PDF files.

# Run flask application
Run your flask application by following
![image](https://github.com/user-attachments/assets/474c138a-b835-4f47-8b5a-f28fa61d868c)


## Installation

You can install the required dependencies using pip:

```bash
pip install gradio
pip install fuzzywuzzy
pip install python-Levenshtein  # Optional, for faster fuzzywuzzy processing
pip install matplotlib
pip install pillow
pip install pymupdf
```

# Usage
- Upload Resume: Click on the file input to upload a resume in PDF format.
- Enter Job Description: Paste the job description into the text area.
- Analyze: Click the "Analyze" button to process the resume and job description.
- View Results: Check the match score and recommendations. The pie chart provides a visual representation of the match score.

# Block diagram
<img src="https://github.com/user-attachments/assets/e3f8dca0-daa6-436f-bddb-e7840881dadc" alt="drawing" style="width:200px;"/>


# Notes
- Ensure that the resume is in PDF format and properly formatted for optimal text extraction.
- The tool provides both fuzzy and rule-based recommendations to help you improve your resume based on the job description.
- The pie chart visualization helps in understanding the match score in a graphical format.

## Version History

| Version | Description                                       |
|---------|---------------------------------------------------|
| v1.0.0  | Initial release of a JavaScript-based application. |
| v1.0.1  | - Integrated Gradio for enhanced user interaction.<br>- Implemented a rule-based recommendation system. |
| v1.0.2  | - Added fuzzy logic implementation alongside rule-based recommendations. |


## Demo Video
<video src='https://github.com/user-attachments/assets/78553915-355e-4959-a7ad-65efef385aaf' width="100" height="auto" controls />

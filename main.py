from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
import PyPDF2 as pdf
import google.generativeai as genai
import json
from dotenv import load_dotenv
import re
import hashlib
import traceback
import ast

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes
cache = {}

def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def get_gemini_response(prompt, retry=False):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    response = model.generate_content(
        prompt if not retry else prompt.replace("Act as an expert ATS system", "You are a strict recruiter"),
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048
        )
    )
    return response.text

def extract_json_block(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            try:
                return ast.literal_eval(match.group())
            except:
                return {}
    return {}

def extract_score_from_structure_response(text):
    parsed = extract_json_block(text)
    if "JD Match" in parsed:
        score_text = parsed.get("JD Match", "0").replace("%", "")
        try:
            return float(score_text)
        except:
            return 0
    return 0

def hash_inputs(resume_text, jd):
    combined = resume_text + jd
    return hashlib.md5(combined.encode()).hexdigest()

@app.route("/evaluate", methods=["POST"])
def evaluate_resume():
    try:
        if 'file' not in request.files or 'job_description' not in request.form:
            return jsonify({"error": "Missing required fields"}), 400

        uploaded_file = request.files['file']
        jd = request.form['job_description']

        resume_text = extract_text_from_pdf(uploaded_file)
        input_hash = hash_inputs(resume_text, jd)

        if input_hash in cache:
            return jsonify(cache[input_hash])

        structure_prompt = f"""
Act as an expert ATS system. Compare this resume and job description. Score based on these criteria:

- Keyword Match (40% weight): Does the resume include key job-specific terms from the JD?
- Job Title & Role Match (20% weight): Does the resume have a matching or similar role?
- Skills & Tools Match (20% weight): Are the required tools/skills mentioned?
- Soft Skills & Experience Fit (20% weight): Evaluate any relevant professional traits such as adaptability, teamwork, time management, critical thinking, creativity, leadership, communication, or others mentioned in the job description or evident in the resume.

Output JSON in this format:
{{
  "JD Match": "<final % score based on above breakdown>"
}}

Resume:
{resume_text}

Job Description:
{jd}
"""

        ats_structure_result = get_gemini_response(structure_prompt)
        structure_score = extract_score_from_structure_response(ats_structure_result)

        if structure_score == 60:
            retry_result = get_gemini_response(structure_prompt, retry=True)
            structure_score = extract_score_from_structure_response(retry_result)

        final_score = round(structure_score, 2)
        result = {
            "Final_ATS_Score_Percentage": f"{final_score}%"
        }

        cache[input_hash] = result
        return jsonify(result)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

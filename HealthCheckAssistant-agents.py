import google.generativeai as generativeai
import google.genai as genai
import os
from google.colab import userdata

# 1. Setup (Get your free API key from aistudio.google.com)
#os.environ["GOOGLE_API_KEY"] = "YOUR_FREE_API_KEY_HERE"
generativeai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# Use Pro for the logic
model = generativeai.GenerativeModel('gemini-pro-latest')


def run_health_advisor(pdf_file_path):

    # --- AGENT 1: PARSER (Multimodal) ---
    print("... Agent 1 (Parser) is reading the file ...")
    file_upload = generativeai.upload_file(pdf_file_path)

    parser_prompt = """
    Extract all blood test results from this file.
    Return a structured list of: Test Name, Value, Units, Flag (High/Low).
    Only include tests flagged as Abnormal.
    """
    # agent_1_output = model.generate_content([parser_prompt, file_upload])
    # For demo simplicity, let's assume we get the text. In real code, you'd parse JSON.
    extracted_data = model.generate_content([parser_prompt, file_upload]).text
    print(f"Data Found: {extracted_data}")

    # --- AGENT 2: RESEARCHER (with Search Tool) ---
    print("... Agent 2 (Researcher) is looking up definitions ...")

    # Enable Google Search Tool for this call
    research_model = generativeai.GenerativeModel("gemini-pro-latest",tools="code_execution")
    research_prompt = f"""
    Based on these results: {extracted_data}
    Search 'site:labtestsonline.org.uk' to define what these tests are and what causes these specific high/low values.
    """
    research_data = research_model.generate_content(research_prompt).text

    # --- AGENT 3 & 4: WRITER & CRITIC ---
    print("... Agent 3 & 4 (Writer/Reviewer) are drafting ...")

    writer_prompt = f"""
    You are a Medical Writer and a Safety Reviewer.

    RAW DATA: {extracted_data}
    RESEARCH: {research_data}

    Task 1 (Write): Draft a calm, non-alarmist summary for the patient. Explain the results simply.
    Task 2 (Review): Check your own draft. Ensure you have NOT diagnosed a specific disease. Ensure numbers match the Raw Data.

    Output: Provide the Final Safe Summary.
    """
    final_response = model.generate_content(writer_prompt).text

    return final_response

# To run this, you would just upload a PDF to Colab and call:
# print(run_health_advisor("test_results.pdf"))
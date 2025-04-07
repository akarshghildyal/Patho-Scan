import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import os
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import streamlit as st

load_dotenv()

# Extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")
    return text

# Define models for API responses
class BloodTestAnalysisResponse(BaseModel):
    summary: str
    abnormal_values: list[dict]

class HealthIssueIdentificationResponse(BaseModel):
    potential_health_issues: list[str]

class LifestyleAdviceResponse(BaseModel):
    lifestyle_advice: list[str]
    medical_tests_recommendations: list[str]

# Define tools (agents)
def agent1(report_text: str, llm) -> dict:
    prompt1 = f"""
You are a Blood Test Analyst. Analyze the following blood test report, identify abnormalities, and summarize findings.

Provide a structured JSON response with this format:

{{
  "summary": "Brief summary of the findings",
  "abnormal_values": [
    {{
      "parameter": "Parameter Name",
      "value": "Recorded Value",
      "reference_range": "Normal Range",
      "interpretation": "What the abnormal value indicates"
    }}
  ]
}}

Report: {report_text}
"""
    raw_response = llm.invoke(prompt1)
    try:
        response_dict = json.loads(raw_response.content)
        parsed_response = BloodTestAnalysisResponse(**response_dict)
        return parsed_response.model_dump()
    except json.JSONDecodeError:
        print("Error parsing response from Agent 1.")
        return {}

def agent2(abnormal_values: list[dict], llm) -> dict:
    prompt2 = f"""
You are a Medical Condition Identifier. Analyze these abnormal values and identify potential health issues.

Provide a structured JSON response with this format:

{{
  "potential_health_issues": [
    "Description of potential health issue 1",
    "Description of potential health issue 2",
    ...
  ]
}}

Abnormal Values: {json.dumps(abnormal_values)}
"""
    raw_response = llm.invoke(prompt2)
    try:
        response_dict = json.loads(raw_response.content)
        parsed_response = HealthIssueIdentificationResponse(**response_dict)
        return parsed_response.model_dump()
    except json.JSONDecodeError:
        print("Error parsing response from Agent 2.")
        return {}

def agent3(agent2_output: dict, llm) -> dict:
    prompt3 = f"""
You are a Health Advisor. Based on the following potential health issues, provide lifestyle advice and recommended tests.

Potential Health Issues:
{json.dumps(agent2_output, indent=2)}

Provide a structured JSON response with this format:

{{
  "lifestyle_advice": ["List of lifestyle changes."],
  "medical_tests_recommendations": ["List of recommended tests or consultations."]
}}
"""
    raw_response = llm.invoke(prompt3)
    try:
        response_dict = json.loads(raw_response.content)
        parsed_response = LifestyleAdviceResponse(**response_dict)
        return parsed_response.model_dump()
    except json.JSONDecodeError:
        print("Error parsing response from Agent 3.")
        return {}

if __name__ == "__main__":
    # update the file path to your PDF file
    file_path = "<>"
    api_key = st.secrets["OPENROUTER_API_KEY"]

    # Initialize LLM
    llm = ChatOpenAI(
        model="openrouter/quasar-alpha",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
    )

    # Extract text from PDF
    report_text = extract_text_from_pdf(file_path)
    if not report_text:
        print("No text found in the PDF file. Please check the file and try again.")
        exit()

    # Define tools
    tools = [
        Tool(
            name="Blood Test Analysis",
            func=lambda report_text: agent1(report_text, llm),
            description="Useful for analyzing blood test reports and identifying abnormalities.",
        ),
        Tool(
            name="Health Issue Identification",
            func=lambda abnormal_values_str: agent2(json.loads(abnormal_values_str), llm),
            description="Useful for identifying potential health issues based on abnormal blood test values. Input should be a JSON string.",
        ),
        Tool(
            name="Lifestyle Advice and Recommendations",
            func=lambda health_issues_str: agent3(json.loads(health_issues_str), llm),
            description="Useful for providing lifestyle advice and recommending medical tests based on potential health issues. Input should be a JSON string.",
        ),
    ]

    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Run agent
    try:
        final_output = agent.run(f"Analyze the following blood test report: {report_text}")

        # Parse the final output (assuming it's a string containing the combined results)
        try:
            final_data = json.loads(final_output)
            lifestyle_advice = final_data.get("lifestyle_advice", [])
            medical_tests_recommendations = final_data.get("medical_tests_recommendations", [])

            print("\nLifestyle Advice:")
            for advice in lifestyle_advice:
                print(f"- {advice}")

            print("\nRecommended Tests:")
            for recommendation in medical_tests_recommendations:
                print(f"- {recommendation}")

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing final output: {e}")
            print("Raw final output:", final_output)

    except Exception as e:
        print(f"An error occurred during the agent run: {e}")

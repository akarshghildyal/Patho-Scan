from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os, json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st

load_dotenv()

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-thinking-exp:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["OPENROUTER_API_KEY"]
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

def clean_json_response(raw_response: str) -> str:
    lines = raw_response.strip().splitlines()
    if lines[0].strip().startswith("```") and lines[-1].strip().endswith("```"):
        return "\n".join(lines[1:-1])
    return raw_response

def extract_text_from_pdf(file) -> str:
    return "\n".join([page.extract_text() or "" for page in PdfReader(file).pages])

def blood_test_analysis_tool(input_text: str) -> str:
    prompt = f"""
You are a blood test analyst. Return JSON with:
- summary
- abnormal_values (list of dicts: parameter, value, reference_range, interpretation)

Report: {input_text}
"""
    response = llm.invoke(prompt)
    print("response.content from model:\n", response.content if response else "No response object")
    cleaned_response = clean_json_response(response.content) if response else "No response object"
    print("Cleaned response:\n", cleaned_response)
    return cleaned_response

def health_issue_identifier_tool(abnormal_values_json: str) -> str:
    prompt = f"""
You are a medical expert. Given these abnormal values, identify potential health issues.

Return JSON with:
- potential_health_issues (list of strings)

Abnormal Values: {abnormal_values_json}
"""
    response = llm.invoke(prompt)
    print("response.content from model:\n", response.content if response else "No response object")
    cleaned_response = clean_json_response(response.content) if response else "No response object"
    print("Cleaned response:\n", cleaned_response)
    return cleaned_response



def lifestyle_advice_tool(issues_json: str) -> str:
    prompt = f"""
You are a health advisor. Given these health issues, provide only actionable lifestyle recommendations as a *bullet list*. Be specific and concise, do not give any header. Avoid generic health info and give only recommendations.
Issues: {issues_json}
"""
    response = llm.invoke(prompt)
    print("response.content from model:\n", response.content if response else "No response object")
    # cleaned_response = clean_json_response(response.content) if response else "No response object"
    # print("Cleaned response:\n", cleaned_response)
    return response.content if response else "No response object"

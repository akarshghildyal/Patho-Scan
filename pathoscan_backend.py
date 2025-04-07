from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import os, json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
load_dotenv()

llm = ChatOpenAI(
    model="openrouter/quasar-alpha",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["OPENROUTER_API_KEY"]
)

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
    return response.content

def health_issue_identifier_tool(abnormal_values_json: str) -> str:
    prompt = f"""
You are a medical expert. Given these abnormal values, identify potential health issues.

Return JSON with:
- potential_health_issues (list of strings)

Abnormal Values: {abnormal_values_json}
"""
    response = llm.invoke(prompt)
    return response.content



def lifestyle_advice_tool(issues_json: str) -> str:
    prompt = f"""
You are a health advisor. Given these health issues, provide:
- A bullet list of lifestyle_advice (list)
Issues: {issues_json}
"""
    try:
        print("new prompt:", prompt)
        response = llm.invoke(prompt)
        print("Raw response from Quasar:\n", response.content if response else "No response object")
        return response.content if response and response.content.strip() else "⚠️ Agent 3 could not generate a response. Please try again later."
    except Exception as e:
        print("Error during LLM call:", e)
        return "⚠️ Agent 3 failed due to an error. Check the backend logs."
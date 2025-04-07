import streamlit as st
import json, os, re
from pathoscan_backend import extract_text_from_pdf, blood_test_analysis_tool, health_issue_identifier_tool, lifestyle_advice_tool

st.set_page_config(page_title="🧬 PathoScan", layout="centered")
st.title("🧬 PathoScan")
st.markdown("Upload a blood test report (PDF) and get insights from three agents working together:")

uploaded_file = st.file_uploader("Upload your blood test report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_text = extract_text_from_pdf("temp_uploaded.pdf")

    st.success("✅ PDF text extracted.")
    st.markdown("---")

    if st.button("🔍 Run PathoScan Analysis"):
        with st.status("Agent 1: Analyzing Blood Test...", expanded=True) as status1:
            try:
                agent1_output = blood_test_analysis_tool(extracted_text)
                agent1_data = json.loads(agent1_output)
                st.json(agent1_data)
                status1.update(label="Agent 1 Completed ✅", state="complete")
            except Exception as e:
                st.error("Agent 1 failed ❌")
                st.code(str(e))
                status1.update(label="Agent 1 Failed ❌", state="error")
                st.stop()

        st.markdown("---")

        with st.status("Agent 2: Identifying Health Issues...", expanded=True) as status2:
            try:
                abnormal_values_json = json.dumps(agent1_data["abnormal_values"])
                agent2_output = health_issue_identifier_tool(abnormal_values_json)
                agent2_data = json.loads(agent2_output)
                st.json(agent2_data)
                status2.update(label="Agent 2 Completed ✅", state="complete")
            except Exception as e:
                st.error("Agent 2 failed ❌")
                st.code(str(e))
                status2.update(label="Agent 2 Failed ❌", state="error")
                st.stop()

        st.markdown("---")

        with st.status("Agent 3: Providing Lifestyle Advice...", expanded=True) as status3:
            try:
                issues_json = json.dumps(agent2_data["potential_health_issues"])
                agent3_output = lifestyle_advice_tool(issues_json)

                if "⚠️" in agent3_output:
                    raise ValueError("Empty or failed response from LLM")

                # Clean markdown code blocks
                cleaned_output = re.sub(r"```(?:json)?\n([\s\S]*?)```", r"\1", agent3_output).strip()

                try:
                    # Try parsing as JSON first
                    agent3_data = json.loads(cleaned_output)
                    st.json(agent3_data)
                    status3.update(label="Agent 3 Completed ✅", state="complete")
                except json.JSONDecodeError:
                    # st.warning("⚠️ Agent 3 returned plain text instead of JSON. Displaying lifestyle section only.")

                    st.markdown("### 🧘 Lifestyle Advice")

                    # Try parsing Python-style list
                    lifestyle_list_match = re.search(r"lifestyle_advice\s*=\s*\[(.*?)\]", cleaned_output, re.DOTALL)
                    if lifestyle_list_match:
                        list_items_raw = lifestyle_list_match.group(1)
                        items = re.findall(r'"(.*?)"|\'(.*?)\'', list_items_raw)
                        formatted_items = [item[0] or item[1] for item in items if item[0] or item[1]]
                        for item in formatted_items:
                            st.markdown(f"- {item}")

                    # Otherwise, fallback to markdown-style bullets from plain text
                    elif "-" in cleaned_output:
                        bullet_lines = [line.lstrip("-• ").strip() for line in cleaned_output.splitlines() if line.strip().startswith("-")]
                        for line in bullet_lines:
                            st.markdown(f"- {line}")
                    else:
                        st.markdown("_No structured lifestyle advice found, but here’s the full text response:_")
                        st.markdown(f"```\n{cleaned_output}\n```")

                    # Add disclaimer
                    st.markdown("\n---\n")
                    st.markdown(
                        "📝 *Always discuss these recommendations with your healthcare provider, who can tailor advice and interpret test results in your specific context.*"
                    )

                    status3.update(label="Agent 3 Completed ✅ (Lifestyle Only)", state="complete")

            except Exception as e:
                st.error("Agent 3 failed ❌")
                st.code(str(e))
                status3.update(label="Agent 3 Failed ❌", state="error")
                st.stop()

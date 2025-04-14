import streamlit as st
import json, os, re
from pathoscan_backend import extract_text_from_pdf, blood_test_analysis_tool, health_issue_identifier_tool, lifestyle_advice_tool, personalized_chat_tool

st.set_page_config(page_title="🧬 PathoScan", layout="centered")
st.title("🧬 PathoScan")
st.markdown("Upload a blood test report (PDF) and get insights from three agents working together:")

uploaded_file = st.file_uploader("Upload your blood test report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        extracted_text = extract_text_from_pdf("temp_uploaded.pdf")
        st.session_state.extracted_text = extracted_text

    st.success("✅ PDF text extracted.")
    st.markdown("---")

    if st.button("🔍 Run PathoScan Analysis"):
        # Agent 1
        with st.status("Agent 1: Analyzing Blood Test...", expanded=True) as status1:
            try:
                agent1_output = blood_test_analysis_tool(st.session_state.extracted_text)
                agent1_data = json.loads(agent1_output)
                st.json(agent1_data)
                st.session_state.agent1_data = agent1_data
                status1.update(label="Agent 1 Completed ✅", state="complete")
            except Exception as e:
                st.error("Agent 1 failed ❌")
                st.code(str(e))
                status1.update(label="Agent 1 Failed ❌", state="error")
                st.stop()

        st.markdown("---")

        # Agent 2
        with st.status("Agent 2: Identifying Health Issues...", expanded=True) as status2:
            try:
                abnormal_values_json = json.dumps(agent1_data["abnormal_values"])
                agent2_output = health_issue_identifier_tool(abnormal_values_json)
                agent2_data = json.loads(agent2_output)
                st.json(agent2_data)
                st.session_state.agent2_data = agent2_data
                status2.update(label="Agent 2 Completed ✅", state="complete")
            except Exception as e:
                st.error("Agent 2 failed ❌")
                st.code(str(e))
                status2.update(label="Agent 2 Failed ❌", state="error")
                st.stop()

        st.markdown("---")

        # Agent 3
        with st.status("Agent 3: Providing Lifestyle Advice...", expanded=True) as status3:
            try:
                issues_json = json.dumps(agent2_data["potential_health_issues"])
                agent3_output = lifestyle_advice_tool(issues_json)

                if not agent3_output.strip():
                    raise ValueError("Empty response from LLM")

                cleaned_output = re.sub(r"```(?:json)?\n([\s\S]*?)```", r"\1", agent3_output).strip()
                st.session_state.agent3_raw_output = cleaned_output

                try:
                    agent3_data = json.loads(cleaned_output)
                    st.json(agent3_data)
                    st.session_state.agent3_data = agent3_data
                    status3.update(label="Agent 3 Completed ✅", state="complete")
                except json.JSONDecodeError:
                    st.markdown("### 🧘 Lifestyle Advice")

                    match = re.search(r"lifestyle_advice\s*=\s*\[(.*?)\]", cleaned_output, re.DOTALL)
                    if match:
                        list_items_raw = match.group(1)
                        items = re.findall(r'"(.*?)"|\'(.*?)\'', list_items_raw)
                        formatted_items = [item[0] or item[1] for item in items if item[0] or item[1]]
                        for item in formatted_items:
                            st.markdown(f"- {item}")
                        st.session_state.agent3_data = formatted_items
                    elif any(b in cleaned_output for b in ["-", "*", "•"]):
                        bullet_lines = []
                        for line in cleaned_output.splitlines():
                            stripped = line.strip(" -*•\t")
                            if stripped.endswith(("**:", ":**", ":")):
                                stripped = stripped.rstrip(":*")
                            bullet_lines.append(stripped)
                        for line in bullet_lines:
                            st.markdown(f"- {line}")
                        st.session_state.agent3_data = bullet_lines
                    else:
                        st.markdown("_No structured lifestyle advice found, but here’s the full text response:_")
                        st.markdown(f"```\n{cleaned_output}\n```")
                        st.session_state.agent3_data = cleaned_output

                    status3.update(label="Agent 3 Completed ✅", state="complete")

            except Exception as e:
                st.error("Agent 3 failed ❌")
                st.code(str(e))
                status3.update(label="Agent 3 Failed ❌", state="error")
                st.stop()

    # 👉 Show agent data from session state
    if "agent1_data" in st.session_state:
        st.markdown("### 🧪 Agent 1: Blood Test Analysis")
        st.json(st.session_state.agent1_data)

    if "agent2_data" in st.session_state:
        st.markdown("### 🧬 Agent 2: Health Issues Identified")
        st.json(st.session_state.agent2_data)

    if "agent3_data" in st.session_state:
        st.markdown("### 🧘 Agent 3: Lifestyle Advice")
        agent3_data = st.session_state.agent3_data

        if isinstance(agent3_data, list):
            for item in agent3_data:
                st.markdown(f"- {item}")
        elif isinstance(agent3_data, dict):
            st.json(agent3_data)
        else:
            st.markdown(f"```\n{agent3_data}\n```")

    st.markdown("---")
    st.subheader("💬 Ask a Question About Your Report")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Type your question here (e.g., 'Can I eat spinach?')", key="user_question")

    if st.button("Ask") and user_question:
        if not all(k in st.session_state for k in ["extracted_text", "agent1_data", "agent2_data", "agent3_data"]):
            st.error("Please run the full PathoScan analysis before asking questions.")
            st.stop()

        with st.spinner("Thinking..."):
            try:
                answer = personalized_chat_tool(
                    question=user_question,
                    extracted_text=st.session_state.extracted_text,
                    agent1_data=st.session_state.agent1_data,
                    agent2_data=st.session_state.agent2_data,
                    agent3_data=st.session_state.agent3_data
                )
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("AI", answer))
            except Exception as e:
                st.error("Failed to answer your question.")
                st.code(str(e))

    for role, msg in reversed(st.session_state.chat_history):
        if role == "AI":
            st.markdown(f"**🧠 AI:** {msg}")
        else:
            st.markdown(f"**👤 You:** {msg}")

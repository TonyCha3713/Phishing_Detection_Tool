import streamlit as st
import requests
from typing import Dict, Any, Optional

# Configure the page
st.set_page_config(
    page_title="Email Phishing Detection Tool",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://localhost:8000"

def analyze_eml_file(uploaded_file) -> Optional[Dict[str, Any]]:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
        response = requests.post(f"{API_URL}/analyze_eml", files=files, timeout=60)
        if response.ok:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def display_email_results(result: Dict[str, Any]):
    st.subheader("📧 Email Phishing Analysis Result")
    st.metric("Hybrid Risk Score (%)", f"{result['hybrid_risk_percent']}%")
    # Color-coordinated risk indicator
    risk = result['risk_indicator']
    if risk in ["Very Low", "Low"]:
        st.success(f"Risk Indicator: {risk}")
    elif risk == "Medium":
        st.warning(f"Risk Indicator: {risk}")
    else:  # High or Very High
        st.error(f"Risk Indicator: {risk}")
    st.write(f"**LLM Explanation:** {result['llm_explanation']}")
    st.write(f"**XGBoost Risk Score:** {result['xgboost_risk_score']:.4f}")
    st.write(f"**LLM Risk Score:** {result['llm_risk_score']:.4f}")

def main():
    st.title("🛡️ Email Phishing Detection Tool")
    st.markdown("---")
    st.header("Email File Upload (.eml)")
    uploaded_file = st.file_uploader("Upload a raw email file (.eml)", type=["eml"])
    if uploaded_file is not None:
        with st.spinner("Analyzing email..."):
            result = analyze_eml_file(uploaded_file)
        if result:
            display_email_results(result)

    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool uses advanced AI techniques to detect phishing attempts in email:
    - **Hybrid Model**: Combines XGBoost and LLM for email analysis
    - **LangChain & OpenAI** for semantic detection
    - **FastAPI (Backend) & Streamlit (Frontend)**
    """)

if __name__ == "__main__":
    main() 
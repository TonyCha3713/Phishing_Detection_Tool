import streamlit as st
import requests
import json
from typing import Dict, Any, Optional

# Configure the page
st.set_page_config(
    page_title="SMS Phishing Detection Tool",
    page_icon="🛡️",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

def analyze_sms(sms_text: str) -> Optional[Dict[str, Any]]:
    """Send SMS text to the API for analysis"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"text": sms_text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def main():
    st.title("🛡️ SMS Phishing Detection Tool")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool uses advanced AI techniques to detect SMS phishing attempts:
    
    - **Rule-based Analysis**: Checks for known SMS spam indicators
    - **Similarity Analysis**: Compares with known spam/legitimate SMS messages
    - **Feature Analysis**: Detects suspicious SMS text patterns
    
    Built with:
    - FastAPI (Backend)
    - Streamlit (Frontend)
    - Sentence Transformers (Embeddings)
    - Free Local Models
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("SMS Analysis")
        
        # SMS input
        sms_text = st.text_area(
            "Enter the SMS content to analyze:",
            height=200,
            placeholder="Paste the SMS content here..."
        )
        
        if st.button("🔍 Analyze SMS", type="primary"):
            if sms_text.strip():
                with st.spinner("Analyzing SMS..."):
                    result = analyze_sms(sms_text)
                    
                if result:
                    display_results(result)
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("Quick Test")
        st.markdown("Try these example SMS messages:")
        
        example_sms = {
            "Suspicious Account Alert": "Your account has been suspended. Click here to verify your identity immediately.",
            "Legitimate Meeting": "Hi team, just a reminder about our meeting at 3 PM today. Please prepare your updates.",
            "Urgent Bank Warning": "URGENT: Your bank account will be locked unless you update your password now!",
            "Prize Notification": "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
            "SMS Spam": "FREE RINGTONE text 12345 to claim your prize now! Limited time offer!",
            "Legitimate SMS": "Hi, just checking if you're still coming to dinner tonight?"
        }
        
        for title, sms in example_sms.items():
            if st.button(f"Test: {title}", key=title):
                st.session_state.test_sms = sms
                st.rerun()

def display_results(result: Dict[str, Any]):
    """Display the analysis results"""
    
    # Risk score visualization
    st.subheader("📊 Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_score = result['risk_score']
        st.metric("Overall Risk Score", f"{risk_score:.1%}")
        
        # Color-coded risk level
        if risk_score > 0.7:
            st.error("🔴 HIGH RISK - Likely SMS Spam/Phishing")
        elif risk_score > 0.4:
            st.warning("🟡 MEDIUM RISK - Suspicious")
        else:
            st.success("🟢 LOW RISK - Likely Legitimate")
    
    with col2:
        st.metric("Rule Score", f"{result['rule_score']:.1%}")
        st.metric("Similarity Score", f"{result['similarity_score']:.1%}")
    
    with col3:
        st.metric("Feature Score", f"{result['feature_score']:.1%}")
        st.metric("Confidence", result['confidence'].upper())
    
    # Analysis details
    st.subheader("🔍 Detailed Analysis")
    st.info(result['analysis'])
    
    # Similar messages
    if result['similar_messages']:
        st.subheader("📋 Similar Messages Found")
        
        for i, similar in enumerate(result['similar_messages'], 1):
            with st.expander(f"Similar Message {i} (Similarity: {similar['similarity']:.1%})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(similar['message'])
                with col2:
                    if similar['label'] == 'SPAM/PHISHING':
                        st.error("SPAM/PHISHING")
                    else:
                        st.success("LEGITIMATE")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if result['is_phishing']:
        st.error("""
        **⚠️ This appears to be SMS spam/phishing!**
        
        **Do NOT:**
        - Click any links in the SMS
        - Reply to the SMS
        - Call any numbers provided
        - Provide personal information
        
        **Do:**
        - Block the sender
        - Delete the SMS
        - Report to your carrier if needed
        - Be extra cautious with similar messages
        """)
    else:
        st.success("""
        **✅ This appears to be legitimate**
        
        **Still recommended:**
        - Verify the sender's number
        - Be cautious with any links
        - Contact the sender directly if unsure
        """)

if __name__ == "__main__":
    main() 
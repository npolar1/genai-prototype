import os
import re
from matplotlib import get_data_path
import pandas as pd
import streamlit as st  
from groq import Groq

def get_data_path():
    # Placeholder function to simulate data path retrieval
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, 'data', 'customer_reviews.csv')
    return csv_file_path

def clean_text(text):
    """
    Cleans the input text by removing punctuation, converting to lowercase, and stripping whitespace.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing whitespace
    return text


@st.cache_data
def get_chat_completion(user_message, temp):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        temperature=temp,
        model="llama-3.3-70b-versatile",
        max_completion_tokens=256
    )
    return response

client = Groq(
    api_key= st.secrets["GROQ_API_KEY"]
)

st.title("Groq LLM with Streamlit")
st.write("This is a simple app to demonstrate the Groq LLM API with Streamlit.")

temp = st.slider("Model temperature:",
                 min_value=0.0,
                  max_value= 1.0, 
                  value=0.7,
                   step= 0.1,
                   help="Controls the randomness of the model's output. Higher values make the output more random."
                )

user_message = st.text_area("Enter your message:", "Hello, Groq LLM!")

with st.spinner("Generating response..."):
    chat_completion = get_chat_completion(user_message, temp)

st.write(chat_completion.choices[0].message.content)

col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest DataSet"):
        try:
            csv_file_path = get_data_path()  # Ensure this function is defined elsewhere
            st.session_state['df'] = pd.read_csv(csv_file_path)
            st.success("DataSet ingested successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    if st.button("Parse reviews"):
        if 'df' in st.session_state:
            try:
                st.session_state['df']['cleaned_review'] = st.session_state['df']['SUMMARY'].apply(clean_text)
                st.success("Reviews parsed and cleaned successfully!")
            except Exception as e:
                st.error(f"Error: {e}")


if 'df' in st.session_state:
    st.subheader(f"Filter by Product")
    product_list = st.selectbox("Select a product:", ["All Product"] + list(st.session_state['df']['PRODUCT'].unique()))
    st.subheader(f"Data Preview")
    if product_list != "All Product":
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == product_list]
    else:
        filtered_df = st.session_state['df']
    st.dataframe(filtered_df)

    st.subheader("Sentiment Score by Product")
    grouped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)

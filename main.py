from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import requests
import streamlit as st
from langchain.llms import Together
# from langchain_openai import ChatOpenAI
import os

# Load environment variables from the .env file
load_dotenv()

# Load API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Validate Together AI API key
if not TOGETHER_API_KEY:
    st.error("Together AI API key is missing. Please add it to the .env file.")
    st.stop()

# Create a prompt template for generating tweets
tweet_template = "Give me {number} tweets on {topic} in {language}"
tweet_prompt = PromptTemplate(template=tweet_template, input_variables=['number', 'topic', 'language'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
together_model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Updated to use an available model
    temperature=0.7,
)
# AI model options
ai_models = {
    "Gemini (Google)": gemini_model,
    "Together AI": together_model,  # Placeholder for Together AI
}

# Streamlit App UI
st.header("Tweet Generator - Masu KasimAli")
st.subheader("🚀 Generate tweets using Generative AI")

# Input for topic and number of tweets
topic = st.text_input("Topic")
number = st.number_input("Number of tweets", min_value=1, max_value=10, value=1, step=1)

# Dropdown for selecting language
language = st.selectbox("Select language for tweets", ["English", "Spanish", "French", "German", "Hindi", "Gujarati"], index=0)

# Dropdown for selecting AI model
selected_ai = st.selectbox("Select the AI model", list(ai_models.keys()))

if st.button("Generate"):
    # Use the selected AI model
    chosen_model = ai_models[selected_ai]
    try:
        # Create a prompt using the template
        prompt = tweet_template.format(number=number, topic=topic, language=language)
        
        # Use the selected model to create an LLM chain
        tweet_chain = tweet_prompt | chosen_model
        
        # Generate tweets
        tweets = tweet_chain.invoke({"number": number, "topic": topic, "language": language})
        
        if selected_ai == "Together AI":
            # For Together AI, handle the response as a string
            st.success("Generated Tweets")
            st.write(tweets)  # Directly write the string response
        
        elif selected_ai == "Gemini (Google)":
            # For Google Gemini, handle the response as an object with 'content'
            st.success("Generated Tweets")
            st.write(tweets.content)
        
        else:
            st.error("Selected AI model is not yet implemented.")
    
    except Exception as e:
        st.error("Error generating tweets: " + str(e))
 

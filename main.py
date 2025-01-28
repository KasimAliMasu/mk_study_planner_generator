from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import requests
import streamlit as st
from langchain.llms import Together
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

# Create a prompt template for generating a study plan
study_plan_template = "Create a study plan for {topic} over {days} days, {hours_per_day} hours per day, in {language}"
study_plan_prompt = PromptTemplate(template=study_plan_template, input_variables=['topic', 'days', 'hours_per_day', 'language'])

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
st.header("Study Plan Generator - Masu KasimAli")
st.subheader("📚 Generate a personalized study plan using Generative AI")

# Input for topic, number of days, and hours per day
topic = st.text_input("Study Topic")
days = st.number_input("Number of days to learn", min_value=1, max_value=30, value=7, step=1)
hours_per_day = st.number_input("Hours per day", min_value=1, max_value=12, value=4, step=1)

# Dropdown for selecting language
language = st.selectbox("Select language for study plan", ["English", "Spanish", "French", "German", "Hindi", "Gujarati","chinese"], index=0)

# Dropdown for selecting AI model
selected_ai = st.selectbox("Select the AI model", list(ai_models.keys()))

if st.button("Generate Study Plan"):
    # Use the selected AI model
    chosen_model = ai_models[selected_ai]
    try:
        # Create a prompt using the template
        prompt = study_plan_template.format(topic=topic, days=days, hours_per_day=hours_per_day, language=language)
        
        # Use the selected model to create an LLM chain
        study_plan_chain = study_plan_prompt | chosen_model
        
        # Generate the study plan
        study_plan = study_plan_chain.invoke({"topic": topic, "days": days, "hours_per_day": hours_per_day, "language": language})
        
        if selected_ai == "Together AI":
            # For Together AI, handle the response as a string
            st.success("Generated Study Plan")
            st.write(study_plan)  # Directly write the string response
        
        elif selected_ai == "Gemini (Google)":
            # For Google Gemini, handle the response as an object with 'content'
            st.success("Generated Study Plan")
            st.write(study_plan.content)
        
        else:
            st.error("Selected AI model is not yet implemented.")
    
    except Exception as e:
        st.error("Error generating study plan: " + str(e))

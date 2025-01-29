from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import streamlit as st
from langchain.llms import Together
import os

# Load environment variables from the .env file
load_dotenv()

# Load API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Validate API keys
if not TOGETHER_API_KEY:
    st.error("⚠️ Together AI API key is missing. Please add it to the .env file.")
    st.stop()

# Create a prompt template for generating a study plan
study_plan_template = """
📖 **Personalized Study Plan**  

🔹 **Topic:** {topic}  
📅 **Duration:** {days} days  
⏳ **Daily Study Time:** {hours_per_day} hours  
🌍 **Language Preference:** {language}  

"""

study_plan_prompt = PromptTemplate(
    template=study_plan_template,
    input_variables=['topic', 'days', 'hours_per_day', 'language']
)

# Streamlit App UI
st.header("✨ Study Plan Generator - Masu KasimAli ✨")
st.subheader("📚 Generate a personalized study plan with the power of AI! 🌟")

# Dropdown for selecting language
language = st.selectbox("🌍 Choose language for your study plan", ["English", "Spanish", "French", "German", "Hindi", "Gujarati", "Chinese"], index=0)

# Dropdown for selecting AI model
selected_ai = st.selectbox("🤖 Choose an AI model", ["Gemini (Google)", "Meta-Llama 3.1"])

# Input for topic, number of days, and hours per day
topic = st.text_input("🔍 What would you like to study?")
days = st.number_input("⏳ Number of days to learn", min_value=1, max_value=30, value=7, step=1)
hours_per_day = st.number_input("⏰ Hours per day", min_value=1, max_value=12, value=4, step=1)

# Input for selecting temperature value (this controls creativity/randomness)
temperature = st.slider("🌡️ Select Temperature (Randomness)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    system_instruction="Only return a structured study plan. Do NOT add disclaimers, personal remarks, or unnecessary commentary.",
    temperature=temperature  # Add temperature here
)

# Initialize Together AI model (updated model name)
together_model = Together(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    temperature=temperature  # Add temperature here
)

# AI model options
ai_models = {
    "Gemini (Google)": gemini_model,
    "Meta-Llama 3.1": together_model,  # Updated the key here
}

if st.button("🚀 Generate Your Study Plan"):
    chosen_model = ai_models[selected_ai]
    try:
        # Create a prompt using the template
        prompt = study_plan_template.format(topic=topic, days=days, hours_per_day=hours_per_day, language=language)
        
        # Use the selected model to create an LLM chain
        study_plan_chain = study_plan_prompt | chosen_model
    
        # Generate the study plan
        study_plan = study_plan_chain.invoke({
            "topic": topic,
            "days": days,
            "hours_per_day": hours_per_day,
            "language": language
        })
        
        # **Filter out disclaimers manually**
        filtered_plan = study_plan
        if isinstance(study_plan, str):
            # Remove common disclaimers
            for unwanted_text in ["Please note that", "I do not speak"]:
                filtered_plan = filtered_plan.replace(unwanted_text, "")

        st.success("🎉 Study Plan Generated Successfully!")
        
        # Use st.markdown to correctly render the study plan with emojis
        # This is where the emoji rendering should be handled
        st.markdown(filtered_plan if isinstance(filtered_plan, str) else filtered_plan.content)

        # Encourage users to start their learning journey
        st.info("💡 Tip: Take breaks regularly! Your brain loves a bit of rest. 🌱")

    except Exception as e:
        st.error("❌ Error generating study plan: {str(e)}")

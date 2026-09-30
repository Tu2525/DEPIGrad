import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
try:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("No key found!")
        exit(1)
        
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say exactly: 'GEMINI API IS SUCCESSFULLY AUTHENTICATED AND WORKING!'")
    print(response.text.strip())
except Exception as e:
    print(f"Error testing Gemini: {e}")

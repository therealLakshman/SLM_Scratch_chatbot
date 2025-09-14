from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model_inference import Chatbot

# Initialize the FastAPI application
app = FastAPI(title="SLM Chatbot API", description="An API for a custom Small Language Model.")

# --- CORS (Cross-Origin Resource Sharing) ---
# This is crucial for allowing your frontend (running in a browser)
# to communicate with this backend server.
origins = [
    "*",  # Allows all origins for simplicity in development.
          # For production, you should restrict this to your frontend's domain.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Model Loading ---
# The model is loaded only ONCE when the server starts up.
# This is efficient and ensures fast response times.
print("Loading the SLM Chatbot model...")
chatbot = Chatbot()
print("Model loaded successfully.")

# --- Pydantic Model for Request Body ---
# This defines the expected structure of the data your API will receive.
class Prompt(BaseModel):
    text: str

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    """ A simple endpoint to check if the API is running. """
    return {"message": "Welcome to the SLM Chatbot API. Send a POST request to /generate to get a response."}

@app.post("/generate", tags=["Inference"])
def generate(prompt: Prompt):
    """
    This is the main endpoint for generating text.
    It receives a prompt and returns the model's response.
    """
    # The 'prompt' object will contain the text sent from the frontend.
    # The 'chatbot' object (from model.py) does the heavy lifting.
    response_text = chatbot.generate_response(prompt.text)
    return {"response": response_text}
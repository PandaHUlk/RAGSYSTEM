# capstoneproject
Setup Instructions
Follow these steps to set up and run the project:
Step 1: Create a Virtual Environment
Create and activate a virtual environment using virtualenv:
bash# Install virtualenv if not installed
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Step 2: Install Dependencies
Install the project and its dependencies:
bash# Install the project in development mode
pip install -e .
This will install all the required packages listed in requirements.txt.
Step 3: Setup Environment Variables (For OpenAI)
If you plan to use OpenAI models:

Copy .env.example to .env:
bashcp .env.example .env

Edit the .env file and add your OpenAI API key.

Step 4: Run the Application
Start the Streamlit application:
bashstreamlit run app.py
The web application should now be accessible at http://localhost:8501.
Using the Application

Initialize the System:

Select the model provider (Ollama for local models or OpenAI)
Choose the specific model (e.g., llama2, gpt-3.5-turbo)
Select vector store type (FAISS or Chroma)
Click "Initialize System"


Upload Documents:

Upload one or more PDF files using the file uploader
Click "Process Documents" to analyze, tag, and index them


View Document Topics:

Examine the topics automatically assigned to each document


Chat with Documents:

Ask questions about the uploaded documents
Optionally filter by topic or specific document
The system will retrieve relevant information and generate answers


Clear Chat History when needed to start a fresh conversation.
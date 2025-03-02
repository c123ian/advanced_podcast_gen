import modal
import os
import sqlite3
import uuid
from typing import Optional
import torch
from fasthtml.common import (
    fast_app, H1, P, Div, Form, Input, Button, Group,
    Title, Main, Audio
)
# Core PDF support - required
import PyPDF2

# Optional format support
import whisper

from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import WebBaseLoader
from common_image import common_image, shared_volume


import modal

# Ensure these functions reference the correct deployed app
generate_script = modal.Function.from_name("multi-file-podcast", "generate_script")
generate_audio = modal.Function.from_name("multi-file-podcast", "generate_audio")


# HELPER CLASSES



class BaseIngestor:
    """Base class for all ingestors"""
    def validate(self, source: str) -> bool:
        pass
    def extract_text(self, source: str, max_chars: int = 100000) -> Optional[str]:
        pass

class PDFIngestor(BaseIngestor):
    """PDF ingestion - core functionality"""
    def validate(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True
    def extract_text(self, file_path: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                extracted_text = []
                total_chars = 0
                for page_num in range(num_pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        total_chars += len(page_text)
                        if total_chars > max_chars:
                            remaining_chars = max_chars - total_chars
                            extracted_text.append(page_text[:remaining_chars])
                            print(f"Reached {max_chars} character limit at page {page_num + 1}")
                            break
                        extracted_text.append(page_text)
                        print(f"Processed page {page_num + 1}/{num_pages}")
                return "\n".join(extracted_text)
        except PyPDF2.PdfReadError:
            print("Error: Invalid or corrupted PDF file")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None

class WebsiteIngestor(BaseIngestor):
    """Website ingestion using LangChain's WebBaseLoader"""
    def validate(self, url: str) -> bool:
        if not url.startswith(('http://', 'https://')):
            print("Error: Invalid URL format")
            return False
        return True
    def extract_text(self, url: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(url):
            return None
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in documents])
            if len(extracted_text) > max_chars:
                extracted_text = extracted_text[:max_chars]
                print(f"Truncated extracted text to {max_chars} characters")
            print(f"Extracted text from website: {url}")
            return extracted_text
        except Exception as e:
            print(f"An error occurred while extracting from website: {str(e)}")
            return None

class AudioIngestor(BaseIngestor):
    """Audio ingestion using OpenAI's Whisper model"""
    def __init__(self, model_type: str = "base"):
        self.model_type = model_type
        self.model = whisper.load_model(self.model_type)
    def validate(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found at path: {audio_file}")
            return False
        if not audio_file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            print("Error: Unsupported audio format. Supported formats are .mp3, .wav, .flac, .m4a")
            return False
        return True
    def extract_text(self, audio_file: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(audio_file):
            return None
        try:
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            if len(transcription) > max_chars:
                transcription = transcription[:max_chars]
                print(f"Truncated transcription to {max_chars} characters")
            print(f"Transcribed audio file: {audio_file}")
            return transcription
        except Exception as e:
            print(f"An error occurred during audio transcription: {str(e)}")
            return None

class IngestorFactory:
    """Factory to create appropriate ingestor based on input type"""
    @staticmethod
    def get_ingestor(input_type: str, **kwargs) -> Optional[BaseIngestor]:
        input_type = input_type.lower()
        if input_type == "pdf":
            return PDFIngestor()
        elif input_type == "website":
            return WebsiteIngestor()
        elif input_type == "audio":
            return AudioIngestor(**kwargs)
        else:
            print(f"Unsupported input type: {input_type}")
            return None
        
class TextIngestor(BaseIngestor):
    """Simple ingestor for .txt or .md files."""
    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not (file_path.lower().endswith('.txt') or file_path.lower().endswith('.md')):
            print("Error: File is not .txt or .md")
            return False
        return True

    def extract_text(self, file_path: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if len(text) > max_chars:
                print(f"Truncating text to {max_chars} chars")
                text = text[:max_chars]
            print(f"Extracted text from: {file_path}")
            return text
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None


class IngestorFactory:
    @staticmethod
    def get_ingestor(input_type: str, **kwargs) -> Optional[BaseIngestor]:
        input_type = input_type.lower()
        if input_type == "pdf":
            return PDFIngestor()
        elif input_type == "website":
            return WebsiteIngestor()
        elif input_type == "audio":
            return AudioIngestor(**kwargs)
        elif input_type == "text":
            return TextIngestor()  # <-- new line here
        else:
            print(f"Unsupported input type: {input_type}")
            return None
        
# Create Modal App
app = modal.App("content_injection")

# Directories
UPLOAD_DIR = "/data/uploads"
OUTPUT_DIR = "/data/processed"
DB_PATH = "/data/injections.db"

# Ensure Directories Exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Database
def setup_database(db_path: str):
    """Initialize SQLite database for tracking injections"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injections (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            input_type TEXT NOT NULL,
            processed_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# Determine Input Type
def get_input_type(filename: str) -> str:
    """Classifies input type based on filename or URL"""
    lower_filename = filename.lower()
    if lower_filename.endswith('.pdf'):
        return 'pdf'
    elif lower_filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
        return 'audio'
    elif lower_filename.startswith(('http://', 'https://')):
        return 'website'
    elif lower_filename.endswith(('.txt', '.md')):
        return 'text'
    else:
        raise ValueError(f"Unsupported file type: {filename}")

# Process Uploaded Content
def process_content(source_path: str, input_type: str, max_chars: int = 100000) -> Optional[str]:
    """Processes content using the appropriate ingestor"""
    ingestor = IngestorFactory.get_ingestor(input_type)
    if not ingestor:
        print(f"❌ No ingestor found for type: {input_type}")
        return None
    return ingestor.extract_text(source_path, max_chars)

# Start Modal App with ASGI
@app.function(
    image=common_image,
    volumes={"/data": shared_volume},
    gpu=modal.gpu.T4(count=1),
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server"""
    conn = setup_database(DB_PATH)
    fasthtml_app, rt = fast_app()

    @rt("/")
    def homepage():
        """Render upload form with status checker"""
        upload_input = Input(
            type="file",
            name="content",
            accept=".pdf,.txt,.md,.mp3,.wav,.m4a,.flac",
            required=False
        )
        url_input = Input(
            type="text",
            name="url",
            placeholder="Or enter website URL",
            cls="w-full px-3 py-2 border rounded"
        )

        upload_form = Form(
            Group(
                upload_input,
                url_input,
                Button("Process Content"),
                cls="space-y-4"
            ),
            hx_post="/inject",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post",
        )
        
        status_form = Form(
            Group(
                Input(
                    type="text",
                    name="injection_id",
                    placeholder="Enter your podcast ID",
                    cls="w-full px-3 py-2 border rounded"
                ),
                Button("Check Status"),
                cls="space-y-4"
            ),
            action="/status-redirect",
            method="get",
        )

        return Title("Content Injection"), Main(
            H1("Upload Content for Podcast Generation"),
            P("Upload a file or provide a URL to process content for podcast generation."),
            upload_form,
            Div(id="injection-status"),
            Div(
                H1("Check Existing Podcast Status", cls="mt-8"),
                P("Already have a podcast ID? Check its status:"),
                status_form,
                cls="mt-10 pt-8 border-t"
            )
        )

    @rt("/inject", methods=["POST"])
    async def inject_content(request):
        """Handles content ingestion and starts the podcast generation pipeline."""
        form = await request.form()
        injection_id = uuid.uuid4().hex

        try:
            # Handle File Upload or URL
            if "content" in form and form["content"].filename:
                file_field = form["content"]
                original_filename = file_field.filename
                input_type = get_input_type(original_filename)
                save_path = os.path.join(UPLOAD_DIR, f"upload_{injection_id}{os.path.splitext(original_filename)[1]}")
                content = await file_field.read()
                with open(save_path, "wb") as f:
                    f.write(content)
            elif "url" in form and form["url"].strip():
                url = form["url"].strip()
                input_type = get_input_type(url)
                save_path = url
                original_filename = url
            else:
                return Div(
                    P("⚠️ Please select a file or provide a URL."),
                    id="injection-status",
                    cls="text-red-500"
                )

            # Extract Text Content
            processed_text = process_content(save_path, input_type)
            if not processed_text:
                return Div(
                    P("❌ Failed to process content"),
                    id="injection-status",
                    cls="text-red-500"
                )

            # Insert record into database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO injections (id, original_filename, input_type, status) VALUES (?, ?, ?, ?)",
                (injection_id, original_filename, input_type, "processing")
            )
            conn.commit()
            conn.close()

            # Call the Modal functions using `.remote(...)`
            print("🚀 Kicking off script generation...")
            script_pkl_path = generate_script.remote(processed_text)

            print("🔊 Kicking off audio generation...")
            # Generate the audio
            audio_path = generate_audio.remote(script_pkl_path, injection_id)
            
            # Return a processing message with ID
            return Div(
                P(f"✅ Content processed successfully! Your podcast is being generated."),
                P(f"Your podcast ID is: {injection_id}"),
                P("Copy this ID and use the status checker below to check when your podcast is ready."),
                P("It usually takes 3-5 minutes to generate the audio."),
                id="injection-status",
                cls="text-green-500"
            )

        except Exception as e:
            return Div(
                P(f"⚠️ Error processing content: {str(e)}"),
                id="injection-status",
                cls="text-red-500"
            )

    @rt("/status/{injection_id}")
    async def check_status(injection_id: str):
        """Check status and display audio if ready"""
        import base64
        
        # Check database for completion status
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT processed_path, status FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return Title("Podcast Status"), Main(
                H1("Podcast Status"),
                P(f"No record found for ID: {injection_id}"),
                cls="text-red-500"
            )
        
        audio_path, status = result
        
        if status != "completed" or not audio_path:
            return Title("Podcast Status"), Main(
                H1("Podcast Status"),
                P(f"Your podcast is still being generated. Status: {status}"),
                P("Please check back in a few minutes."),
                cls="text-blue-500"
            )
        
        # File is ready - display the audio player
        if os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                    b64_audio = base64.b64encode(audio_data).decode("ascii")
                    
                return Title("Your Podcast"), Main(
                    H1("Your Podcast is Ready!"),
                    P(f"ID: {injection_id}"),
                    Audio(
                        src=f"data:audio/wav;base64,{b64_audio}",
                        controls=True,
                        style="width: 100%;"
                    )
                )
            except Exception as e:
                return Title("Error"), Main(
                    H1("Error"),
                    P(f"Error loading audio file: {str(e)}"),
                    cls="text-red-500"
                )
        else:
            return Title("File Not Found"), Main(
                H1("File Not Found"),
                P(f"Audio file not found at expected location: {audio_path}"),
                cls="text-red-500"
            )

    @rt("/status-redirect")
    def status_redirect(injection_id: str):
        """Redirect to the status page for an ID"""
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"/status/{injection_id}")

    return fasthtml_app

if __name__ == "__main__":
    with modal.app.run():
        serve()  # Starts the FastHTML server with the correct function references

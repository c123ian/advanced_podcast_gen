import modal
import os
import sqlite3
import uuid
from typing import Optional
import torch
from fasthtml.common import *
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
        elif input_type == "text":
            return TextIngestor()
        else:
            print(f"Unsupported input type: {input_type}")
            return None


# Create Modal App
app = modal.App("content_injection")

# Directories
UPLOAD_DIR = "/data/uploads"
OUTPUT_DIR = "/data/processed"
DB_PATH = "/data/injections.db"  # <-- ensure consistent path

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
    cpu=8.0,
    # gpu=modal.gpu.T4(count=1),
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server"""
    conn = setup_database(DB_PATH)
    # Add DaisyUI and Tailwind CSS to headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        )
    )

    @rt("/")
    def homepage():
        """Render upload form with status checker"""
        # DaisyUI styled file input
        upload_input = Input(
            type="file",
            name="content",
            accept=".pdf,.txt,.md,.mp3,.wav,.m4a,.flac",
            required=False,
            cls="file-input file-input-secondary w-full"
        )
        
        # DaisyUI styled URL input with prefix label
        url_input_container = Div(
            Span(cls="bg-base-300 px-3 py-2 rounded-l-lg"),
            Input(
                type="text",
                name="url",
                placeholder="https://",
                cls="grow px-3 py-2 bg-base-300 rounded-r-lg focus:outline-none"
            ),
            cls="flex items-center w-full"
        )
        
        # Side-by-side layout with divider
        side_by_side = Div(
            # Left card - File upload
            Div(
                Div(
                    upload_input,
                    cls="grid place-items-center p-4"
                ),
                cls="card bg-base-300 rounded-box grow"
            ),
            # Divider
            Div("OR", cls="divider divider-horizontal"),
            # Right card - URL input
            Div(
                Div(
                    url_input_container,
                    cls="grid place-items-center p-4"
                ),
                cls="card bg-base-300 rounded-box grow"
            ),
            cls="flex w-full"
        )
        
        # Process button styled with DaisyUI
        process_button = Button(
            "Process Content", 
            cls="btn btn-primary w-full mt-4"
        )
        
        upload_form = Form(
            side_by_side,
            process_button,
            hx_post="/inject",
            hx_swap="afterbegin",
            enctype="multipart/form-data",
            method="post",
            cls="mb-6"
        )
        
        # Status checker form with DaisyUI styling
        status_form = Form(
            Div(
                Input(
                    type="text",
                    name="injection_id",
                    placeholder="Enter your podcast ID",
                    cls="input input-bordered w-full"
                ),
                Button("Check Status", cls="btn btn-secondary w-full mt-2"),
                cls="space-y-2"
            ),
            action="/status-redirect",
            method="get",
            cls="w-full"
        )

        return Title("AI Podcast Generator"), Main(
            Div(
                H1("Generate AI Podcast from Content", cls="text-2xl font-bold text-center mb-4"),
                P("Upload a file or provide a URL to process content for podcast generation.", 
                  cls="text-center mb-6"),
                upload_form,
                Div(id="injection-status", cls="my-4"),
                Div(
                    H1("Check Existing Podcast Status", cls="text-xl font-bold text-center mb-2"),
                    P("Already have a podcast ID? Check its status:", cls="text-center mb-4"),
                    status_form,
                    cls="mt-10 pt-8 border-t"
                ),
                cls="container mx-auto px-4 py-8 max-w-3xl"
            ),
            cls="min-h-screen bg-base-100"
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
                    Div(
                        P("⚠️ Please select a file or provide a URL."),
                        cls="alert alert-warning"
                    ),
                    id="injection-status"
                )

            # Extract Text Content
            processed_text = process_content(save_path, input_type)
            if not processed_text:
                return Div(
                    Div(
                        P("❌ Failed to process content"),
                        cls="alert alert-error"
                    ),
                    id="injection-status"
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
            
            # Create a status area that polls for updates using HTMX
            return Div(
                Div(
                    P(f"✅ Content processed successfully! Your podcast is being generated."),
                    cls="alert alert-success"
                ),
                Div(
                    P(f"Your podcast ID: ", 
                      Span(injection_id, cls="font-mono bg-base-300 px-2 py-1 rounded")),
                    cls="mt-2 mb-4"
                ),
                Div(
                    Div(
                        P("Checking status...", id="status-message"),
                        cls="animate-pulse"
                    ),
                    # HTMX polling
                    hx_get=f"/check-status-partial/{injection_id}",
                    hx_trigger="load delay:5s, every 10s",
                    hx_swap="innerHTML",
                    cls="p-4 border rounded-lg"
                ),
                id="injection-status"
            )

        except Exception as e:
            return Div(
                Div(
                    P(f"⚠️ Error processing content: {str(e)}"),
                    cls="alert alert-error"
                ),
                id="injection-status"
            )

    @rt("/check-status-partial/{injection_id}")
    async def check_status_partial(injection_id: str):
        """Check status and return HTML partial for the status area"""
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
            return Div(
                Div(
                    P(f"No record found for ID: {injection_id}"),
                    cls="alert alert-error"
                )
            )
        
        audio_path, status = result
        
        print(f"Status check for {injection_id}: path={audio_path}, status={status}")
        
        # If not completed yet, return status with continued polling
        if status != "completed" or not audio_path:
            return Div(
                Div(
                    P(f"Your podcast is still being generated. Status: {status}"),
                    cls="alert alert-info"
                ),
                P("This will update automatically when ready.", cls="mt-2 text-sm opacity-75"),
                # Continue polling
                hx_get=f"/check-status-partial/{injection_id}",
                hx_trigger="every 10s",
                hx_swap="innerHTML"
            )
        
        # File is ready - display the audio player and stop polling
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                    b64_audio = base64.b64encode(audio_data).decode("ascii")
                    
                return Div(
                    Div(
                        P("🎉 Your Podcast is Ready!"),
                        cls="alert alert-success"
                    ),
                    Div(
                        H2("Listen to your podcast:", cls="text-lg font-bold mb-2"),
                        P(f"ID: {injection_id}", cls="text-sm opacity-75 mb-2"),
                        Audio(
                            src=f"data:audio/wav;base64,{b64_audio}",
                            controls=True,
                            cls="w-full rounded-lg shadow"
                        ),
                        cls="mt-4 p-4 bg-base-200 rounded-lg"
                    )
                    # No more hx_trigger => polling stops
                )
            except Exception as e:
                print(f"Error for {injection_id}: {str(e)}")
                return Div(
                    Div(
                        P(f"Error loading audio file: {str(e)}"),
                        cls="alert alert-error"
                    )
                    # Stop polling on error
                )
        else:
            print(f"File not found at path: {audio_path}")
            return Div(
                Div(
                    P(f"Audio file not found. Status shows completed but file is missing."),
                    cls="alert alert-error"
                ),
                Button("Retry", hx_get=f"/check-status-partial/{injection_id}", 
                      hx_swap="innerHTML", cls="btn btn-sm mt-2")
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
                Div(
                    H1("Podcast Status", cls="text-2xl font-bold text-center mb-4"),
                    Div(
                        P(f"No record found for ID: {injection_id}"),
                        cls="alert alert-error"
                    ),
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                )
            )
        
        audio_path, status = result
        
        if status != "completed" or not audio_path:
            # If still processing, show status with auto-updating div
            return Title("Podcast Status"), Main(
                Div(
                    H1("Podcast Status", cls="text-2xl font-bold text-center mb-4"),
                    Div(
                        Div(
                            P(f"Your podcast is still being generated. Status: {status}"),
                            cls="alert alert-info"
                        ),
                        P("This page will automatically update when ready.", 
                          cls="mt-2 text-center text-sm opacity-75"),
                        # Add HTMX polling
                        hx_get=f"/check-status-partial/{injection_id}",
                        hx_trigger="load delay:1s, every 10s",
                        hx_swap="outerHTML",
                        cls="p-4 border rounded-lg"
                    ),
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                )
            )
        
        # File is ready - display the audio player
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                    b64_audio = base64.b64encode(audio_data).decode("ascii")
                    
                return Title("Your Podcast"), Main(
                    Div(
                        H1("Your Podcast is Ready!", cls="text-2xl font-bold text-center mb-4"),
                        Div(
                            P(f"ID: {injection_id}", cls="text-sm opacity-75 mb-2"),
                            Audio(
                                src=f"data:audio/wav;base64,{b64_audio}",
                                controls=True,
                                cls="w-full rounded-lg shadow"
                            ),
                            cls="p-6 bg-base-200 rounded-lg"
                        ),
                        cls="container mx-auto px-4 py-8 max-w-3xl"
                    )
                )
            except Exception as e:
                return Title("Error"), Main(
                    Div(
                        H1("Error", cls="text-2xl font-bold text-center mb-4 text-error"),
                        Div(
                            P(f"Error loading audio file: {str(e)}"),
                            cls="alert alert-error"
                        ),
                        cls="container mx-auto px-4 py-8 max-w-3xl"
                    )
                )
        else:
            return Title("File Not Found"), Main(
                Div(
                    H1("File Not Found", cls="text-2xl font-bold text-center mb-4 text-error"),
                    Div(
                        P(f"Audio file not found at expected location: {audio_path}"),
                        cls="alert alert-error mb-4"
                    ),
                    Button("Retry", hx_get=f"/status/{injection_id}", cls="btn btn-primary"),
                    cls="container mx-auto px-4 py-8 max-w-3xl text-center"
                )
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

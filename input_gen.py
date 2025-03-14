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

# Global config at module level - single consistent limit
MAX_CONTENT_CHARS = 45000  # ~18,750 tokens

class BaseIngestor:
    """Base class for all ingestors"""
    def validate(self, source: str) -> bool:
        pass

    def extract_text(self, source: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        pass
    
    def truncate_with_warning(self, text: str, max_chars: int) -> str:
        """Intelligently truncate text with warning message"""
        if len(text) <= max_chars:
            return text
            
        truncated = text[:max_chars]
        print(f"⚠️ Content truncated from {len(text)} to {max_chars} characters")
        
        # Add an explanatory note at the end
        truncation_note = "\n\n[Note: The original content was truncated due to length limitations.]"
        truncated = truncated[:max_chars - len(truncation_note)] + truncation_note
        
        return truncated


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

    def extract_text(self, file_path: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
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
                        extracted_text.append(page_text)
                        total_chars += len(page_text)
                        print(f"Processed page {page_num + 1}/{num_pages}, total chars: {total_chars}")
                        if total_chars > max_chars * 1.1:  # Read slightly more than needed
                            print(f"Reached character limit at page {page_num + 1}/{num_pages}")
                            break
                full_text = "\n".join(extracted_text)
                return self.truncate_with_warning(full_text, max_chars)
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

    def extract_text(self, url: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(url):
            return None
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in documents])
            print(f"Extracted {len(extracted_text)} chars from website: {url}")
            return self.truncate_with_warning(extracted_text, max_chars)
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

    def extract_text(self, audio_file: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(audio_file):
            return None
        try:
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            print(f"Transcribed {len(transcription)} chars from audio file: {audio_file}")
            return self.truncate_with_warning(transcription, max_chars)
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

    def extract_text(self, file_path: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
        if not self.validate(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Extracted {len(text)} chars from text file: {file_path}")
            return self.truncate_with_warning(text, max_chars)
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
UPLOAD_DIR = "/data/uploads_truncate"
OUTPUT_DIR = "/data/processed_truncate"
DB_PATH = "/data/injections_truncate.db"  # <-- ensure consistent path

# Ensure Directories Exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Setup NLTK for summarisation
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

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
            content_length INTEGER,
            processing_notes TEXT,
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
def process_content(source_path: str, input_type: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
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
        
        # Content limits information
        content_info = Div(
            P(f"Maximum content length: {MAX_CONTENT_CHARS//1000}K characters (longer content will be truncated)",
              cls="text-sm text-center opacity-70 mt-2")
        )
        
        # Process button (no spinner initially)
        process_button = Button(
            "Process Content",
            cls="btn btn-primary w-full mt-4",
            type="submit"
        )

        # Add script to handle loading state
        loading_script = Script("""
        document.addEventListener('htmx:beforeRequest', function(evt) {
            if (evt.target.matches('form')) {
                // Find the submit button
                var btn = evt.target.querySelector('button[type="submit"]');
                if (btn) {
                    // Save the original text
                    btn.dataset.originalText = btn.textContent;
                    // Replace with loading spinner
                    btn.innerHTML = '<span class="loading loading-spinner loading-lg text-secondary"></span>';
                    btn.disabled = true;
                }
            }
        });
        document.addEventListener('htmx:afterRequest', function(evt) {
            if (evt.target.matches('form')) {
                // Find the submit button
                var btn = evt.target.querySelector('button[type="submit"]');
                if (btn && btn.dataset.originalText) {
                    // Restore original text
                    btn.innerHTML = btn.dataset.originalText;
                    btn.disabled = false;
                }
            }
        });
        """)

        upload_form = Form(
            side_by_side,
            content_info,
            process_button,
            loading_script,
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

            # Insert record into database with content size info
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO injections 
                   (id, original_filename, input_type, status, content_length, processing_notes) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (injection_id, original_filename, input_type, "processing", 
                 len(processed_text), f"Original content: {len(processed_text)} chars")
            )
            conn.commit()
            conn.close()

            # Proper function sequence with correctly coupled inputs/outputs
            print("🚀 Kicking off script generation...")
            # First run script generation and get the result
            script_data = generate_script.remote(processed_text)
            
            # Then pass that directly to audio generation
            print("🔊 Kicking off audio generation...")
            generate_audio.spawn(script_data, injection_id)
            
            # Redirect to the generating page instead of returning HTML directly
            from starlette.responses import RedirectResponse
            return RedirectResponse(f"/generating/{injection_id}", status_code=303)

        except Exception as e:
            return Div(
                Div(
                    P(f"⚠️ Error processing content: {str(e)}"),
                    cls="alert alert-error"
                ),
                id="injection-status"
            )

    @rt("/generating/{injection_id}")
    def generating_podcast(injection_id: str):
        """Page with animation for generating podcast"""
        # Add the animation style
        animation_style = Style("""
        .animated-bg {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            height: 100vh;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .floating {
            animation: float 6s ease-in-out infinite;
            transform-origin: center;
        }
        
        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        """)
        
        # Check the status from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, content_length FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        status = result[0] if result else "unknown"
        content_length = result[1] if result and len(result) > 1 else "unknown"
        
        # Add some pulsing circles for visual interest
        circle_animation = Div(
            Div(cls="w-64 h-64 bg-purple-500 rounded-full absolute opacity-10 floating", 
                style="top: 10%; left: 10%;"),
            Div(cls="w-48 h-48 bg-blue-500 rounded-full absolute opacity-10 floating", 
                style="top: 60%; left: 70%;"),
            Div(cls="w-32 h-32 bg-green-500 rounded-full absolute opacity-10 floating", 
                style="top: 30%; left: 80%;"),
            cls="w-full h-full absolute top-0 left-0 overflow-hidden"
        )
        
        content_info = ""
        if content_length != "unknown":
            content_info = f", content size: {content_length} chars"
        
        return Title("Generating Your Podcast"), Main(
            animation_style,
            circle_animation,
            Div(
                Div(
                    H1("Your Podcast is Being Generated", cls="text-3xl font-bold text-center text-white mb-6"),
                    Div(
                        Div(cls="loading loading-dots loading-lg mb-6"),
                        P(f"Current status: {status}{content_info}", cls="text-lg mb-4 text-center text-white"),
                        P(f"Podcast ID: {injection_id}", cls="font-mono text-sm mb-6 text-center text-white"),
                        Div(
                            A(
                                "Check Status",
                                href=f"/status/{injection_id}",
                                cls="btn btn-primary btn-lg"
                            ),
                            cls="text-center"
                        ),
                        cls="p-8 rounded-lg bg-black bg-opacity-20 backdrop-blur-sm"
                    ),
                    A("← Back to Home", href="/", cls="btn btn-ghost mt-8 text-white"),
                    cls="flex flex-col items-center justify-center min-h-screen"
                ),
                cls="container mx-auto px-4 py-16 max-w-3xl relative z-10"
            ),
            cls="animated-bg"
        )
    
    @rt("/status/{injection_id}")
    async def check_status(injection_id: str):
        """Check status and display audio if ready"""
        import base64
        
        try:
            # Check database for completion status
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT processed_path, status, content_length, processing_notes FROM injections WHERE id = ?", 
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
                        A("← Back to Home", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
                        cls="container mx-auto px-4 py-8 max-w-3xl"
                    )
                )
            
            audio_path, status, content_length, processing_notes = result
            
            if status != "completed" or not audio_path:
                # Simple CSS animation without SVG
                animation_style = Style("""
                .pulse-bg {
                    background: linear-gradient(45deg, #f3ec78, #af4261, #3cdd8c, #5e9df9);
                    background-size: 400% 400%;
                    animation: gradient 15s ease infinite;
                }
                
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                """)
                
                # Content info for display
                content_info = ""
                if content_length:
                    content_info = f"Content size: {content_length} characters"
                    if content_length > MAX_CONTENT_CHARS:
                        content_info += f" (truncated from original size)"
                
                processing_info = ""
                if processing_notes:
                    processing_info = processing_notes
                
                # Show the "still processing" page with manual refresh
                return Title("Podcast Status"), Main(
                    animation_style,
                    Div(
                        H1("Podcast Status", cls="text-2xl font-bold text-center mb-4"),
                        Div(
                            P(f"Your podcast is still being generated. Status: {status}", 
                              cls="mb-4 text-center font-bold"),
                            P("Processing usually takes 30-60 minutes depending on content length.", 
                              cls="mb-6 text-center"),
                            P(content_info, cls="mb-2 text-sm text-center"),
                            P(processing_info, cls="mb-6 text-sm text-center"),
                            P(f"Podcast ID: {injection_id}", cls="font-mono text-sm mb-8 text-center"),
                            Button(
                                Span("Refresh Status", cls="mr-2"),
                                Span(cls="loading loading-spinner loading-xs"),
                                onClick="window.location.reload()",
                                cls="btn btn-primary block mx-auto"
                            ),
                            cls="p-6 bg-base-200 rounded-lg"
                        ),
                        A("← Back to Home", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
                        cls="container mx-auto px-4 py-8 max-w-3xl"
                    ),
                    cls="min-h-screen pulse-bg"
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
                                P(f"ID: {injection_id}", cls="text-sm opacity-75 mb-4 text-center"),
                                Div(
                                    H2("Listen to Your Podcast", cls="text-lg font-bold mb-3"),
                                    Audio(
                                        src=f"data:audio/wav;base64,{b64_audio}",
                                        controls=True,
                                        preload="auto",
                                        cls="w-full rounded-lg shadow mb-4"
                                    ),
                                    A(
                                        "Download Podcast", 
                                        href=f"data:audio/wav;base64,{b64_audio}",
                                        download=f"podcast_{injection_id}.wav",
                                        cls="btn btn-secondary w-full"
                                    ),
                                    cls="mb-6"
                                ),
                                cls="p-6 bg-base-200 rounded-lg"
                            ),
                            A("← Generate Another Podcast", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
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
                            Button(
                                "Try Again", 
                                onClick="window.location.reload()",
                                cls="btn btn-primary mt-4 block mx-auto"
                            ),
                            A("← Back to Home", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
                            cls="container mx-auto px-4 py-8 max-w-3xl"
                        )
                    )
            else:
                return Title("File Not Found"), Main(
                    Div(
                        H1("File Not Found", cls="text-2xl font-bold text-center mb-4 text-error"),
                        Div(
                            P(f"Audio file not found at expected location.", cls="mb-2"),
                            P("The file may have been removed or there was an error in processing.", 
                              cls="text-sm opacity-75"),
                            cls="alert alert-error mb-6"
                        ),
                        Button(
                            "Retry", 
                            onClick="window.location.reload()",
                            cls="btn btn-primary block mx-auto"
                        ),
                        A("← Back to Home", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
                        cls="container mx-auto px-4 py-8 max-w-3xl text-center"
                    )
                )
        except Exception as e:
            # Add detailed error handling to debug the 500 error
            print(f"Error in status page: {str(e)}")
            return Title("Error"), Main(
                Div(
                    H1("Error", cls="text-2xl font-bold text-center mb-4 text-error"),
                    Div(
                        P(f"An error occurred: {str(e)}"),
                        cls="alert alert-error"
                    ),
                    A("← Back to Home", href="/", cls="btn btn-ghost mt-6 block mx-auto"),
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                )
            )
    
    @rt("/status-redirect")
    def status_redirect(injection_id: str):
        """Redirect to the status page for an ID"""
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"/status/{injection_id}")

    @rt("/update-schema")
    def update_schema():
        """Update database schema if needed"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if columns exist first
        cursor.execute("PRAGMA table_info(injections)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'content_length' not in columns:
            cursor.execute("ALTER TABLE injections ADD COLUMN content_length INTEGER")
        
        if 'processing_notes' not in columns:
            cursor.execute("ALTER TABLE injections ADD COLUMN processing_notes TEXT")
            
        conn.commit()
        conn.close()
        
        return Title("Database Updated"), Main(
            Div(
                H1("Database Schema Updated", cls="text-2xl font-bold text-center mb-4"),
                P("The database schema has been updated to track content lengths.", cls="text-center mb-4"),
                A("← Back to Home", href="/", cls="btn btn-primary mt-4 block mx-auto"),
                cls="container mx-auto px-4 py-8 max-w-3xl"
            )
        )

    return fasthtml_app


if __name__ == "__main__":
    with modal.app.run():
        serve()  # Starts the FastHTML server with the correct function references

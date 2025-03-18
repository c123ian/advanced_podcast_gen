import modal
import os
import sqlite3
import uuid
from typing import Optional
import torch
import json
import asyncio
import base64
from fasthtml.common import *
# Core PDF support - required
import PyPDF2

# Optional format support
import whisper

from langchain_community.document_loaders import WebBaseLoader
from common_image import common_image, shared_volume

from starlette.responses import FileResponse, StreamingResponse

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
#NLTK_DATA_DIR = "/tmp/nltk_data"
#os.makedirs(NLTK_DATA_DIR, exist_ok=True)
#nltk.data.path.append(NLTK_DATA_DIR)
#nltk.download("punkt", download_dir=NLTK_DATA_DIR)
#nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

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

# Function to update injection status
def update_injection_status(injection_id, status, notes=None):
    """Update the status and optional notes for an injection"""
    if not injection_id:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Update status and/or notes
        if notes:
            cursor.execute(
                "UPDATE injections SET status = ?, processing_notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, notes, injection_id)
            )
        else:
            cursor.execute(
                "UPDATE injections SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, injection_id)
            )
            
        conn.commit()
        conn.close()
        print(f"✅ Updated status to '{status}' for ID: {injection_id}")
        
        if notes:
            print(f"📝 Notes: {notes}")
            
    except Exception as e:
        print(f"⚠️ Error updating injection status: {e}")

# Start Modal App with ASGI
@app.function(
    image=common_image,
    volumes={"/data": shared_volume},
    cpu=8.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server"""
    conn = setup_database(DB_PATH)
    # Add DaisyUI, Tailwind CSS, and HTMX SSE extension to headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
            Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),
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

            # Insert record into database with content size info and initial status
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO injections 
                   (id, original_filename, input_type, status, content_length, processing_notes) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (injection_id, original_filename, input_type, "pending", 
                 len(processed_text), f"Content ingested: {len(processed_text)} chars")
            )
            conn.commit()
            conn.close()

            # Update status to show script generation starting
            update_injection_status(injection_id, "processing", "Starting script generation...")

            # Proper function sequence with correctly coupled inputs/outputs
            print("🚀 Kicking off script generation...")
            # First run script generation and get the result
            script_data = generate_script.remote(processed_text)
            
            # Update status to show audio generation starting
            update_injection_status(injection_id, "processing", "Script generated. Starting audio generation...")
            
            # Then pass that directly to audio generation
            print("🔊 Kicking off audio generation...")
            generate_audio.spawn(script_data, injection_id)
            
            # Redirect to the generating page instead of returning HTML directly
            from starlette.responses import RedirectResponse
            return RedirectResponse(f"/generating-fixed/{injection_id}", status_code=303)

        except Exception as e:
            return Div(
                Div(
                    P(f"⚠️ Error processing content: {str(e)}"),
                    cls="alert alert-error"
                ),
                id="injection-status"
            )

    @rt("/podcast-updates-fixed/{injection_id}")
    async def podcast_updates_fixed(injection_id: str):
        """Fixed SSE endpoint that properly formats events for HTMX with debug logging"""
        
        async def event_generator():
            # Initial connection event
            print(f"SSE connection started for {injection_id}")
            yield "event: message\ndata: Connected to podcast updates\n\n"
            
            # Immediate status check
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
                (injection_id,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                status, processed_path, notes = result
                print(f"Initial status check: status={status}, path={processed_path}, notes={notes}")
                
                # Send immediate status update
                status_html = f"""
                <div id="status-indicator" class="{'loading loading-dots loading-lg mb-6' if status != 'completed' else 'text-success mb-6'}">
                    {'' if status != 'completed' else '<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>'}
                </div>
                <p id="status-message" class="text-lg mb-2 text-center text-white">Status: {status}</p>
                <p id="processing-notes" class="text-sm mb-4 text-center text-white">{notes}</p>
                <p class="font-mono text-sm mb-6 text-center text-white">Podcast ID: {injection_id}</p>
                """
                yield f"event: status\ndata: {status_html}\n\n"
                
                # If already completed, send audio player immediately
                if status == "completed" and processed_path:
                    # First verify the file exists
                    if os.path.exists(processed_path):
                        print(f"File exists at path: {processed_path}, sending audio player")
                        audio_html = f"""
                        <div class="p-6 bg-black bg-opacity-30 backdrop-blur-sm rounded-lg mt-4">
                            <h2 class="text-lg font-bold mb-3 text-white">Listen to Your Podcast</h2>
                            <audio src="/audio/{injection_id}" controls class="w-full rounded-lg shadow mb-4"></audio>
                            <a href="/audio/{injection_id}" download="podcast_{injection_id}.wav" 
                               class="btn btn-secondary w-full">Download Podcast</a>
                        </div>
                        """
                        yield f"event: audio\ndata: {audio_html}\n\n"
                    else:
                        print(f"File does not exist at path: {processed_path}")
                        
                        # Even if file doesn't exist, still show link to direct-audio page
                        direct_link_html = f"""
                        <div class="mt-4 p-4 bg-yellow-800 bg-opacity-50 rounded-lg">
                            <p class="text-white mb-2">Audio file not found at expected location.</p>
                            <a href="/direct-audio/{injection_id}" class="btn btn-warning w-full">Try Direct Access</a>
                        </div>
                        """
                        yield f"event: audio\ndata: {direct_link_html}\n\n"
            else:
                print(f"No record found for ID: {injection_id}")
                yield f"event: message\ndata: <div class='alert alert-warning'>No record found for this ID</div>\n\n"
                
            last_status = None
            last_notes = None
            sent_audio = False
            
            while True:
                try:
                    # Check database for status
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
                        (injection_id,)
                    )
                    result = cursor.fetchone()
                    conn.close()
                    
                    if not result:
                        print(f"No record found for ID: {injection_id} in SSE loop")
                        yield "event: message\ndata: <div class='alert alert-warning'>No record found for this ID</div>\n\n"
                        await asyncio.sleep(10)
                        continue
                    
                    status, processed_path, notes = result
                    
                    # Only send update if status or notes changed
                    if status != last_status or notes != last_notes:
                        print(f"Status changed: {last_status} -> {status}")
                        last_status = status
                        last_notes = notes
                        
                        # Send status update event
                        status_html = f"""
                        <div id="status-indicator" class="{'loading loading-dots loading-lg mb-6' if status != 'completed' else 'text-success mb-6'}">
                            {'' if status != 'completed' else '<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>'}
                        </div>
                        <p id="status-message" class="text-lg mb-2 text-center text-white">Status: {status}</p>
                        <p id="processing-notes" class="text-sm mb-4 text-center text-white">{notes}</p>
                        <p class="font-mono text-sm mb-6 text-center text-white">Podcast ID: {injection_id}</p>
                        """
                        print(f"Sending status update event: status={status}")
                        yield f"event: status\ndata: {status_html}\n\n"
                        
                        # If completed AND we have a valid path, send the audio player
                        if status == "completed" and processed_path and not sent_audio:
                            # First verify the file exists
                            if os.path.exists(processed_path):
                                print(f"File exists at path: {processed_path}, sending audio player")
                                audio_html = f"""
                                <div class="p-6 bg-black bg-opacity-30 backdrop-blur-sm rounded-lg mt-4">
                                    <h2 class="text-lg font-bold mb-3 text-white">Listen to Your Podcast</h2>
                                    <audio src="/audio/{injection_id}" controls class="w-full rounded-lg shadow mb-4"></audio>
                                    <a href="/audio/{injection_id}" download="podcast_{injection_id}.wav" 
                                       class="btn btn-secondary w-full">Download Podcast</a>
                                </div>
                                """
                                yield f"event: audio\ndata: {audio_html}\n\n"
                                sent_audio = True
                            else:
                                print(f"File does not exist at path: {processed_path}")
                                yield f"event: message\ndata: File not found at {processed_path}\n\n"
                    
                    # Sleep before next check
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"Error in SSE generator: {str(e)}")
                    yield f"event: error\ndata: <div class='alert alert-error'>Error: {str(e)}</div>\n\n"
                    await asyncio.sleep(10)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    @rt("/audio/{injection_id}")
    async def serve_audio(injection_id: str):
        """Serve audio file for a specific podcast with enhanced debugging"""
        print(f"📢 Audio request received for ID: {injection_id}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT processed_path FROM injections WHERE id = ? AND status = 'completed'", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            print(f"❌ No database record found for ID: {injection_id}")
            return Div(
                P("Audio file not found: No database record"),
                cls="alert alert-error"
            )
        
        if not result[0]:
            print(f"❌ Audio path is empty for ID: {injection_id}")
            return Div(
                P("Audio file not found: Empty file path"),
                cls="alert alert-error"
            )
        
        audio_path = result[0]
        print(f"📂 Looking for audio file at path: {audio_path}")
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file does not exist at path: {audio_path}")
            # List files in the podcast_audio directory to help debug
            audio_dir = os.path.dirname(audio_path)
            if os.path.exists(audio_dir):
                files = os.listdir(audio_dir)
                print(f"📂 Files in {audio_dir}: {files}")
                
                # Try to find a matching file by pattern
                matching_files = [f for f in files if injection_id in f]
                if matching_files:
                    print(f"📝 Found possible matching files: {matching_files}")
                    # Use the first matching file
                    alt_path = os.path.join(audio_dir, matching_files[0])
                    print(f"🔄 Using alternate file path: {alt_path}")
                    return FileResponse(
                        alt_path,
                        media_type="audio/wav", 
                        filename=f"podcast_{injection_id}.wav"
                    )
                    
            return Div(
                P(f"Audio file not found on disk: {os.path.basename(audio_path)}"),
                cls="alert alert-error"
            )
        
        # Additional file info
        file_size = os.path.getsize(audio_path)
        print(f"✅ Serving audio file: {audio_path} (size: {file_size} bytes)")
        
        # Serve the actual file
        return FileResponse(
            audio_path, 
            media_type="audio/wav",
            filename=f"podcast_{injection_id}.wav"
        )

    @rt("/direct-audio/{injection_id}")
    def direct_audio_player(injection_id: str):
        """Direct access to podcast audio for completed podcasts"""
        # Check database for status and audio path
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return Title("Audio Not Found"), Main(
                Div(
                    H1("Podcast Not Found", cls="text-2xl font-bold text-center text-error mb-4"),
                    P(f"No podcast found with ID: {injection_id}", cls="text-center mb-4"),
                    A("← Back to Home", href="/", cls="btn btn-primary block mx-auto"),
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                )
            )
        
        status, audio_path, notes = result
        
        # Check if file exists 
        file_exists = audio_path and os.path.exists(audio_path)
        
        # If file doesn't exist but status is completed, try to find alternate file
        if not file_exists and status == "completed" and audio_path:
            # List files in the podcast_audio directory to help debug
            audio_dir = os.path.dirname(audio_path)
            if os.path.exists(audio_dir):
                files = os.listdir(audio_dir)
                print(f"📂 Files in {audio_dir}: {files}")
                
                # Try to find a matching file by pattern
                matching_files = [f for f in files if injection_id in f]
                if matching_files:
                    print(f"📝 Found possible matching files: {matching_files}")
                    # Use the first matching file
                    audio_path = os.path.join(audio_dir, matching_files[0])
                    file_exists = True
                    print(f"🔄 Using alternate file path: {audio_path}")
                    
        file_size = os.path.getsize(audio_path) if file_exists else 0
        
        # Add animation style
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
        """)
        
        # Create audio element manually
        audio_element = NotStr(f'<audio src="/audio/{injection_id}" controls class="w-full rounded-lg shadow mb-4"></audio>')
        
        return Title("Your Podcast"), Main(
            animation_style,
            Div(
                H1("Your Podcast is Ready!", cls="text-3xl font-bold text-center text-white mb-6"),
                
                # Status display
                Div(
                    Div(
                        # Success icon - using a simple ✓ instead of SVG
                        Div(
                            Span("✓", cls="text-4xl"),
                            cls="text-success mb-6 mx-auto"
                        ),
                        
                        # Status text
                        P(f"Status: {status}", cls="text-lg mb-2 text-center text-white"),
                        P(notes or "Processing complete.", cls="text-sm mb-4 text-center text-white"),
                        P(f"Podcast ID: {injection_id}", cls="font-mono text-sm mb-6 text-center text-white"),
                        
                        cls="p-6 bg-black bg-opacity-20 rounded-lg mb-6 text-center"
                    ),
                    
                    # Audio player - only show if file exists
                    file_exists and Div(
                        H2("Listen to Your Podcast", cls="text-lg font-bold mb-3 text-white text-center"),
                        Div(
                            audio_element,
                            A("Download Podcast", href=f"/audio/{injection_id}", download=f"podcast_{injection_id}.wav", 
                            cls="btn btn-secondary w-full"),
                            P(f"File size: {file_size / 1024 / 1024:.2f} MB", cls="text-xs text-center mt-2 text-white opacity-70"),
                            cls="bg-black bg-opacity-30 p-4 rounded-lg"
                        ),
                        cls="mb-6"
                    ) or Div(
                        Div(
                            P("Audio file not found. The podcast may still be processing or there may have been an issue with generation.",
                            cls="text-warning text-center"),
                            cls="bg-black bg-opacity-30 p-4 rounded-lg"
                        ),
                        cls="mb-6"
                    ),
                    
                    # Return to home button
                    A("← Back to Home", href="/", cls="btn btn-primary block mx-auto"),
                    
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                ),
                cls="animated-bg min-h-screen"
            )
        )
        
        
        
    @rt("/generating-fixed/{injection_id}")
    def generating_podcast_fixed(injection_id: str):
        """Updated podcast generation page with proper SSE implementation"""
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
        """)
        
        # Debug script
        debug_script = Script("""
        document.addEventListener('htmx:sseMessage', function(event) {
            console.log('SSE event received:', event.detail);
        });
        
        document.addEventListener('htmx:sseError', function(event) {
            console.error('SSE error:', event.detail);
        });
        
        // Check DB status directly after a few seconds
        setTimeout(function() {
            // Add alternate access link if still on initializing
            var statusMsg = document.getElementById('status-message');
            if (statusMsg && statusMsg.innerText.includes('Initializing')) {
                var container = document.getElementById('direct-access');
                container.innerHTML = `
                    <div class="mt-6 p-4 bg-yellow-800 bg-opacity-50 rounded-lg">
                        <p class="text-white mb-3">If your podcast is ready but not showing up, access it directly:</p>
                        <a href="/direct-audio/${document.getElementById('injection-id').innerText}" 
                           class="btn btn-warning w-full">Access Podcast Directly</a>
                    </div>
                `;
            }
        }, 8000);
        """)
        
        # Check database for current status
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, processing_notes FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        initial_status = "pending"
        initial_notes = "Please wait while we generate your podcast..."
        completed = False
        
        if result:
            initial_status = result[0]
            initial_notes = result[1] or initial_notes
            completed = initial_status == "completed"
        
        return Title("Podcast Generator"), Main(
            animation_style,
            debug_script,
            Div(
                H1("Your Podcast is Being Generated" if not completed else "Your Podcast is Ready!", 
                   cls="text-3xl font-bold text-center text-white mb-6"),
                
                # Status container with HTMX SSE
                Div(
                    Div(
                        # Loading indicator (replaced by success icon when complete)
                        Div(cls=f"{'loading loading-dots loading-lg mb-6' if not completed else 'text-success mb-6'}", id="status-indicator"),
                        
                        # Status text
                        P(f"Status: {initial_status}", id="status-message", cls="text-lg mb-2 text-center text-white"),
                        P(initial_notes, id="processing-notes", cls="text-sm mb-4 text-center text-white"),
                        P(f"Podcast ID: {injection_id}", id="injection-id", cls="font-mono text-sm mb-6 text-center text-white"),
                        
                        # These attrs tell HTMX to listen for SSE events
                        hx_ext="sse",
                        sse_connect=f"/podcast-updates-fixed/{injection_id}",
                        sse_swap="status",
                        id="status-container", 
                        cls="p-6 bg-black bg-opacity-20 rounded-lg mb-6 text-center"
                    ),
                    
                    # Audio container (initially empty, filled when ready)
                    Div(
                        id="audio-container",
                        hx_ext="sse",
                        sse_connect=f"/podcast-updates-fixed/{injection_id}",
                        sse_swap="audio",
                    ),
                    
                    # Direct access container (filled by JS after a delay if needed)
                    Div(
                        id="direct-access",
                        cls="mt-4"
                    ),
                    
                    # If already completed, show direct link immediately
                    completed and Div(
                        Div(
                            P("For direct access to your podcast:", cls="text-white mb-3"),
                            A("Access Podcast Directly", href=f"/direct-audio/{injection_id}", 
                             cls="btn btn-warning w-full"),
                            cls="p-4 bg-yellow-800 bg-opacity-50 rounded-lg"
                        ),
                        cls="mt-6"
                    ),
                    
                    # Return to home button
                    A("← Back to Home", href="/", cls="btn btn-primary block mx-auto mt-6"),
                    
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                ),
                cls="animated-bg min-h-screen"
            )
        )

    @rt("/podcast-updates/{injection_id}")
    async def podcast_updates(injection_id: str):
        """SSE endpoint for real-time podcast generation updates using HTMX"""
        
        async def event_generator():
            # Initial connection event (no need to send this as HTMX event)
            yield f"data: Connection established for {injection_id}\n\n"
            
            last_status = None
            last_notes = None
            check_count = 0
            sent_completed = False
            
            while True:
                try:
                    # Check database for status
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
                        (injection_id,)
                    )
                    result = cursor.fetchone()
                    conn.close()
                    
                    if not result:
                        # Send error as actual HTML to be swapped
                        error_html = '<p class="text-error">Record not found</p>'
                        yield f"event: error\ndata: {error_html}\n\n"
                        break
                        
                    status, processed_path, notes = result
                    
                    # Only send update if status or notes changed
                    if status != last_status or notes != last_notes:
                        last_status = status
                        last_notes = notes
                        
                        # Update status container with direct HTML
                        status_html = f"""
                        <div id="status-indicator" class="{'loading loading-dots loading-lg mb-6' if status != 'completed' else 'text-success mb-6'}">
                            {'' if status != 'completed' else '<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>'}
                        </div>
                        <p id="status-message" class="text-lg mb-2 text-center text-white">Status: {status}</p>
                        <p id="processing-notes" class="text-sm mb-4 text-center text-white">{notes}</p>
                        <p class="font-mono text-sm mb-6 text-center text-white">Podcast ID: {injection_id}</p>
                        """
                        yield f"event: status\ndata: {status_html}\n\n"
                        
                        # If completed, also send the audio player HTML
                        if status == "completed" and processed_path and not sent_completed:
                            sent_completed = True
                            
                            # Send completed header - no quotes here to avoid escaping issues
                            title_html = '<h1 class="text-3xl font-bold text-center text-white mb-6">Your Podcast is Ready!</h1>'
                            yield f"event: title\ndata: {title_html}\n\n"
                            
                            # Send audio player HTML directly - no quotes here to avoid escaping issues
                            audio_html = f'''
                            <div class="p-6 bg-black bg-opacity-30 backdrop-blur-sm rounded-lg mt-4">
                                <h2 class="text-lg font-bold mb-3 text-white">Listen to Your Podcast</h2>
                                <audio src="/audio/{injection_id}" controls class="w-full rounded-lg shadow mb-4"></audio>
                                <a href="/audio/{injection_id}" download="podcast_{injection_id}.wav" 
                                   class="btn btn-secondary w-full">Download Podcast</a>
                            </div>
                            '''
                            yield f"event: complete\ndata: {audio_html}\n\n"
                            
                            # After sending completed data, close the connection after a delay
                            await asyncio.sleep(2)
                            yield f"event: close\ndata: Connection closed\n\n"
                            break
                    
                    # Keep connection alive with heartbeat periodically
                    elif check_count % 30 == 0 and check_count > 0:
                        # Just send a comment as heartbeat
                        yield f": heartbeat {check_count}\n\n"
                        
                    check_count += 1
                    
                    # Sleep before next check
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"Error in SSE generator: {str(e)}")
                    # Send error as actual HTML to be swapped
                    error_html = f'<p class="text-error">Error: {str(e)}</p>'
                    yield f"event: error\ndata: {error_html}\n\n"
                    await asyncio.sleep(10)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    @rt("/podcast-updates-simple/{injection_id}")
    async def podcast_updates_simple(injection_id: str):
        """Simplified SSE endpoint for HTMX - only sends plain text to swap"""
        
        async def event_generator():
            # Send initial message
            yield "event: message\ndata: Connecting to SSE stream...\n\n"
            
            last_status = None
            last_notes = None
            sent_completed = False
            
            while True:
                try:
                    # Check database for status
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
                        (injection_id,)
                    )
                    result = cursor.fetchone()
                    conn.close()
                    
                    if not result:
                        yield "event: message\ndata: No record found for this ID\n\n"
                        await asyncio.sleep(10)
                        continue
                    
                    status, processed_path, notes = result
                    
                    # Only send update if status or notes changed
                    if status != last_status or notes != last_notes:
                        last_status = status
                        last_notes = notes
                        
                        # For basic status updates - send VERY simple HTML
                        status_html = f"""
                        <span id="status-text">{status}</span>
                        <script>document.getElementById('notes-text').textContent = {json.dumps(notes)};</script>
                        """
                        yield f"event: message\ndata: {status_html}\n\n"
                        
                        # If completed, also send the audio player
                        if status == "completed" and processed_path and not sent_completed:
                            sent_completed = True
                            
                            # Send very simple audio player HTML
                            audio_html = f"""
                            <div class="bg-black bg-opacity-50 p-4 rounded-lg">
                            <p class="text-white mb-2">Your podcast is ready!</p>
                            <audio src="/audio/{injection_id}" controls class="w-full mb-2"></audio>
                            <a href="/audio/{injection_id}" download="podcast.wav" class="btn btn-sm btn-success w-full">Download</a>
                            </div>
                            """
                            yield f"event: audio\ndata: {audio_html}\n\n"
                    
                    # Sleep before next check
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"Error in simple SSE generator: {str(e)}")
                    yield f"event: message\ndata: Error: {str(e)}\n\n"
                    await asyncio.sleep(10)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    @rt("/podcast-updates-debug/{injection_id}")
    async def podcast_updates_debug(injection_id: str):
        """Raw debug SSE endpoint that just forwards database status as plain text"""
        
        async def event_generator():
            yield f"data: Debug SSE connected for ID: {injection_id}\n\n"
            
            while True:
                try:
                    # Check database for status
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, processed_path, processing_notes FROM injections WHERE id = ?", 
                        (injection_id,)
                    )
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        status, processed_path, notes = result
                        yield f"data: {{\"status\": \"{status}\", \"notes\": \"{notes}\", \"path\": \"{processed_path}\"}}\n\n"
                    else:
                        yield f"data: No record found\n\n"
                    
                    await asyncio.sleep(5)
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
                    await asyncio.sleep(10)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    @rt("/status/{injection_id}")
    async def check_status(injection_id: str):
        """Direct status page that redirects to the generating page with SSE updates"""
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"/generating-fixed/{injection_id}", status_code=303)
    
    @rt("/status-redirect")
    def status_redirect(injection_id: str):
        """Redirect to the fixed generating page for an ID"""
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"/generating-fixed/{injection_id}")

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

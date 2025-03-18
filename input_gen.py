import modal
import os
import sqlite3
import uuid
from typing import Optional
import torch
import json
import base64
from fasthtml.common import *
# Core PDF support - required
import PyPDF2

# Optional format support
import whisper

from langchain_community.document_loaders import WebBaseLoader
from common_image import common_image, shared_volume

from starlette.responses import FileResponse, StreamingResponse, HTMLResponse, RedirectResponse

import modal

# Ensure these functions reference the correct deployed app
generate_script = modal.Function.from_name("multi-file-podcast", "generate_script")
generate_audio = modal.Function.from_name("multi-file-podcast", "generate_audio")

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
DB_PATH = "/data/injections_truncate.db"
AUDIO_DIR = "/data/podcast_audio"  # Standard location for all audio files

# Ensure Directories Exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

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

# Function to get standardized audio file path
def get_audio_file_path(injection_id):
    """Returns a standardized path for audio files based on injection_id"""
    return os.path.join(AUDIO_DIR, f"podcast_{injection_id}.wav")

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
            action="/inject",
            method="post",
            enctype="multipart/form-data",
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

            # Set the standardized audio file path
            audio_file_path = get_audio_file_path(injection_id)

            # Insert record into database with content size info and initial status
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO injections 
                   (id, original_filename, input_type, status, content_length, processing_notes, processed_path) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (injection_id, original_filename, input_type, "pending", 
                 len(processed_text), f"Content ingested: {len(processed_text)} chars", audio_file_path)
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
            
            # Redirect to the podcast status page
            return RedirectResponse(f"/podcast-status/{injection_id}", status_code=303)

        except Exception as e:
            return Div(
                Div(
                    P(f"⚠️ Error processing content: {str(e)}"),
                    cls="alert alert-error"
                ),
                id="injection-status"
            )

    # The key changes are in the podcast_status route handler
# No need to change any other parts of the file

    @rt("/podcast-status/{injection_id}")
    def podcast_status(injection_id: str):
        """Simple status page that shows current progress and auto-refreshes until complete"""
        # First explicitly reload the volume to get latest changes
        shared_volume.commit()  # Ensure any pending changes are saved
        shared_volume.reload()  # Then reload to get latest state
        
        # Check database for current status
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, processed_path, processing_notes, created_at FROM injections WHERE id = ?", 
            (injection_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return Title("Podcast Not Found"), Main(
                Div(
                    H1("Podcast Not Found", cls="text-2xl font-bold text-center text-error mb-4"),
                    P(f"No podcast with ID: {injection_id} was found", cls="text-center mb-4"),
                    A("← Back to Home", href="/", cls="btn btn-primary block mx-auto"),
                    cls="container mx-auto px-4 py-8 max-w-3xl"
                )
            )
        
        status, audio_path, notes, created_at = result
        is_completed = status == "completed"
        
        # Log current state for debugging
        print(f"DEBUG: Podcast ID: {injection_id}, Status: {status}, Is Completed: {is_completed}")
        print(f"DEBUG: Audio path from DB: {audio_path}")
        
        # Modified approach to check for audio file existence
        # Try all possible paths where the file might exist
        standard_path = get_audio_file_path(injection_id)
        print(f"DEBUG: Standard path: {standard_path}")
        
        db_path_exists = False
        standard_path_exists = False
        
        if audio_path:
            db_path_exists = os.path.exists(audio_path)
            print(f"DEBUG: DB path exists: {db_path_exists}")
        
        standard_path_exists = os.path.exists(standard_path)
        print(f"DEBUG: Standard path exists: {standard_path_exists}")
        
        # Check for any possible audio files with this ID
        if os.path.exists(AUDIO_DIR):
            all_files = os.listdir(AUDIO_DIR)
            matching_files = [f for f in all_files if injection_id in f]
            print(f"DEBUG: All matching files in {AUDIO_DIR}: {matching_files}")
        
        # More permissive file existence check - file exists if found in ANY location
        file_exists = db_path_exists or standard_path_exists
        
        # If status is completed but file not found in expected locations, try to find it
        if is_completed and not file_exists and os.path.exists(AUDIO_DIR):
            all_files = os.listdir(AUDIO_DIR)
            matching_files = [f for f in all_files if injection_id in f]
            
            if matching_files:
                # Update the database with the found file path
                found_path = os.path.join(AUDIO_DIR, matching_files[0])
                print(f"DEBUG: Found matching file: {found_path}")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE injections SET processed_path = ? WHERE id = ?",
                    (found_path, injection_id)
                )
                conn.commit()
                conn.close()
                
                file_exists = True
                print(f"DEBUG: Updated database with found path: {found_path}")
        
        # Create animation style 
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
        
        # Auto-refresh script - only needed if not completed
        refresh_script = Script("""
        if (!document.getElementById('completed-indicator')) {
            setTimeout(function() {
                window.location.reload();
            }, 5000);
        }
        """) if not is_completed else None
        
        # IMPORTANT CHANGE: Use a direct link to audio-raw endpoint 
        # instead of trying to serve the file directly
        audio_player = None
        # Show player if either:
        # 1. Status is completed AND we found the file, OR
        # 2. Status is completed and we made a direct request to see if file exists
        if is_completed:
            print(f"DEBUG: Creating audio player, status is completed")
            # Try direct request to audio endpoint as final check
            try:
                audio_player = Div(
                    H2("Listen to Your Podcast", cls="text-lg font-bold mb-3 text-white text-center"),
                    Div(
                        # Use direct link to audio-raw which we know works
                        NotStr(f'<audio src="/audio-raw/{injection_id}" controls class="w-full rounded-lg shadow mb-4"></audio>'),
                        A("Download Podcast", 
                        href=f"/audio-raw/{injection_id}", 
                        download=f"podcast_{injection_id}.wav", 
                        cls="btn btn-secondary w-full"),
                        cls="bg-black bg-opacity-30 p-4 rounded-lg"
                    ),
                    cls="mb-6"
                )
            except Exception as e:
                print(f"DEBUG: Error creating audio player: {e}")
        
        # Status indicator
        status_indicator = Div(
            Div(
                # Success icon or loading indicator
                Div(
                    Span("✓", cls="text-4xl", id="completed-indicator") if is_completed else NotStr('<div class="loading loading-dots loading-lg"></div>'),
                    cls="text-success mb-6 mx-auto"
                ),
                
                # Status text
                P(f"Status: {status}", cls="text-lg mb-2 text-center text-white"),
                P(notes or "Processing in progress...", cls="text-sm mb-4 text-center text-white"),
                P(f"Podcast ID: {injection_id}", cls="font-mono text-sm mb-6 text-center text-white"),
                
                cls="p-6 bg-black bg-opacity-20 rounded-lg mb-6 text-center"
            )
        )
        
        # Expected time info
        time_info = None
        if not is_completed:
            time_info = Div(
                P("Podcast generation takes approximately 5-10 minutes.", 
                cls="text-center text-white mb-2"),
                P("This page will automatically refresh until your podcast is ready.", 
                cls="text-center text-white mb-4"),
                cls="mb-4 p-4 bg-black bg-opacity-30 rounded-lg"
            )
        
        # Debug section for completed podcasts where audio player isn't showing
        debug_info = None
        if is_completed and not audio_player:
            debug_info = Div(
                P("Audio file information:", cls="text-white text-center font-bold"),
                P(f"Status: {status}, File found: {file_exists}", cls="text-white text-center text-sm"),
                P(f"Try downloading directly: ", cls="text-white text-center text-sm"),
                A("Direct Download Link", href=f"/audio-raw/{injection_id}", cls="text-white underline"),
                cls="p-4 bg-black bg-opacity-30 rounded-lg mb-4"
            )
        
        return Title("Podcast Status"), Main(
            animation_style,
            refresh_script,
            Div(
                H1(
                    "Your Podcast is Ready!" if is_completed else "Your Podcast is Being Generated",
                    cls="text-3xl font-bold text-center text-white mb-6"
                ),
                status_indicator,
                time_info,
                debug_info,  # Add debug section if needed
                audio_player,
                A("← Back to Home", href="/", cls="btn btn-primary block mx-auto mt-6"),
                cls="container mx-auto px-4 py-8 max-w-3xl"
            ),
            cls="animated-bg min-h-screen"
        )

    # Also update the audio-raw endpoint to be more permissive
    @rt("/audio-raw/{injection_id}")
    def serve_audio_raw(injection_id: str):
        """Direct audio file handler with standardized path and volume reload"""
        print(f"📢 Audio request received for ID: {injection_id}")
        
        # First explicitly reload the volume to get latest changes
        shared_volume.reload()
        
        # Try standardized path first
        audio_path = get_audio_file_path(injection_id)
        
        # If standardized path doesn't exist, check database
        if not os.path.exists(audio_path):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT processed_path FROM injections WHERE id = ?", 
                (injection_id,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                audio_path = result[0]
        
        # If the file exists, serve it
        if os.path.exists(audio_path):
            print(f"✅ Serving audio file: {audio_path}")
            return FileResponse(
                audio_path, 
                media_type="audio/wav",
                filename=f"podcast_{injection_id}.wav"
            )
                
        # If not found, try to find any file with this ID in the audio directory
        if os.path.exists(AUDIO_DIR):
            all_files = os.listdir(AUDIO_DIR)
            matching_files = [f for f in all_files if injection_id in f]
            
            print(f"🔍 Looking for files matching {injection_id}")
            print(f"📂 All files in {AUDIO_DIR}: {len(all_files)} files")
            print(f"🔍 Files matching {injection_id}: {matching_files}")
            
            # If we find a matching file, use it
            if matching_files:
                alt_path = os.path.join(AUDIO_DIR, matching_files[0])
                print(f"🔄 Using alternate file path: {alt_path}")
                
                # Update the database with the correct path
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE injections SET processed_path = ? WHERE id = ?",
                    (alt_path, injection_id)
                )
                conn.commit()
                conn.close()
                
                return FileResponse(
                    alt_path, 
                    media_type="audio/wav",
                    filename=f"podcast_{injection_id}.wav"
                )
        
        # If no file found, return error
        return HTMLResponse(
            """
            <div style="padding: 20px; background-color: #fff3cd; color: #856404; border-radius: 5px; margin: 20px auto; max-width: 600px; text-align: center;">
                <h3>Audio file not yet available</h3>
                <p>The podcast may still be processing. Please check back in a few minutes.</p>
                <p>If the issue persists, you can try refreshing the main status page.</p>
            </div>
            """,
            status_code=404
        )
            
    @rt("/status-redirect")
    def status_redirect(injection_id: str):
        """Redirect to the podcast status page"""
        return RedirectResponse(f"/podcast-status/{injection_id}")

    return fasthtml_app


if __name__ == "__main__":
    with modal.app.run():
        serve()  # Starts the FastHTML server with the correct function references

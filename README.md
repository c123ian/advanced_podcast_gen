# Multi-File Podcast Generator

A system that automatically creates podcasts from any text content. Upload a PDF, text file, website, or audio, and the system generates a natural conversational podcast between two speakers using AI.


## 🎙️ Overview

This application takes content from various sources, transforms it into a conversational podcast script, adds natural speech patterns (umms, laughs, pauses), and generates high-quality audio using Bark TTS. The entire pipeline runs on Modal Labs for scalable cloud deployment.

![image](https://github.com/user-attachments/assets/ae1bbdc0-c337-4075-a741-1c774067df46)

![image](https://github.com/user-attachments/assets/2ff07f65-7f3c-4d94-913d-e62981aa271d)


### ✨ Key Features

- **Multiple Input Types**: Process PDFs, web pages, text files, or audio transcriptions
- **Two-Speaker Conversational Format**: Content is rewritten as an engaging dialogue
- **Natural Speech Patterns**: Adds disfluencies like "umm", "[laughs]", pauses, and emphasis
- **Consistent Voice Models**: Maintains consistent voices across the entire podcast
- **Progress Tracking**: Shows real-time generation progress
- **Web UI**: Upload content and monitor podcast creation progress
- **Audio Playback**: Listen to the podcast directly in your browser

## 🏗️ Architecture

The system consists of three main components that run as separate but coordinated services:

![NotebookLLama](https://github.com/user-attachments/assets/20b5ba98-4947-4c96-9a15-d1f87868bd97)
[source](https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/NotebookLlama/TTS_Notes.md) 

### 📁 Files and Their Roles

1. **`deploy.py`**: Main orchestration file that combines all components
2. **`common_image.py`**: Defines shared resources (Docker image, volume)
3. **`input_gen.py`**: Web UI, file processing, and content extraction
4. **`scripts_gen.py`**: Converts raw content to podcast scripts using language models
5. **`audio_gen.py`**: Converts scripts to audio using Bark TTS

## 🔄 How It Works

### Content Flow

1. User uploads content via web UI
2. Content is extracted and processed
3. Raw text is passed to script generator
4. Script generator creates conversational podcast dialogue
5. Audio generator converts dialogue to spoken audio
6. Final podcast is available for listening in browser

### Handling Long Content

The system has multiple safeguards to prevent content that would create excessively long podcasts:

1. **Initial Content Truncation**: Long inputs are truncated to 75,000 characters
2. **Summarization**: For very long content (>30,000 chars), a dedicated summarizer creates a condensed version
3. **Script Length Control**: Scripts are limited to 35-40 exchanges between speakers
4. **Separator Detection**: Any script sections after "---" markers are truncated
5. **Audio Generation Limits**: If a script would take >45 minutes to generate, it's automatically shortened

## 🚀 Installation

### Prerequisites

- Python 3.10+
- [Modal CLI](https://modal.com/docs/guide/cli-reference) installed and authenticated
- Modal account (free tier works for testing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-file-podcast.git
   cd multi-file-podcast
   ```

2. Install dependencies:
   ```bash
   pip install modal python-fasthtml PyPDF2 langchain-community openai-whisper 
   ```

3. Create Modal volume for persistent storage:
   ```bash
   modal volume create combined_volume
   ```

4. Deploy to Modal:
   ```bash
   modal deploy deploy.py
   ```

## 💻 Usage

### Web Interface

1. Access the app at the URL provided after deployment
2. Upload a PDF, text file, website URL, or audio file
3. Wait for processing (can take 15-40 minutes depending on content length)
4. Listen to your podcast directly in the browser

### Checking Status

1. Note your podcast ID after uploading content
2. Use the "Check Status" form to monitor progress
3. The status page will automatically refresh to show progress updates
4. When complete, the audio player appears automatically

## ⚙️ Technical Details

### Audio Generation

- Uses Bark TTS with specific voice presets for each speaker
- Supports disfluencies and emotional expressions
- Maintains consistent voice characteristics across segments
- Employs randomized seed consistency for reproducible voices
- Parallel processing of speaker lines for faster generation

### Content Optimization

- Preprocesses text to remove irrelevant formatting
- Normalizes quotes and punctuation for better speech output
- Handles script formats consistently to prevent audio hallucination
- Intelligent chunking of sentences for natural speech cadence
- Adds proper silence between speakers for natural conversation flow

## 🔧 Troubleshooting

### Common Issues

1. **Audio Player Not Appearing**: Wait for full processing completion or use the direct audio URL
2. **HTMX Polling Issues**: If status updates stop, manually refresh the page
3. **Script Generation Failures**: Check for extremely long content or unusual formatting
4. **Audio Quality Issues**: Script may contain quotes or formatting that affects Bark TTS output

### Advanced Debugging

Check Modal logs for detailed error messages:
```bash
modal app logs multi-file-podcast
```

## 🔍 Further Development

- Add more voice options
- Support additional input formats
- Implement background music
- Add more granular control over podcast style and format
- Improve summarization for very long content

## 📄 License

MIT

## 🙏 Acknowledgments

- [Modal Labs](https://modal.com) for serverless infrastructure
- [Bark TTS](https://github.com/suno-ai/bark) for high-quality text-to-speech
- [FastHTML](https://github.com/fastai/fasthtml) for the web UI framework
- [LangChain](https://github.com/langchain-ai/langchain) for document processing

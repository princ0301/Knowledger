import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import yt_dlp
import assemblyai as aai
from dotenv import load_dotenv

from src.doc.doc_processor import DocumentChunk

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeTranscriber:

    def __init__(self, assembly_api_key: str):
        self.assembly_api_key = assembly_api_key
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_transcriber"
        self.temp_dir.mkdir(exist_ok=True)

        aai.settings.api_key = assembly_api_key

        logging.info("YouTubeTranscriber initialized")

    def extract_video_id(self, url: str) -> Optional[str]:
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = None
        return video_id

    def download_audio(self, url: str) -> str:
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        expected_path = self.temp_dir / f"{video_id}.m4a"
        if expected_path.exists():
            logger.info(f"Audio already exists: {expected_path}")
            return str(expected_path)

        logger.info(f"Downloading audio from: {url}")

        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            'quiet': True,
            'no_warnings': True,
            # Add headers to mimic a real browser
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Keep-Alive': '300',
                'Connection': 'keep-alive',
            },
            # Use cookies if available
            'cookiefile': None,
            # Retry on errors
            'retries': 3,
            # Add some delay between requests
            'sleep_interval': 1,
            # Use IPv4 to avoid some blocking
            'force_ipv4': True,
        }

        # Try multiple approaches if the first one fails
        approaches = [
            # Standard approach with browser headers
            ydl_opts,
            # Fallback with different extractor
            {**ydl_opts, 'extractor_args': {'youtube': {'player_client': ['android', 'web']}}},
            # Another fallback with different format selection
            {**ydl_opts, 'format': 'bestaudio/best'},
        ]

        last_error = None
        for i, opts in enumerate(approaches):
            try:
                logger.info(f"Attempting download approach {i+1}/{len(approaches)}")
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                
                if expected_path.exists():
                    logger.info(f"Audio downloaded successfully: {expected_path}")
                    return str(expected_path)
                else:
                    # Check for other possible extensions
                    for ext in ['m4a', 'mp4', 'webm', 'mp3']:
                        alt_path = self.temp_dir / f"{video_id}.{ext}"
                        if alt_path.exists():
                            logger.info(f"Audio downloaded with different extension: {alt_path}")
                            return str(alt_path)
                    
                    raise FileNotFoundError(f"Expected audio file not found: {expected_path}")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Approach {i+1} failed: {str(e)}")
                continue
        
        # If all approaches failed, raise the last error
        raise Exception(f"All download approaches failed. Last error: {str(last_error)}")

    def transcribe_youtube_video(
        self,
        url: str,
        cleanup_audio: bool = True
    ) -> List[DocumentChunk]:
        try:
            audio_path = self.download_audio(url)

            config = aai.TranscriptionConfig(
                speaker_labels=True,
                punctuate=True
            )

            logger.info("Starting transcription with speaker diarization...")
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_path)

            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")

            chunks = []
            video_id = self.extract_video_id(url)
            for i, utterance in enumerate(transcript.utterances):
                chunk = DocumentChunk(
                    content=f"Speaker {utterance.speaker}: {utterance.text}",
                    source_file=f"YouTube Video {video_id}",
                    source_type="youtube",
                    page_number=None,
                    chunk_index=i,
                    start_char=utterance.start,
                    end_char=utterance.end,
                    metadata={
                        'speaker': utterance.speaker,
                        'start_time': utterance.start,
                        'end_time': utterance.end,
                        'confidence': getattr(utterance, 'confidence', None),
                        'video_url': url,
                        'video_id': video_id
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Transcription completed: {len(chunks)} utterances")

            if cleanup_audio and os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("Audio file cleaned up")

            return chunks
        
        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {str(e)}")
            raise

    def cleanup_temp_files(self):
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*.m4a"):
                    file.unlink()
                logger.info("Temporary files cleanup up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")

if __name__ == "__main__":
    import os

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("Please set ASSEMBLYAI_API_KEY environment variable")
        exit(1)

    transcriber = YouTubeTranscriber(api_key)

    try:
        test_url = "https://www.youtube.com/watch?v=V-eGWtA6_2w"
        chunks = transcriber.transcribe_youtube_video(test_url)

        print(f"Transcribed {len(chunks)} utternaces:")
        for chunk in chunks[:5]:
            print(f"  {chunk.content}")
        
    except Exception as e:
        print(f"Error: {e}")
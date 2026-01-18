import logging
import os
import soundfile as sf
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None

from src.podcast.script_generator import PodcastScript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    speaker: str
    text: str
    audio_data: Any
    duration: float
    file_path: str

class PodcastTTSGenerator:

    def __init__(self, lang_code: str = 'a', sample_rate: int = 24000, max_retries: int = 3):
        if KPipeline is None:
            raise ImportError("Kokoro TTS not available. Install with: uv add kokoro>=0.9.4 soundfile")
        
        self.sample_rate = sample_rate
        self.max_retries = max_retries
        
        # Initialize with retry logic
        self.pipeline = self._initialize_pipeline_with_retry(lang_code)

        self.speaker_voices = {
            "Speaker 1": "af_heart",  # Female voice
            "Speaker 2": "am_liam"    # Male voice
        }

        logger.info(f"Kokoro TTS initialized with lang_code='{lang_code}', sample_rate={sample_rate}")
    
    def _initialize_pipeline_with_retry(self, lang_code: str) -> KPipeline:
        """Initialize Kokoro pipeline with retry logic for model download"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Initializing Kokoro pipeline (attempt {attempt + 1}/{self.max_retries})")
                
                # Set environment variables for better download performance
                os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface')
                
                pipeline = KPipeline(lang_code=lang_code)
                logger.info("✓ Kokoro pipeline initialized successfully")
                return pipeline
                
            except Exception as e:
                logger.warning(f"✗ Pipeline initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("All pipeline initialization attempts failed")
                    raise e
    
    def check_model_availability(self) -> bool:
        """Check if Kokoro model is available locally"""
        try:
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M')
            model_file = os.path.join(cache_dir, 'snapshots')
            
            if os.path.exists(model_file):
                # Check if model files exist in any snapshot directory
                for snapshot_dir in os.listdir(model_file):
                    snapshot_path = os.path.join(model_file, snapshot_dir)
                    if os.path.isdir(snapshot_path):
                        model_path = os.path.join(snapshot_path, 'kokoro-v1_0.pth')
                        if os.path.exists(model_path):
                            logger.info(f"✓ Kokoro model found at: {model_path}")
                            return True
            
            logger.info("✗ Kokoro model not found locally, will need to download")
            return False
            
        except Exception as e:
            logger.warning(f"Could not check model availability: {e}")
            return False
    
    def generate_podcast_audio(
        self, 
        podcast_script: PodcastScript,
        output_dir: str = "outputs/podcast_audio",
        combine_audio: bool = True
    ) -> List[str]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating podcast audio for {podcast_script.total_lines} segments")
        logger.info(f"Output directory: {output_dir}")

        audio_segments = []
        output_files = []

        for i, line_dict in enumerate(podcast_script.script):
            speaker, dialogue = next(iter(line_dict.items()))

            logger.info(f"Processing segemnt {i+1}/{podcast_script.total_lines}: {speaker}")

            try:
                segment_audio = self._generate_single_segment(speaker, dialogue)
                segment_filename = f"segment_{i+1:03d}_{speaker.replace(' ', '_').lower()}.wav"
                segment_path = os.path.join(output_dir, segment_filename)

                sf.write(segment_path, segment_audio, self.sample_rate)
                output_files.append(segment_path)

                if combine_audio:
                    audio_segment = AudioSegment(
                        speaker=speaker,
                        text=dialogue,
                        audio_data=segment_audio,
                        duration=len(segment_audio) / self.sample_rate,
                        file_path=segment_path
                    )
                    audio_segments.append(audio_segment)
                
                logger.info(f"✓ Generated segment {i+1}: {segment_filename}")
                
            except Exception as e:
                logger.error(f"✗ Failed to generate segment {i+1}: {str(e)}")
                continue
        
        if combine_audio and audio_segments:
            combined_path = self._combine_audio_segments(audio_segments, output_dir)
            output_files.append(combined_path)
        
        logger.info(f"Podcast generation complete! Generated {len(output_files)} files")
        return output_files
    
    def _generate_single_segment(self, speaker: str, text: str) -> Any:
        voice = self.speaker_voices.get(speaker, "af_heart")
        clean_text = self._clean_text_for_tts(text)

        generator = self.pipeline(clean_text, voice=voice)
        
        combined_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            combined_audio.append(audio)
        
        if len(combined_audio) == 1:
            return combined_audio[0]
        else:
            import numpy as np
            return np.concatenate(combined_audio)

    def _clean_text_for_tts(self, text: str) -> str:
        clean_text = text.strip()

        clean_text = clean_text.replace("...", ".")
        clean_text = clean_text.replace("!!", "!")
        clean_text = clean_text.replace("??", "?")

        if not clean_text.endswith(('.', '!', '?')):
            clean_text += '.'
        
        return clean_text
    
    def _combine_audio_segments(
        self, 
        segments: List[AudioSegment], 
        output_dir: str
    ) -> str:
        logger.info(f"Combining {len(segments)} audio segments")
        
        try:
            import numpy as np
            
            pause_duration = 0.2  # seconds
            pause_samples = int(pause_duration * self.sample_rate)
            pause_audio = np.zeros(pause_samples, dtype=np.float32)
            
            combined_audio = []
            for i, segment in enumerate(segments):
                combined_audio.append(segment.audio_data)
                
                if i < len(segments) - 1:
                    combined_audio.append(pause_audio)
            
            final_audio = np.concatenate(combined_audio)
            
            combined_filename = "complete_podcast.wav"
            combined_path = os.path.join(output_dir, combined_filename)
            sf.write(combined_path, final_audio, self.sample_rate)
            
            duration = len(final_audio) / self.sample_rate
            logger.info(f"✓ Combined podcast saved: {combined_path} (Duration: {duration:.1f}s)")
            
            return combined_path
            
        except Exception as e:
            logger.error(f"✗ Failed to combine audio segments: {str(e)}")
            raise


if __name__ == "__main__":
    import json
    
    try:
        tts_generator = PodcastTTSGenerator()
        
        sample_script_data = {
            "script": [
                {"Speaker 1": "Hello everyone and welcome back to TechTalk Weekly! Today we’re diving deep into how artificial intelligence is reshaping our daily lives."},
                {"Speaker 2": "Thanks for inviting me! AI has quietly become part of almost everything we do—from smartphones to online shopping."},
                {"Speaker 1": "That’s true. Many people use AI every day without even realizing it. Can you give some simple examples?"},
                {"Speaker 2": "Of course! Recommendation systems on Netflix, spam filters in email, voice assistants like Alexa, and even Google Maps traffic predictions are all powered by AI."},
                {"Speaker 1": "Wow, that really puts things into perspective. Let’s talk about data. Why is data so important for AI systems?"},
                {"Speaker 2": "Data is the foundation of AI. Without high-quality data, even the most advanced algorithms can’t perform well. AI learns patterns directly from examples."},
                {"Speaker 1": "So does that mean more data always leads to better results?"},
                {"Speaker 2": "Not necessarily. More data helps, but clean and relevant data matters far more than just large volumes."},
                {"Speaker 1": "Interesting point. Now, people often worry that AI will replace jobs. What’s your take on that?"},
                {"Speaker 2": "It’s a valid concern, but history shows that technology usually transforms jobs rather than completely eliminating them."},
                {"Speaker 1": "So instead of replacing humans, AI works alongside them?"},
                {"Speaker 2": "Exactly. AI can handle repetitive tasks, allowing humans to focus on creativity, strategy, and decision-making."},
                {"Speaker 1": "Let’s shift toward the future. What advancements in AI excite you the most?"},
                {"Speaker 2": "I’m particularly excited about AI in healthcare—early disease detection, drug discovery, and personalized treatment plans."},
                {"Speaker 1": "That could truly save lives. Before we wrap up, what advice would you give to students who want to work in AI?"},
                {"Speaker 2": "Start with strong fundamentals in programming and mathematics, experiment with real projects, and stay curious because this field evolves rapidly."},
                {"Speaker 1": "Fantastic advice. Thank you so much for sharing your insights today."},
                {"Speaker 2": "Thank you! It was a pleasure being part of this conversation."},
                {"Speaker 1": "And thanks to our listeners for tuning in. We’ll see you in the next episode of TechTalk Weekly!"}
            ]
        }
        
        from src.podcast.script_generator import PodcastScript
        test_script = PodcastScript(
            script=sample_script_data["script"],
            source_document="AI Overview Test",
            total_lines=len(sample_script_data["script"]),
            estimated_duration="2 minutes"
        )
        
        print("Generating podcast audio...")
        output_files = tts_generator.generate_podcast_audio(
            test_script,
            output_dir="./podcast_output",
            combine_audio=True
        )
        
        print(f"\nGenerated files:")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        print("\nPodcast TTS test completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install Kokoro TTS: pip install kokoro>=0.9.4")
    except Exception as e:
        print(f"Error: {e}")
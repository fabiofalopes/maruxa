import asyncio
import edge_tts
from datetime import datetime
import os
from config.config import VOICE_OUTPUTS_DIR
import io

class EdgeTTS:
    SELECT_VOICE = ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"]

    def __init__(self):
        self.voice = self.SELECT_VOICE[0]
        self.rate = "+25%"    # Speed up by 25%
        self.volume = "+0%"   # Default volume
        self.pitch = "+10Hz"  # Default pitch

    async def generate_speech(self, text: str, output_file: str, voice: str = None):
        """
        Generate speech from text using edge-tts

        Args:
            text (str): Text to convert to speech
            output_file (str): Output audio file path
            voice (str, optional): Voice to use (defaults to instance voice)
        """
        communicate = edge_tts.Communicate(
            text,
            voice or self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )
        await communicate.save(output_file)

    async def create_audio(self, text: str) -> str:
        """
        Generate audio from text.

        Args:
            text (str): Text to convert to speech

        Returns:
            str: Path to the generated audio file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = f"tts_output_{timestamp}.mp3"
        audio_file_path = os.path.join(VOICE_OUTPUTS_DIR, audio_file)
        
        try:
            await self.generate_speech(
                text=text,
                output_file=audio_file_path
            )
            print(f"Audio file generated: {audio_file_path}")
            return audio_file_path
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

    async def stream_speech(self, text: str, chunk_callback: callable):
        """
        Stream audio chunks in real-time
        Args:
            text: Text to synthesize
            chunk_callback: Function to handle audio chunks (receives bytes)
        """
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                await chunk_callback(chunk["data"])

    async def create_streaming_audio(self, text: str) -> int:
        """
        Returns the sample rate for streaming audio
        (Needed for proper playback configuration)
        """
        return 24000  # EdgeTTS default sample rate

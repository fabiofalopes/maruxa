import edge_tts
from .tts_base import BaseTTS
from config.config import VOICE_OUTPUTS_DIR
import os
import uuid
from typing import Callable, Awaitable
import asyncio
import logging

logger = logging.getLogger(__name__)

class EdgeTTSWrapper(BaseTTS):
    """EdgeTTS implementation supporting both file and streaming output"""

    SELECT_VOICE_PT_PT = ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"]
    SELECT_VOICE_PT_BR = ["pt-BR-AntonioNeural", "pt-BR-IsabelaNeural"]

    def __init__(self, 
                voice: str = SELECT_VOICE_PT_PT[1],
                rate: str = "+25%",
                volume: str = "+0%",
                pitch: str = "+100Hz"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self._connection_retries = 3
        self._connection_retry_delay = 1

    async def generate_audio(self, text: str) -> str:
        """Generate and save audio file with improved connection handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting TTS generation (attempt {attempt + 1}/{max_retries})")
                output_path = os.path.join(VOICE_OUTPUTS_DIR, f"tts_{uuid.uuid4()}.mp3")
                
                communicate = edge_tts.Communicate(
                    text,
                    self.voice,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch
                )
                
                # Add connection retry loop
                for conn_attempt in range(self._connection_retries):
                    try:
                        async with asyncio.timeout(30):
                            await communicate.save(output_path)
                            
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            logger.info(f"Successfully generated audio file: {output_path}")
                            return output_path
                        else:
                            raise Exception("Generated audio file is empty or does not exist")
                            
                    except asyncio.CancelledError:
                        logger.error("Connection attempt cancelled, retrying...")
                        if conn_attempt < self._connection_retries - 1:
                            await asyncio.sleep(self._connection_retry_delay)
                            self._connection_retry_delay *= 2
                            continue
                        raise
                        
                    except Exception as e:
                        if conn_attempt < self._connection_retries - 1:
                            logger.error(f"Connection error (attempt {conn_attempt + 1}): {str(e)}")
                            await asyncio.sleep(self._connection_retry_delay)
                            self._connection_retry_delay *= 2
                            continue
                        raise
                        
            except asyncio.CancelledError:
                logger.error("TTS generation was cancelled")
                if 'output_path' in locals() and os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up partial file: {e}")
                raise
                
            except Exception as e:
                logger.error(f"TTS generation error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
                
        raise Exception("Failed to generate audio after all retries")

    async def stream_audio(self, text: str, chunk_handler: Callable[[bytes], Awaitable[None]]):
        """Stream audio chunks"""
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                await chunk_handler(chunk["data"])

    @property
    def default_sample_rate(self) -> int:
        return 24000
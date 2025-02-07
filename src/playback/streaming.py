import sounddevice as sd
import numpy as np
import ffmpeg
import subprocess
import logging
from threading import RLock
from typing import Optional

logger = logging.getLogger(__name__)

class AudioStreamer:
    """Handles real-time audio streaming using FFmpeg"""
    
    def __init__(self):
        self._lock = RLock()
        self.samplerate: Optional[int] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self.streaming_active = False

    def start_stream(self, samplerate: int) -> None:
        """Initialize audio streaming with specified sample rate"""
        with self._lock:
            self._cleanup()
            try:
                if not 8000 <= samplerate <= 48000:
                    raise ValueError(f"Invalid sample rate: {samplerate}")
                
                self._ffmpeg_process = (
                    ffmpeg
                    .input('pipe:', format='mp3', loglevel='error')
                    .output('pipe:', format='f32le', acodec='pcm_f32le', 
                           ac=1, ar=samplerate, hide_banner=None, nostats=None)
                    .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
                )
                
                self._output_stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback,
                    blocksize=1024,
                    latency='low'
                )
                self._output_stream.start()
                self.samplerate = samplerate
                self.streaming_active = True
                logger.info(f"Streaming started at {samplerate}Hz")
                
            except Exception as e:
                logger.error(f"Stream initialization failed: {str(e)}")
                self._cleanup()
                raise

    def add_audio_data(self, chunk: bytes) -> None:
        """Send MP3 data to FFmpeg decoder"""
        with self._lock:
            if self.streaming_active and self._ffmpeg_process:
                try:
                    self._ffmpeg_process.stdin.write(chunk)
                    self._ffmpeg_process.stdin.flush()
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg: {str(e)}")

    def stop_stream(self) -> None:
        """Stop streaming and clean up resources"""
        with self._lock:
            self.streaming_active = False
            try:
                if self._ffmpeg_process:
                    self._ffmpeg_process.stdin.close()
                    self._ffmpeg_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg termination timed out")
            finally:
                if self._output_stream:
                    self._output_stream.stop()
                    self._output_stream.close()
                self._cleanup()
                logger.info("Streaming resources released")

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                       time, status: int) -> None:
        """Sounddevice audio output callback"""
        try:
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            if self._ffmpeg_process:
                pcm_data = self._ffmpeg_process.stdout.read(frames * 4)
                if pcm_data:
                    audio_array = np.frombuffer(pcm_data, dtype=np.float32)
                    outdata[:] = audio_array.reshape(-1, 1)
                else:
                    outdata.fill(0)
                    if not self.streaming_active:
                        raise sd.CallbackStop()
        except Exception as e:
            logger.error(f"Audio callback error: {str(e)}")
            raise sd.CallbackAbort from e

    def _cleanup(self) -> None:
        """Internal resource cleanup"""
        self._ffmpeg_process = None
        self._output_stream = None
        self.samplerate = None
        self.streaming_active = False
import sounddevice as sd
import soundfile as sf
import threading
import time
import platform
from pynput import keyboard
from rich.console import Console
from rich.progress import Progress
from typing import Optional
import numpy as np
import queue
from pydub import AudioSegment
import io
import logging
import ffmpeg
import subprocess
from .streaming import AudioStreamer
from .file_playback import FileAudioController

logger = logging.getLogger(__name__)

class AudioController:
    """Main audio controller handling both streaming and file playback"""
    
    def __init__(self):
        self.console = Console()
        self.is_playing = False
        self.is_paused = False
        self.stream = None
        self.current_frame = 0
        self.total_frames = 0
        self.data = None
        self.samplerate = None
        self.paused = False
        self.should_stop = False
        self.listener = None
        self._lock = threading.RLock()
        self.streaming_buffer = queue.Queue()
        self.streaming_active = False
        self.current_stream = None
        self._ffmpeg_process = None
        self._ffmpeg_stdin = None
        self._ffmpeg_stdout = None
        self.streamer = AudioStreamer()
        self.file_player = FileAudioController()
        self.active_mode = None  # 'stream' or 'file'
        self._playback_complete = threading.Event()
        self._playback_thread = None

    def cleanup(self):
        """Clean up audio resources and reset state."""
        with self._lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    self.console.print(f"[yellow]Warning during stream cleanup: {str(e)}[/yellow]")
                finally:
                    self.stream = None

            # Stop and remove keyboard listener
            if self.listener:
                try:
                    self.listener.stop()
                except Exception:
                    pass
                self.listener = None

            self.is_playing = False
            self.is_paused = False
            self.current_frame = 0
            self.total_frames = 0
            self.data = None
            self.samplerate = None
            self.paused = False
            self.should_stop = False

    async def play_audio(self, file_path: str):
        """Async wrapper for audio playback"""
        with self._lock:
            self.stop_all()
            self._playback_complete.clear()
            self.active_mode = 'file'
            self._start_listener()
            self.is_playing = True
            self.should_stop = False
            
            # Play audio directly in the file player
            self.file_player.play_audio_file(file_path)
            self.samplerate = self.file_player.samplerate
            self.total_frames = self.file_player.total_frames
            self.current_frame = 0

    def _run_file_playback(self, file_path):
        """Blocking file playback runner"""
        try:
            self.file_player.play_audio_file(file_path)
            self.samplerate = self.file_player.samplerate
            self.total_frames = self.file_player.total_frames
            self.current_frame = 0
            self.is_playing = True
            self.file_player.wait_for_completion()
        finally:
            self.is_playing = False
            self._playback_complete.set()

    def stop_file_playback(self):
        if self.active_mode == 'file':
            self.file_player.stop()
            self.active_mode = None

    def stop_all(self):
        """Stop all playback immediately"""
        with self._lock:
            self.is_playing = False
            self.should_stop = True
            self.file_player.stop()
            self.streamer.stop_stream()
            self.active_mode = None
            
            if self.listener:
                self.listener.stop()
                self.listener = None
            
            if self._playback_thread and self._playback_thread.is_alive():
                self._playback_thread.join(timeout=0.5)

    def start_streaming_playback(self, samplerate: int):
        self.active_mode = 'stream'
        self.streamer.start_stream(samplerate)

    def add_audio_chunk(self, chunk: bytes):
        if self.active_mode == 'stream':
            self.streamer.add_audio_data(chunk)

    def stop_streaming(self):
        if self.active_mode == 'stream':
            self.streamer.stop_stream()
            self.active_mode = None

    def _skip_frames(self, frames):
        """Skip forward or backward in the audio"""
        if not self.samplerate or self.active_mode != 'file':
            return
            
        with self._lock:
            new_position = self.file_player.current_frame + frames
            if 0 <= new_position < self.file_player.total_frames:
                self.file_player.seek(new_position / self.samplerate)
                self.current_frame = new_position
                direction = "→" if frames > 0 else "←"
                self.console.print(f"[dim]{direction} {abs(frames/self.samplerate):.1f}s[/dim]")

    def _toggle_pause(self):
        """Toggle pause/resume"""
        with self._lock:
            if self.active_mode == 'file':
                is_paused = self.file_player.toggle_pause()
                self.is_paused = is_paused
                status = "⏸ Paused" if is_paused else "▶ Resumed"
                self.console.print(f"[dim]{status}[/dim]")

    def _stop(self):
        """Stop playback"""
        with self._lock:
            self.is_playing = False
            self.should_stop = True
            self.stop_all()
            self.console.print("[dim]⏹ Stopped[/dim]")

    def on_press(self, key):
        try:
            if not self.is_playing:
                return
            
            if hasattr(key, 'char'):
                with self._lock:
                    match key.char:
                        case 'p':
                            self._toggle_pause()
                        case 'a':
                            self._skip_frames(-int(self.samplerate))
                        case 'd':
                            self._skip_frames(int(self.samplerate))
                        case 's':
                            self._skip_frames(-int(self.samplerate * 10))
                        case 'w':
                            self._skip_frames(int(self.samplerate * 10))
                        case 'q':
                            self.should_stop = True
                            self._stop()
        except AttributeError:
            pass

    def __del__(self):
        """Safe destructor that handles cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass

    def _stream_callback(self, outdata, frames, time, status):
        """Audio output callback with error handling"""
        try:
            if status:
                logger.warning(f"Audio stream status: {status}")
            
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

    def wait_for_playback(self):
        """Non-blocking wait for playback completion"""
        while self.is_playing and not self.should_stop:
            if self.active_mode == 'file':
                if not self.file_player.is_playing:
                    self.is_playing = False
                    break
            time.sleep(0.05)  # Reduced sleep time

    def _start_listener(self):
        """Start keyboard listener if not already running"""
        if not self.listener or not self.listener.is_alive():
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()

    def update_progress(self):
        """Update current frame from file player"""
        if self.active_mode == 'file':
            self.current_frame = self.file_player.current_frame

# Maintain backward compatibility
audio_controller = AudioController()
play_audio = audio_controller.play_audio

# Export both the class and the function
__all__ = ['AudioController', 'audio_controller', 'play_audio']

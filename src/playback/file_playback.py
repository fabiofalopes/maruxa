from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import io
import logging
from threading import RLock
import soundfile as sf
import threading
import time

logger = logging.getLogger(__name__)

class FileAudioController:
    """Handles file-based audio playback"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._is_playing = False
        self._is_paused = False
        self._file_pos = 0
        self.total_frames = 0
        self.current_frame = 0
        self.samplerate = None
        self.current_stream = None
        self._current_samples = None
        
    @property
    def is_playing(self):
        return self._is_playing
        
    @property
    def position(self):
        return self._file_pos / self.samplerate if self.samplerate else 0
        
    def seek(self, seconds: float):
        """Seek to specific position"""
        with self._lock:
            if not self.samplerate:
                return
            new_pos = int(seconds * self.samplerate)
            self._file_pos = max(0, min(new_pos, self.total_frames))
            self.current_frame = self._file_pos

    def play_audio_file(self, file_path: str):
        """Play audio file"""
        with self._lock:
            self.stop()
            self._current_samples, self.samplerate = sf.read(file_path)
            self.total_frames = len(self._current_samples)
            self._file_pos = 0
            self._is_playing = True
            
            self.current_stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                callback=self._file_callback,
                finished_callback=self._on_playback_finished
            )
            self.current_stream.start()

    def play_audio(self, file_path: str):
        self.active_mode = 'file'
        self.samplerate = self.file_player.samplerate  # Add this line
        self.file_player.play_audio_file(file_path)

    def toggle_pause(self):
        """Toggle pause state"""
        with self._lock:
            self._is_paused = not self._is_paused
            if self.current_stream:
                if self._is_paused:
                    self.current_stream.stop()
                else:
                    self.current_stream.start()
                return self._is_paused

    def _file_callback(self, outdata, frames, time, status):
        """Callback for file playback"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        with self._lock:
            if self._is_paused:
                outdata.fill(0)
                return
                
            if self._current_samples is None:
                outdata.fill(0)
                raise sd.CallbackStop()
                
            remaining = len(self._current_samples) - self._file_pos
            if remaining == 0:
                self._is_playing = False
                raise sd.CallbackStop()
            
            available_frames = min(frames, remaining)
            outdata[:available_frames] = self._current_samples[
                self._file_pos:self._file_pos + available_frames
            ].reshape(-1, 1)
            self._file_pos += available_frames
            self.current_frame = self._file_pos
            
            if available_frames < frames:
                outdata[available_frames:] = 0
                self._is_playing = False
                raise sd.CallbackStop()
            
    def _on_playback_finished(self):
        """Callback when playback finishes"""
        with self._lock:
            self._is_playing = False
            self.current_frame = self.total_frames

    def stop(self):
        """Stop current playback"""
        with self._lock:
            if self.current_stream:
                self.current_stream.stop()
                self.current_stream.close()
                self.current_stream = None
            self._is_playing = False
            self._current_samples = None
            self._file_pos = 0

    def wait_for_completion(self):
        """Block until playback completes"""
        while self._is_playing:
            time.sleep(0.05)
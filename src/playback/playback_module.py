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

class AudioController:
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
        self._lock = threading.Lock()

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

    def play_audio(self, audio_file: str):
        """Start playing audio with controls"""
        # Clean up any existing playback
        self.cleanup()
        
        try:
            # Load the audio file
            self.data, self.samplerate = sf.read(audio_file)
            self.total_frames = len(self.data)
            self.current_frame = 0
            self.is_playing = True
            self.is_paused = False

            # Print controls before progress bar
            self.console.print("\nPlaying Audio Response")
            self.console.print("\nControls:")
            self.console.print("p: Pause/Resume | q: Stop")
            self.console.print("a/d: Skip ±1s | w/s: Skip ±10s\n")
            
            with Progress() as progress:
                task = progress.add_task("", total=100)
                
                # Start keyboard listener
                self.listener = keyboard.Listener(on_press=self.on_press)
                self.listener.start()
                
                try:
                    def callback(outdata, frames, time, status):
                        if status:
                            self.console.print(f"[red]Status: {status}[/red]")
                        
                        if self.is_paused:
                            outdata.fill(0)
                            return
                        
                        if self.current_frame + frames > len(self.data):
                            remaining = len(self.data) - self.current_frame
                            if remaining > 0:
                                data_chunk = self.data[self.current_frame:len(self.data)]
                                outdata[:remaining, 0] = data_chunk
                                outdata[remaining:] = 0
                            else:
                                outdata.fill(0)
                            self.is_playing = False
                            raise sd.CallbackStop()
                        else:
                            data_chunk = self.data[self.current_frame:self.current_frame + frames]
                            outdata[:, 0] = data_chunk
                            self.current_frame += frames
                            progress.update(task, completed=(self.current_frame / len(self.data)) * 100)

                    # Create and start the stream
                    self.stream = sd.OutputStream(
                        samplerate=self.samplerate,
                        channels=1,
                        callback=callback
                    )

                    with self.stream:
                        while self.is_playing and not self.should_stop:
                            sd.sleep(100)  # More efficient than time.sleep

                finally:
                    self.cleanup()

        except Exception as e:
            self.console.print(f"[red]Error playing audio: {str(e)}[/red]")
            self.cleanup()

    def _skip_frames(self, frames):
        """Skip forward or backward in the audio"""
        new_position = self.current_frame + frames
        if 0 <= new_position < self.total_frames:
            self.current_frame = new_position
            skip_seconds = frames / self.samplerate
            direction = "→" if frames > 0 else "←"
            self.console.print(f"[dim]{direction} {abs(skip_seconds)}s[/dim]")

    def _toggle_pause(self):
        """Toggle pause/resume"""
        self.is_paused = not self.is_paused
        status = "⏸ Paused" if self.is_paused else "▶ Resumed"
        self.console.print(f"[dim]{status}[/dim]")

    def _stop(self):
        """Stop playback"""
        self.is_playing = False
        self.console.print("[dim]⏹ Stopped[/dim]")

    def on_press(self, key):
        try:
            if hasattr(key, 'char'):  # Regular keys
                if key.char == 'p':
                    self._toggle_pause()
                elif key.char == 'q':
                    self._stop()
                elif key.char == 'a':
                    self._skip_frames(-int(self.samplerate))  # Back 1s
                elif key.char == 'd':
                    self._skip_frames(int(self.samplerate))   # Forward 1s
                elif key.char == 's':
                    self._skip_frames(-int(self.samplerate * 10))  # Back 10s
                elif key.char == 'w':
                    self._skip_frames(int(self.samplerate * 10))   # Forward 10s
        except AttributeError:
            pass

    def __del__(self):
        """Safe destructor that handles cleanup"""
        try:
            self.cleanup()
        except Exception:
            pass

# Create a singleton instance
audio_controller = AudioController()

# Function to maintain backward compatibility
def play_audio(audio_file: str):
    """Backward compatible function for playing audio"""
    audio_controller.play_audio(audio_file)

# Export both the class and the function
__all__ = ['AudioController', 'audio_controller', 'play_audio']

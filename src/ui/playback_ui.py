from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from rich.console import Console, Group
import time

class PlaybackDisplay:
    def __init__(self, controller):
        self.controller = controller
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed:.1f}s/"),
            TextColumn("[cyan]{task.total:.1f}s"),
            TimeRemainingColumn()
        )
        self.task = None
        self._last_update = 0
        self.update_interval = 0.05  # 50ms update interval
        
    def live_display(self):
        status = "⏸ Paused" if self.controller.is_paused else "▶ Playing"
        controls = "[dim]p: Play/Pause | a/d: ±1s | s/w: ±10s | q: Exit[/dim]"
        
        return Panel(
            Group(
                self.progress,
                Text(status, style="bold yellow"),
                Text(controls)
            ),
            title="[bold]Audio Playback[/bold]"
        )
        
    def update(self):
        current_time = time.time()
        if current_time - self._last_update < self.update_interval:
            return
            
        if not self.task:
            if not self.controller.samplerate:
                return
            total_seconds = self.controller.total_frames / self.controller.samplerate
            self.task = self.progress.add_task(
                "Playing", 
                total=total_seconds
            )
        
        if self.controller.samplerate:
            current_seconds = self.controller.current_frame / self.controller.samplerate
            self.progress.update(self.task, completed=current_seconds)
            self._last_update = current_time
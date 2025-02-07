import sys
import os
import asyncio
from rich.console import Console
from rich.prompt import Prompt
from tts.edge_tts_wrapper import EdgeTTSWrapper
from playback.playback_module import play_audio
from config.config import VOICE_OUTPUTS_DIR

# Add the parent directory of src to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()

def main():
    console.print("[bold blue]Welcome to the Speech Application![/bold blue]")
    
    while True:
        console.print("\nChoose an option:")
        console.print("1. Generate Speech")
        console.print("2. Exit")
        choice = Prompt.ask("Enter your choice", choices=["1", "2"])
        
        if choice == "1":
            text = Prompt.ask("Enter the text you want to convert to speech")
            audio_file = EdgeTTSWrapper.generate_audio(text)
            if audio_file:
                console.print(f"Playing audio: [green]{audio_file}[/green]")
                play_audio(audio_file)
        elif choice == "2":
            console.print("Exiting the application. Goodbye!")
            break

if __name__ == "__main__":
    main()
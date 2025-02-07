from typing import Optional
from rich.console import Console
from rich.progress import Progress
from rich.live import Live
from llama_index.core.llms import ChatMessage
from stt.groq_whisper import GroqWhisperAPI
from tts.edge_tts_wrapper import EdgeTTSWrapper
from llm.groq_llm import GroqLLMWrapper
from utils.index_manager import IndexManager
from audio_processing.recorder import AudioRecorder
from playback.playback_module import audio_controller
from ui.playback_ui import PlaybackDisplay
from config.config import RECORDINGS_DIR
import os
import asyncio
from pynput import keyboard

class VoiceAssistantWorkflow:
    def __init__(self, index_manager: IndexManager):
        self.console = Console()
        self.recorder = AudioRecorder(output_directory=RECORDINGS_DIR)
        self.stt = GroqWhisperAPI()
        self.llm_wrapper = GroqLLMWrapper()
        self.index_manager = index_manager
        self.audio_controller = audio_controller
        self.tts = EdgeTTSWrapper()
        
        # Load system prompts
        self.system_prompt = self._load_prompt("system_prompt.md")
        self.speech_prompt = self._load_prompt("speech_prompt.md")

    async def process_voice_input(self):
        try:
            # Recording and transcription
            self.console.print("[bold green]Recording...[/bold green] (Press Ctrl+C to stop)")
            audio_path = self.recorder.record_until_q("input.wav")
            
            text = self.stt.transcribe_audio(audio_path)
            self.console.print(f"\n[blue]Transcribed:[/blue] {text}")
            
            # Query and LLM processing
            with Progress() as progress:
                task = progress.add_task("[yellow]Processing query...", total=None)
                try:
                    # Get LLM instance once
                    llm = self.llm_wrapper.get_llm()
                    
                    # First get relevant quotes with retry logic
                    max_retries = 3
                    retry_delay = 1
                    
                    for attempt in range(max_retries):
                        try:
                            quotes = self.index_manager.get_document_quotes(text, llm)
                            if quotes:
                                self.console.print("\n[cyan]Retrieved context:[/cyan]")
                                for i, quote in enumerate(quotes, 1):
                                    self.console.print(f"\n[dim]{i}. From {quote['file']} (relevance: {quote['score']:.2f}):[/dim]")
                                    self.console.print(f"[italic]{quote['text']}[/italic]")
                            break
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                self.console.print(f"\n[yellow]LLM service temporarily unavailable. Retrying in {retry_delay} seconds...[/yellow]")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            raise
                    
                    # Get response using query engine with retry logic
                    query_engine = self.index_manager.get_query_engine(llm)
                    if query_engine is None:
                        raise ValueError("No index available")
                    
                    for attempt in range(max_retries):
                        try:
                            rag_response = query_engine.query(text)
                            break
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                self.console.print(f"\n[yellow]LLM service temporarily unavailable. Retrying in {retry_delay} seconds...[/yellow]")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            raise
                    
                    progress.update(task, completed=True)
                    
                    # Process response
                    response_text = str(rag_response)
                    self.console.print(f"\n[green]Response:[/green] {response_text}")
                    
                    # Convert to speech
                    audio_file = await self.tts.generate_audio(response_text)

                    # Stop progress before audio playback
                    progress.stop()

                    if audio_file:
                        try:
                            await self.audio_controller.play_audio(audio_file)
                            display = PlaybackDisplay(self.audio_controller)
                            
                            with Live(display.live_display(), refresh_per_second=20) as live:
                                while self.audio_controller.is_playing and not self.audio_controller.should_stop:
                                    display.update()
                                    live.update(display.live_display())
                                    await asyncio.sleep(0.05)
                                
                                # Only show replay option if playback wasn't stopped with 'q'
                                if not self.audio_controller.should_stop:
                                    self.console.print("\n[dim]Press 'r' to replay, or Enter to continue[/dim]")
                                    response = await asyncio.get_event_loop().run_in_executor(None, input)
                                    
                                    if response.lower() == 'r':
                                        await self.audio_controller.play_audio(audio_file)
                        except Exception as e:
                            self.console.print(f"[red]Error playing audio: {str(e)}[/red]")
                        finally:
                            self.audio_controller.stop_all()
                    
                except Exception as e:
                    progress.update(task, completed=True)
                    if "503" in str(e):
                        self.console.print("[red]Error: The LLM service is currently unavailable. Please try again in a few minutes.[/red]")
                    else:
                        self.console.print(f"[red]Error processing query: {str(e)}[/red]")
                        raise ValueError("Não foi possível processar a consulta. Verifique se existem documentos na pasta 'notes'.")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Recording cancelled[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error in workflow: {str(e)}[/red]")
            
        self.console.print("\n[dim]Press Enter for new interaction, or 'q' to quit[/dim]")

    def _load_prompt(self, filename: str) -> str:
        prompt_path = os.path.join("src", "prompts", filename)
        try:
            # Explicitly specify UTF-8 encoding
            with open(prompt_path, "r", encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.console.print(f"[yellow]Warning: {filename} not found[/yellow]")
            return ""

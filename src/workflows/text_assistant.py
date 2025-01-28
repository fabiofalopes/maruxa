from typing import Optional
from rich.console import Console
from rich.progress import Progress
from llama_index.core.llms import ChatMessage
from tts.edge_tts_integration import EdgeTTS
from llm.groq_llm import GroqLLMWrapper
from utils.index_manager import IndexManager
from playback.playback_module import audio_controller
import os

class TextAssistantWorkflow:
    def __init__(self, index_manager: IndexManager):
        self.console = Console()
        self.tts = EdgeTTS()
        self.llm_wrapper = GroqLLMWrapper()
        self.index_manager = index_manager
        self.audio_controller = audio_controller
        
        # Load system prompts
        self.system_prompt = self._load_prompt("system_prompt.md")
        self.speech_prompt = self._load_prompt("speech_prompt.md")

    async def process_text_input(self, text: str):
        try:
            # Query and LLM processing
            with Progress() as progress:
                task = progress.add_task("[yellow]Processing query...", total=None)
                try:
                    # Get LLM instance once
                    llm = self.llm_wrapper.get_llm()
                    
                    # First get relevant quotes
                    quotes = self.index_manager.get_document_quotes(text, llm)
                    if quotes:
                        self.console.print("\n[cyan]Retrieved context:[/cyan]")
                        for i, quote in enumerate(quotes, 1):
                            self.console.print(f"\n[dim]{i}. From {quote['file']} (relevance: {quote['score']:.2f}):[/dim]")
                            self.console.print(f"[italic]{quote['text']}[/italic]")
                    
                    # Get response using query engine
                    query_engine = self.index_manager.get_query_engine(llm)
                    if query_engine is None:
                        raise ValueError("No index available")
                    rag_response = query_engine.query(text)
                    progress.update(task, completed=True)
                    
                    # Process response
                    response_text = str(rag_response)
                    self.console.print(f"\n[green]Response:[/green] {response_text}")
                    
                    # Convert to speech
                    progress.stop()
                    audio_file = await self.tts.create_audio(response_text)
                    if audio_file:
                        try:
                            self.audio_controller.play_audio(audio_file)
                        except Exception as e:
                            self.console.print(f"[red]Error playing audio: {str(e)}[/red]")
                except Exception as e:
                    progress.update(task, completed=True)
                    self.console.print(f"[red]Error processing query: {str(e)}[/red]")
                    raise ValueError("Não foi possível processar a consulta. Verifique se existem documentos na pasta 'notes'.")
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Processing cancelled[/yellow]")
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
from rich.console import Console
from workflows.voice_assistant import VoiceAssistantWorkflow
from workflows.text_assistant import TextAssistantWorkflow
from utils.index_manager import IndexManager
import asyncio

async def main():
    console = Console()
    console.clear()
    
    # Initialize components
    index_manager = IndexManager()
    
    # Initialize workflows with proper async support
    voice_assistant = VoiceAssistantWorkflow(index_manager)
    text_assistant = TextAssistantWorkflow(index_manager)
    
    console.print("[bold blue]Assistant Interface[/bold blue]")
    
    while True:
        try:
            console.print("\n[bold cyan]Choose mode:[/bold cyan]")
            console.print("  [dim]v - Voice input[/dim]")
            console.print("  [dim]t - Text input[/dim]")
            console.print("  [dim]q - Quit[/dim]")
            
            mode = input("\nMode > ").lower()
            
            if mode == 'q':
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            if mode == 'v':
                await voice_assistant.process_voice_input()
            elif mode == 't':
                text = input("\nEnter your query: ")
                await text_assistant.process_text_input(text)
            else:
                console.print("[red]Invalid mode. Please choose 'v' or 't'.[/red]")
                continue
                
            # After interaction is complete
            console.print("\n[dim]─" * 50)
            console.print("[bold green]Interaction complete![/bold green]")
            console.print("[dim]Press Enter to continue, or 'q' to quit[/dim]")
            
            if input().lower() == 'q':
                console.print("[yellow]Goodbye![/yellow]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            continue

if __name__ == "__main__":
    asyncio.run(main())

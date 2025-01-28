from rich.console import Console
from workflows.voice_assistant import VoiceAssistantWorkflow
from workflows.text_assistant import TextAssistantWorkflow
from utils.index_manager import IndexManager

async def main():
    console = Console()
    console.clear()
    
    # Initialize components
    index_manager = IndexManager()
    voice_assistant = VoiceAssistantWorkflow(index_manager)
    text_assistant = TextAssistantWorkflow(index_manager)
    
    console.print("[bold blue]Assistant Interface[/bold blue]")
    console.print("[dim]Choose mode: 'v' for voice, 't' for text, or 'q' to quit[/dim]")
    
    while True:
        try:
            mode = input("Mode (v/t): ").lower()
            if mode == 'q':
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            if mode == 'v':
                # Start voice interaction
                await voice_assistant.process_voice_input()
            elif mode == 't':
                # Start text interaction
                text = input("Enter your query: ")
                await text_assistant.process_text_input(text)
            else:
                console.print("[red]Invalid mode. Please choose 'v' or 't'.[/red]")
                
            console.print("\n[dim]Press Enter for new interaction, or 'q' to quit[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interaction cancelled[/yellow]")
            console.print("[dim]Press Enter for new interaction, or 'q' to quit[/dim]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            console.print("[dim]Press Enter to try again, or 'q' to quit[/dim]")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

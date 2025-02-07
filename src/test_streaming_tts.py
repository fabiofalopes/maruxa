import asyncio
from playback.playback_module import audio_controller
from tts.edge_tts_wrapper import EdgeTTSWrapper
import logging
import numpy as np
import time
from rich.live import Live
from ui.playback_ui import PlaybackDisplay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Initializing TTS...")
        tts = EdgeTTSWrapper()
        
        logger.info("Getting sample rate...")
        sample_rate = tts.default_sample_rate
        logger.info(f"Sample rate: {sample_rate}")
        
        logger.info("Starting playback system...")
        audio_controller.start_streaming_playback(sample_rate)
        
        # Add small delay to let audio system initialize
        await asyncio.sleep(0.5)
        
        async def handle_chunk(chunk):
            with open("debug_audio.mp3", "ab") as f:
                f.write(chunk)
            logger.info(f"Received chunk: {len(chunk)} bytes")
            start_time = time.time()
            audio_controller.add_audio_chunk(chunk)
            logger.debug(f"Chunk processing took: {time.time()-start_time:.4f}s")
        
        # Test with very short text first
        text = "Test. One. Two. Three."  # Should take ~2 seconds
        
        # Test with 50 words text em português
        text = "Este é um teste de fala rápida. Ele deve ser processado rapidamente pelo sistema de reprodução de áudio."

        logger.info("Starting streaming...")
        stream_task = asyncio.create_task(tts.stream_audio(text, handle_chunk))
        
        # Monitor progress
        start_time = time.time()
        while not stream_task.done():
            logger.info("Streaming in progress...")
            await asyncio.sleep(0.5)
            if time.time() - start_time > 5:
                logger.error("Timeout!")
                break
        
        await stream_task
        
        # Add final silence to flush buffer
        silence = np.zeros(1024, dtype=np.float32)
        audio_controller.streaming_buffer.put(silence)
        await asyncio.sleep(0.5)
        
        # Add UI display
        display = PlaybackDisplay(audio_controller)
        
        with Live(display.live_display(), refresh_per_second=4):
            while audio_controller.streaming_active:
                display.update()
                await asyncio.sleep(0.25)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        logger.info("Cleaning up...")
        audio_controller.stop_streaming()
        if hasattr(audio_controller, '_mp3_buffer'):
            audio_controller._mp3_buffer.clear()

async def run_with_timeout():
    try:
        await asyncio.wait_for(main(), timeout=60)  # 1 minute total timeout
    except asyncio.TimeoutError:
        logger.error("Program timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_with_timeout())
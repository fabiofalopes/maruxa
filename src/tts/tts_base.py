from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable

class BaseTTS(ABC):
    """Abstract base class for TTS implementations"""
    
    @abstractmethod
    async def generate_audio(self, text: str) -> str:
        """Generate audio file and return path"""
        pass
    
    @abstractmethod
    async def stream_audio(self, text: str, chunk_handler: Callable[[bytes], Awaitable[None]]):
        """Stream audio chunks"""
        pass
    
    @property
    @abstractmethod
    def default_sample_rate(self) -> int:
        """Get default sample rate"""
        pass
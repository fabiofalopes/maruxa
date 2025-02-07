# Audio Recording and Transcription Tool

A Python-based tool for recording audio and performing transcription.

## System Requirements

### Windows Setup

1. Install Chocolatey package manager
   - Visit https://chocolatey.org/install#individual
   - Follow the installation instructions for individual use

2. Install FFmpeg using Chocolatey
   ```powershell
   choco install ffmpeg
   ```

3. Verify FFmpeg installation
   ```powershell
   ffmpeg -version
   ```

4. Audio Device Configuration
   - Ensure your microphone is properly connected and set as default input device
   - Check Windows Sound Settings to confirm microphone permissions are enabled
   - The application will list available audio input devices on startup

### Linux Setup

1. Install required audio and FFmpeg packages:

   For Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pyaudio portaudio19-dev
   sudo apt-get install ffmpeg
   sudo apt-get install alsa-utils
   ```

   For Fedora:
   ```bash
   sudo dnf install python3-pyaudio portaudio-devel
   sudo dnf install ffmpeg
   sudo dnf install alsa-utils
   ```

2. Configure ALSA:
   ```bash
   sudo alsactl init
   ```

3. Add your user to the audio group:
   ```bash
   sudo usermod -a -G audio $USER
   ```
   Note: You'll need to log out and back in for this change to take effect.

4. Verify audio device detection:
   ```bash
   arecord -l
   ```

## Common Issues

### Windows
- If FFmpeg is not found: Restart your terminal/IDE after installing FFmpeg
- Audio device not found: Check Windows Sound Settings and ensure microphone permissions are enabled

### Linux
- ALSA device errors: Run `alsamixer` to check if your input device is muted
- Permission denied: Ensure you're part of the `audio` group
- Device busy: Check if another application is using the audio device

## Virtual Environment Setup

It's recommended to run this application in a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Platform-Specific Setup

### Windows
1. Install FFmpeg:
   - Option 1: Download from [FFmpeg official site](https://ffmpeg.org/download.html)
   - Option 2: Using Chocolatey (if installed):
     ```
     choco install ffmpeg
     ```
   - Add FFmpeg to your system PATH

2. Install Python dependencies:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Linux (Debian/Ubuntu)
1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg python3-venv python3-dev portaudio19-dev
   ```

2. Install Python dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Add your user to the audio group:
   ```bash
   sudo usermod -a -G audio $USER
   ```
   Note: Log out and back in for this to take effect.

## Usage

1. Run the application:
```bash
python src/main.py
```

2. The application will:
   - List available audio input devices
   - Start recording (press 'q' to stop)
   - Process the audio
   - Perform transcription

## Notes

- The application requires proper audio input device configuration on your system
- FFmpeg must be available in your system PATH
- Audio files are temporarily stored in the 'recordings' directory

## Real-time Streaming Notes

- Requires FFmpeg 5.0+ with MP3 codec support
- Minimum system requirements:
  - 2GB free memory
  - Audio device supporting 24kHz/32-bit float output
- Streaming latency should be <200ms when properly configured

For development:
- Use `debug_audio.mp3` to verify TTS output
- Monitor CPU usage during streaming
- Test with various text lengths (1-500 characters)

---


# Notes

## Codebase Cleanup

To maintain a clean and efficient codebase, we've removed and reviewed several unused or redundant files. Below are the details of the cleanup process:

### Removed Files

- **`src/llm/rag_engine.py`**
  - **Reason:** This file was not referenced anywhere in the project and has been safely deleted.

### Review files

- **Test Scripts**
  - **`src/test_stt.py`**
  - **`src/test_tts.py`**
  - **Reason:** These were previously used for testing speech-to-text and text-to-speech functionalities. If you plan to perform future tests, consider keeping them; otherwise, they've been removed to streamline the codebase.

### Reviewed for Potential Removal

- **`src/config/setup.py`**
  - **Action Needed:** Verify if this setup script is invoked elsewhere (e.g., during installation or initialization). If it's no longer required, consider removing it or integrating its functionality into the main setup process.

- **`src/platform_setup.py`**
  - **Action Needed:** This script checks for FFmpeg dependencies but isn't currently called from `main.py`. Decide whether to integrate its checks into the main application flow or remove it if redundant.

### Dependency Files

- **`requirements-extra.txt`**
- **`requirements-cuda.txt`**
  - **Action Needed:** Ensure that the dependencies listed are still relevant to the project's current state. Remove any packages that are no longer needed to reduce bloat and potential security vulnerabilities.

### Recommendations

- **Comprehensive Code Audit:**
  - Use tools like [Vulture](https://github.com/jendrikseipp/vulture) to scan the codebase for any additional dead code or unused imports.
  
- **Documentation Update:**
  - Reflect the current project structure and dependencies in the `README.md` and other relevant documentation files to provide accurate guidance for future contributors and users.

- **Version Control Best Practices:**
  - Always commit changes before making deletions. This ensures that you can restore any files if needed in the future.

By adhering to these cleanup steps, we aim to enhance the maintainability and performance of the project.

---


# Copy code tree 

```bash

 tree -a -I ".*" --dirsfirst -L 4 -I notes/ -I venv/ -I __pycache__  | to_clip

```


# J.A.R.V.I.S - Just A Rather Very Intelligent System

An advanced desktop AI assistant inspired by Iron Man's JARVIS, featuring speech recognition, natural language processing, and a modern UI.

## Overview

This project implements a sophisticated personal assistant with voice interaction capabilities, system monitoring, and AI-powered conversations. The assistant is designed with an Iron Man-inspired interface and personality, making it both functional and engaging.

## Key Features

- **Voice Interaction**: Wake word detection, speech recognition, and text-to-speech capabilities
- **Modern UI**: Iron Man-inspired interface with real-time system monitoring visualizations
- **AI Conversations**: Integration with Claude AI for intelligent, contextual responses
- **System Monitoring**: Real-time tracking of CPU, memory, and network status
- **Task Management**: Scheduling, reminders, and note-taking capabilities
- **Multi-language Support**: Primary English support with extensibility for other languages

## Technical Components

- **AdvancedUI**: Modern, Iron Man-inspired UI with animations and visualizations
- **PersonalityEngine**: Manages JARVIS's distinct personality and response formatting
- **WakeWordDetector**: Voice activation with "Hey JARVIS" detection
- **SystemMonitor**: Real-time system resource monitoring
- **VoiceEnhancer**: Natural voice interaction with contextual responses
- **ConversationMemory**: Short-term memory to maintain context in conversations
- **ScheduleManager**: Manages calendar events and reminders

## Requirements

- Python 3.8+
- Required Python packages:
  - tkinter
  - speech_recognition
  - gtts
  - pygame
  - anthropic (Claude AI API)
  - pyaudio
  - webrtcvad
  - numpy
  - PIL
  - sounddevice
  - soundfile
  - psutil

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/jarvis.git
   cd jarvis
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Configure your Claude API key in `config.json`

4. Run the application:
   ```
   python jarvis.py
   ```

## Usage

1. Launch the application
2. Say "Hey JARVIS" to activate voice recognition
3. Ask questions or give commands like:
   - "What's the weather today?"
   - "Schedule a meeting for tomorrow at 2 PM"
   - "Take a note about the project deadline"
   - "Show system status"

## Project Structure

- `jarvis.py`: Main application entry point
- `config.json`: Configuration settings
- `logs/`: Application logs (debug, error, general)
- `notes/`: User notes storage
- `temp/`: Temporary files storage

## Current Status

The project is under active development. Current issues being addressed:
- Wake word detection sensitivity adjustment
- Claude API integration reliability
- Initialization sequence optimization

## Future Enhancements

- Web browsing capabilities
- Integration with smart home devices
- Advanced data visualization
- Customizable UI themes
- Extended task automation

## Acknowledgments

- Inspired by the JARVIS AI from the Iron Man movies
- Uses Claude AI by Anthropic for natural language understanding
- Special thanks to the open-source libraries that made this project possible

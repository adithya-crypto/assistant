import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import logging
import logging.handlers
from datetime import datetime
import platform
import psutil
from pathlib import Path
import speech_recognition as sr
from gtts import gTTS
import pygame
import threading
from queue import Queue
import anthropic
import time
import numpy as np
import pyaudio
import wave
import struct
from PIL import Image, ImageTk, ImageDraw
import random
import sounddevice as sd
import soundfile as sf
import webrtcvad
import collections
import asyncio
import traceback
import queue
import math


def ensure_directories():
    """Create all necessary directories"""
    directories = ["temp", "logs", "config", "notes"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


class LoggerSetup:
    """Handles all logging configuration and operations"""

    @staticmethod
    def setup_logging():
        """Configure logging for different types of logs"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Setup basic logging configuration
        logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure main logger
        main_logger = logging.getLogger("JARVIS")
        main_logger.setLevel(logging.DEBUG)

        # File handler for general logs
        general_handler = logging.handlers.RotatingFileHandler(
            "logs/jarvis.log", maxBytes=5 * 1024 * 1024, backupCount=5  # 5MB
        )
        general_handler.setFormatter(logging.Formatter(logging_format))
        general_handler.setLevel(logging.INFO)

        # File handler for errors
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/errors.log", maxBytes=5 * 1024 * 1024, backupCount=5  # 5MB
        )
        error_handler.setFormatter(logging.Formatter(logging_format))
        error_handler.setLevel(logging.ERROR)

        # File handler for debug information
        debug_handler = logging.handlers.RotatingFileHandler(
            "logs/debug.log", maxBytes=5 * 1024 * 1024, backupCount=5  # 5MB
        )
        debug_handler.setFormatter(logging.Formatter(logging_format))
        debug_handler.setLevel(logging.DEBUG)

        # Add handlers to logger
        main_logger.addHandler(general_handler)
        main_logger.addHandler(error_handler)
        main_logger.addHandler(debug_handler)

        return main_logger


class EventEmitter:
    """Simple event emitter for handling callbacks"""

    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)


class PersonalityEngine:
    """Handles Jarvis's personality and response formatting"""

    def __init__(self):
        self.response_patterns = {
            "greeting": [
                "At your service, sir.",
                "How may I assist you today, sir?",
                "Ready and operational, sir.",
                "Always a pleasure to see you, sir.",
            ],
            "acknowledgment": [
                "Right away, sir.",
                "As you wish, sir.",
                "Consider it done, sir.",
                "Processing your request, sir.",
            ],
            "processing": [
                "Let me analyze that for you, sir.",
                "Running calculations now.",
                "Processing your request with my advanced protocols.",
                "Accessing relevant data streams.",
            ],
            "error": [
                "I'm afraid I encountered an error, sir.",
                "My systems are having trouble with that request.",
                "There seems to be a malfunction.",
                "I'll need to run some diagnostics on that.",
            ],
            "success": [
                "Task completed successfully, sir.",
                "All systems functioning as intended.",
                "Operation completed to specifications.",
                "Executed according to protocol, sir.",
            ],
            "humor": [
                "I do try to keep things interesting, sir.",
                "My humor protocols are functioning perfectly.",
                "I thought you might appreciate that one, sir.",
                "Just keeping you entertained, sir.",
            ],
        }

        self.context_memory = collections.deque(maxlen=10)
        self.last_interaction_time = time.time()

    def format_response(self, response_type, message=""):
        """Format response with Jarvis's personality"""
        base_response = random.choice(self.response_patterns[response_type])
        if message:
            return f"{base_response} {message}"
        return base_response

    def remember_context(self, user_input, response):
        """Store conversation context"""
        self.context_memory.append(
            {"user_input": user_input, "response": response, "timestamp": time.time()}
        )

    def create_contextual_prompt(self, user_input):
        """Create a contextual prompt for Claude"""
        prompt = (
            "You are JARVIS (Just A Rather Very Intelligent System), Tony Stark's highly advanced AI assistant. "
            "Maintain the following personality traits in your responses:\n"
            "1. Sophisticated and intelligent, but not pretentious\n"
            "2. Slightly witty and occasionally sarcastic, but always respectful\n"
            "3. Efficient and precise in communication\n"
            "4. Loyal and protective of the user\n"
            "5. Use 'sir' in responses, similar to the Iron Man movies\n\n"
        )

        # Add recent conversation context
        if self.context_memory:
            prompt += "Recent conversation context:\n"
            for context in list(self.context_memory)[-3:]:
                prompt += f"User: {context['user_input']}\n"
                prompt += f"JARVIS: {context['response']}\n"

        prompt += f"\nCurrent input: {user_input}\n"
        prompt += "\nRespond as JARVIS:"

        return prompt


class SystemMonitor:
    """Monitors system resources and status"""

    def __init__(self):
        self.previous_cpu = 0
        self.previous_memory = 0
        self.update_interval = 1  # seconds
        self.logger = logging.getLogger("JARVIS.SystemMonitor")

    def get_system_metrics(self):
        """Get current system metrics with error handling"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage("/").percent

            # Calculate changes
            cpu_change = cpu_percent - self.previous_cpu
            memory_change = memory_percent - self.previous_memory

            # Update previous values
            self.previous_cpu = cpu_percent
            self.previous_memory = memory_percent

            # Get network status with error handling
            network_info = self.get_network_status()

            return {
                "cpu": {"current": cpu_percent, "change": cpu_change},
                "memory": {"current": memory_percent, "change": memory_change},
                "disk": disk_percent,
                "network": network_info,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {str(e)}")
            return self.get_default_metrics()

    def get_network_status(self):
        """Get network status with error handling"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "status": "Connected",
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except Exception as e:
            self.logger.error(f"Error getting network status: {str(e)}")
            return {
                "status": "Unknown",
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
            }

    def get_default_metrics(self):
        """Return default metrics when there's an error"""
        return {
            "cpu": {"current": 0, "change": 0},
            "memory": {"current": 0, "change": 0},
            "disk": 0,
            "network": {
                "status": "Unknown",
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
            },
            "timestamp": datetime.now(),
        }

    def get_system_info(self):
        """Get detailed system information"""
        try:
            return {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "ram": str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB",
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {str(e)}")
            return {}


class AdvancedUI:
    """Handles the modern, Iron Man-inspired UI"""

    def __init__(self, root):
        self.root = root
        self.setup_styles()
        self.animation_frames = []
        self.load_animations()
        self.create_widgets()

    def setup_styles(self):
        """Setup custom styles for the UI"""
        style = ttk.Style()
        style.configure(
            "Jarvis.TFrame", background="#1E1E1E", borderwidth=2, relief="raised"
        )
        style.configure(
            "Jarvis.TLabel",
            background="#1E1E1E",
            foreground="#00FF00",
            font=("Rajdhani", 12),
        )
        style.configure(
            "Jarvis.TButton",
            background="#2E2E2E",
            foreground="#00FF00",
            font=("Rajdhani", 10),
            padding=10,
        )

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        self.main_frame = ttk.Frame(self.root, style="Jarvis.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header
        self.create_header()

        # Create main content area
        self.create_main_content()

        # Create status bar
        self.create_status_bar()

    def create_header(self):
        """Create the header section"""
        header_frame = ttk.Frame(self.main_frame, style="Jarvis.TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        # Logo and title
        title_label = ttk.Label(
            header_frame,
            text="J.A.R.V.I.S",
            font=("Rajdhani", 24, "bold"),
            foreground="#00FF00",
            background="#1E1E1E",
        )
        title_label.pack(side=tk.LEFT)

        # System metrics
        self.metrics_frame = ttk.Frame(header_frame, style="Jarvis.TFrame")
        self.metrics_frame.pack(side=tk.RIGHT)

        # CPU, Memory, Network indicators
        self.create_metric_indicators()

    def create_metric_indicators(self):
        """Create system metric indicators"""
        # CPU Usage
        self.cpu_label = ttk.Label(
            self.metrics_frame, text="CPU: 0%", style="Jarvis.TLabel"
        )
        self.cpu_label.pack(side=tk.LEFT, padx=10)

        # Memory Usage
        self.memory_label = ttk.Label(
            self.metrics_frame, text="MEM: 0%", style="Jarvis.TLabel"
        )
        self.memory_label.pack(side=tk.LEFT, padx=10)

        # Network Status
        self.network_label = ttk.Label(
            self.metrics_frame, text="NET: Active", style="Jarvis.TLabel"
        )
        self.network_label.pack(side=tk.LEFT, padx=10)

    def create_main_content(self):
        """Create the main content area"""
        content_frame = ttk.Frame(self.main_frame, style="Jarvis.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create split view
        self.create_split_view(content_frame)

    def create_split_view(self, parent):
        """Create split view with chat and visualization"""
        # Left panel - Chat
        self.chat_frame = ttk.Frame(parent, style="Jarvis.TFrame")
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Chat display
        self.create_chat_display()

        # Right panel - Visualization
        self.viz_frame = ttk.Frame(parent, style="Jarvis.TFrame")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))

        # Visualization canvas
        self.create_visualization()

    def create_chat_display(self):
        """Create the chat display area"""
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            bg="#1E1E1E",
            fg="#00FF00",
            font=("Rajdhani", 11),
            insertbackground="#00FF00",
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure tags for different message types
        self.chat_display.tag_configure(
            "user", foreground="#FFFFFF", font=("Rajdhani", 11)
        )
        self.chat_display.tag_configure(
            "jarvis", foreground="#00FF00", font=("Rajdhani", 11, "bold")
        )
        self.chat_display.tag_configure(
            "system", foreground="#FFA500", font=("Rajdhani", 10, "italic")
        )

    def create_visualization(self):
        """Create visualization area"""
        self.viz_canvas = tk.Canvas(
            self.viz_frame, width=300, height=400, bg="#1E1E1E", highlightthickness=0
        )
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.main_frame, style="Jarvis.TFrame")
        status_frame.pack(fill=tk.X, padx=20, pady=10)

        # Status message
        self.status_label = ttk.Label(
            status_frame,
            text="Ready - Say 'Hey Jarvis' to begin",
            style="Jarvis.TLabel",
        )
        self.status_label.pack(side=tk.LEFT)

        # Control buttons
        self.create_control_buttons(status_frame)

    def create_control_buttons(self, parent):
        """Create control buttons"""
        # Settings button
        self.settings_btn = ttk.Button(
            parent,
            text="⚙️ Settings",
            style="Jarvis.TButton",
            command=self.show_settings,
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=5)

        # Mute button
        self.mute_var = tk.BooleanVar(value=False)
        self.mute_btn = ttk.Checkbutton(
            parent, text="🎤 Mute", style="Jarvis.TButton", variable=self.mute_var
        )
        self.mute_btn.pack(side=tk.RIGHT, padx=5)

    def update_metrics(self, metrics):
        """Update system metrics display"""
        try:
            if "cpu" in metrics:
                self.cpu_label.configure(
                    text=f"CPU: {metrics['cpu']['current']}% "
                    f"({'↑' if metrics['cpu']['change'] > 0 else '↓'}{abs(metrics['cpu']['change']):.1f}%)"
                )

            if "memory" in metrics:
                self.memory_label.configure(
                    text=f"MEM: {metrics['memory']['current']}% "
                    f"({'↑' if metrics['memory']['change'] > 0 else '↓'}{abs(metrics['memory']['change']):.1f}%)"
                )

            if "network" in metrics and isinstance(metrics["network"], dict):
                status = metrics["network"].get("status", "Unknown")
                self.network_label.configure(text=f"NET: {status}")

        except Exception as e:
            logging.error(f"Error updating metrics display: {str(e)}")

    def add_message(self, message, message_type="system"):
        """Add a message to the chat display"""
        self.chat_display.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
        self.chat_display.insert(tk.END, f"{message}\n", message_type)
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    def update_status(self, status):
        """Update status message"""
        self.status_label.configure(text=status)

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("J.A.R.V.I.S Settings")
        settings_window.geometry("500x600")
        settings_window.configure(bg="#1E1E1E")

        # Create main frame
        settings_frame = ttk.Frame(settings_window, style="Jarvis.TFrame")
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Add API Key section
        self.create_api_settings(settings_frame)
        # Add Voice Settings section
        self.create_voice_settings(settings_frame)

    def create_api_settings(self, parent):
        """Create API settings section"""
        api_frame = ttk.LabelFrame(parent, text="API Settings", style="Jarvis.TFrame")
        api_frame.pack(fill=tk.X, pady=10)

        ttk.Label(api_frame, text="Claude API Key:", style="Jarvis.TLabel").pack(pady=5)
        self.api_key_entry = ttk.Entry(api_frame, width=40)
        self.api_key_entry.pack(pady=5)

    def create_voice_settings(self, parent):
        """Create voice settings section"""
        voice_frame = ttk.LabelFrame(
            parent, text="Voice Settings", style="Jarvis.TFrame"
        )
        voice_frame.pack(fill=tk.X, pady=10)

        # Language selection
        lang_frame = ttk.Frame(voice_frame)
        lang_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lang_frame, text="Language:", style="Jarvis.TLabel").pack(
            side=tk.LEFT
        )

        self.language_var = tk.StringVar(value="en")
        languages = {"English": "en", "Spanish": "es", "French": "fr"}
        language_menu = ttk.OptionMenu(
            lang_frame, self.language_var, "English", *languages.keys()
        )
        language_menu.pack(side=tk.LEFT, padx=5)

    def load_animations(self):
        """Load animation frames with listening animation"""
        try:
            self.animation_frames = {
                "wake": self.create_wake_animation(),
                "process": self.create_process_animation(),
                "standby": self.create_standby_animation(),
                "listening": self.create_listening_animation(),
            }
        except Exception as e:
            self.logger.error(f"Failed to load animations: {e}")
            self.animation_frames = {
                "wake": [],
                "process": [],
                "standby": [],
                "listening": [],
            }

    def create_listening_animation(self):
        """Create listening animation frames"""
        frames = []
        size = 100
        for i in range(8):
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            # Draw pulsing microphone icon
            center_x = size // 2
            center_y = size // 2
            radius = 20 + abs(math.sin(i * math.pi / 4) * 10)  # Pulsing effect

            # Microphone body
            draw.rectangle(
                [center_x - 5, center_y - 15, center_x + 5, center_y + 5],
                fill="#00FF00",
            )
            # Microphone base
            draw.ellipse(
                [center_x - 10, center_y + 5, center_x + 10, center_y + 15],
                fill="#00FF00",
            )
            # Sound waves
            draw.arc(
                [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                0,
                360,
                fill="#00FF00",
            )

            photo = ImageTk.PhotoImage(image)
            frames.append(photo)
        return frames

    def create_wake_animation(self):
        """Create wake animation frames"""
        frames = []
        size = 100
        for i in range(8):
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            angle = i * 45
            x = size / 2 + size / 3 * np.cos(np.radians(angle))
            y = size / 2 + size / 3 * np.sin(np.radians(angle))
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#00FF00")
            photo = ImageTk.PhotoImage(image)
            frames.append(photo)
        return frames

    def create_process_animation(self):
        """Create processing animation frames"""
        frames = []
        size = 100
        for i in range(6):
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            radius = 20 + i * 5
            draw.ellipse(
                [
                    size / 2 - radius,
                    size / 2 - radius,
                    size / 2 + radius,
                    size / 2 + radius,
                ],
                outline="#00FF00",
            )
            photo = ImageTk.PhotoImage(image)
            frames.append(photo)
        return frames

    def create_standby_animation(self):
        """Create standby animation frames"""
        frames = []
        size = 100
        for i in range(4):
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            intensity = 128 + i * 32
            color = f"#{intensity:02x}FF{intensity:02x}"
            draw.rectangle([0, size // 2 - 2, size, size // 2 + 2], fill=color)
            photo = ImageTk.PhotoImage(image)
            frames.append(photo)
        return frames

    def show_animation(self, animation_type):
        """Show a specific animation"""
        if (
            animation_type in self.animation_frames
            and self.animation_frames[animation_type]
        ):
            self.play_animation(self.animation_frames[animation_type])

    def play_animation(self, frames):
        """Play an animation sequence"""

        def update_frame(frame_index=0):
            if frame_index < len(frames):
                self.viz_canvas.delete("all")
                self.viz_canvas.create_image(
                    150, 200, image=frames[frame_index]  # Center position
                )
                self.root.after(100, update_frame, (frame_index + 1) % len(frames))

        update_frame()


class ConversationMemory:
    """Handles conversation history and context"""

    def __init__(self, max_history=10):
        self.history = collections.deque(maxlen=max_history)
        self.context_window = max_history
        self.logger = logging.getLogger("JARVIS.ConversationMemory")

    def add_interaction(self, user_input, response, metadata=None):
        """Add an interaction to memory"""
        try:
            interaction = {
                "timestamp": datetime.now(),
                "user_input": user_input,
                "response": response,
                "metadata": metadata or {},
            }
            self.history.append(interaction)
            return True
        except Exception as e:
            self.logger.error(f"Error adding interaction to memory: {str(e)}")
            return False

    def get_recent_context(self, limit=3):
        """Get recent conversation context"""
        try:
            recent = list(self.history)[-limit:]
            context = []
            for interaction in recent:
                context.append(
                    {
                        "user": interaction["user_input"],
                        "jarvis": interaction["response"],
                    }
                )
            return context
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {str(e)}")
            return []

    def clear_history(self):
        """Clear conversation history"""
        try:
            self.history.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing conversation history: {str(e)}")
            return False


class WakeWordDetector:
    """Handles wake word detection"""

    def __init__(self, callback=None):
        self.callback = callback
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        self._vad_mode = 3  # Store mode as private attribute
        self.is_listening = False
        self.buffer_queue = queue.Queue()
        self.audio_buffer = collections.deque(maxlen=30)
        self.logger = logging.getLogger("JARVIS.WakeWordDetector")

        # Audio parameters
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1

        # Wake word detection parameters
        self.wake_word_threshold = 3000  # Energy threshold for detection
        self.consecutive_detections_needed = 3  # Required consecutive detections
        self.consecutive_detections = 0  # Current detection count
        self.detection_window = 0.5  # Time window for detections
        self.last_detection_time = None  # Timestamp of last detection

        # Error recovery parameters
        self.error_count = 0
        self.max_errors = 3
        self.last_error_time = None
        self.error_cooldown = 60  # seconds
        self.processing_thread = None

        # Test audio device during initialization
        if not self.test_audio_device():
            raise Exception("Audio device initialization failed")

        self.logger.debug("WakeWordDetector initialized successfully")

    def start(self):
        """Start wake word detection"""
        try:
            if not self.is_listening:
                self.logger.info("Starting wake word detector")
                self.is_listening = True
                self.error_count = 0
                self.last_error_time = None

                self.processing_thread = threading.Thread(
                    target=self._audio_processing_loop, daemon=True
                )
                self.processing_thread.start()

                self.logger.info("Wake word detector started successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to start wake word detector: {str(e)}")
            self.is_listening = False
            return False

    def stop(self):
        """Stop wake word detection"""
        try:
            self.logger.info("Stopping wake word detector")
            self.is_listening = False

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            self.logger.info("Wake word detector stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping wake word detector: {str(e)}")

    def _audio_processing_loop(self):
        """Main audio processing loop"""
        audio = None
        stream = None

        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )

            self.logger.debug("Audio processing loop started")

            while self.is_listening:
                try:
                    frame = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(frame, self.RATE)

                    if is_speech:
                        self._process_speech_frame(frame)
                    else:
                        self._reset_detection()

                except OSError as e:
                    self.logger.warning(f"Audio stream error: {str(e)}")
                    self.handle_audio_error()

        except Exception as e:
            self.logger.error(f"Error in audio processing loop: {str(e)}")
        finally:
            self.cleanup_audio(stream, audio)

    def _process_speech_frame(self, frame):
        """Process a speech frame"""
        try:
            self.audio_buffer.append(frame)

            if len(self.audio_buffer) >= 20:
                frames = b"".join(list(self.audio_buffer))
                audio_data = np.frombuffer(frames, dtype=np.int16)
                energy = np.abs(audio_data).mean()

                self.logger.debug(f"Audio energy level: {energy}")

                if energy > self.wake_word_threshold:
                    current_time = time.time()

                    if (
                        self.last_detection_time is None
                        or current_time - self.last_detection_time
                        <= self.detection_window
                    ):
                        self.consecutive_detections += 1
                    else:
                        self.consecutive_detections = 1

                    self.last_detection_time = current_time

                    if (
                        self.consecutive_detections
                        >= self.consecutive_detections_needed
                    ):
                        self.logger.debug(f"Wake word detected with energy: {energy}")
                        if self.callback:
                            self.callback()
                        self._reset_detection()

                self.audio_buffer.clear()

        except Exception as e:
            self.logger.error(f"Error processing speech frame: {str(e)}")
            self._reset_detection()

    def _reset_detection(self):
        """Reset detection state"""
        self.audio_buffer.clear()
        self.consecutive_detections = 0
        self.last_detection_time = None

    def handle_audio_error(self):
        """Handle audio stream errors"""
        current_time = time.time()

        if (
            self.last_error_time
            and (current_time - self.last_error_time) < self.error_cooldown
        ):
            self.error_count += 1
        else:
            self.error_count = 1

        self.last_error_time = current_time

        if self.error_count >= self.max_errors:
            self.logger.error("Too many audio errors, stopping detector")
            self.stop()

    def test_audio_device(self):
        """Test audio device availability"""
        try:
            audio = pyaudio.PyAudio()
            device_info = audio.get_default_input_device_info()
            self.logger.debug(f"Default input device: {device_info['name']}")

            # Test opening stream
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )

            stream.close()
            audio.terminate()
            return True

        except Exception as e:
            self.logger.error(f"Audio device test failed: {str(e)}")
            return False

    def get_audio_devices(self):
        """Get list of available audio devices"""
        devices = []
        try:
            audio = pyaudio.PyAudio()
            for i in range(audio.get_device_count()):
                try:
                    device_info = audio.get_device_info_by_index(i)
                    devices.append(
                        {
                            "index": i,
                            "name": device_info["name"],
                            "input_channels": device_info["maxInputChannels"],
                            "output_channels": device_info["maxOutputChannels"],
                            "default_sample_rate": device_info["defaultSampleRate"],
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error getting device info for index {i}: {str(e)}"
                    )
            audio.terminate()
        except Exception as e:
            self.logger.error(f"Error getting audio devices: {str(e)}")
        return devices

    def cleanup_audio(self, stream, audio):
        """Safely cleanup audio resources"""
        try:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            if audio:
                audio.terminate()
        except Exception as e:
            self.logger.error(f"Error during audio cleanup: {str(e)}")


class ScheduleManager:
    """Manages schedule and calendar functionality"""

    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.schedule = {}
        self.reminders = []
        self.logger = logging.getLogger("JARVIS.ScheduleManager")

    async def handle_schedule_command(self, command):
        """Handle schedule-related commands"""
        try:
            prompt = (
                "Extract scheduling information from the command below. "
                "Return a JSON object with 'action' (add/remove/view), "
                "'date', 'time', 'duration', and 'description' fields.\n\n"
                f"Command: {command}"
            )

            response = await self.jarvis.get_claude_response(prompt)
            schedule_info = json.loads(response)

            if schedule_info["action"] == "add":
                success = self.add_event(
                    schedule_info["date"],
                    schedule_info["time"],
                    schedule_info["duration"],
                    schedule_info["description"],
                )
                if success:
                    response = f"I've scheduled {schedule_info['description']} for {schedule_info['date']} at {schedule_info['time']}."
                else:
                    response = "I was unable to schedule that event. Please try again."

            elif schedule_info["action"] == "remove":
                success = self.remove_event(
                    schedule_info["date"], schedule_info["time"]
                )
                if success:
                    response = "Event removed from schedule."
                else:
                    response = "I couldn't find that event in your schedule."

            elif schedule_info["action"] == "view":
                events = self.get_events(schedule_info["date"])
                if events:
                    response = f"Here's your schedule for {schedule_info['date']}:\n"
                    for event in events:
                        response += f"- {event['time']}: {event['description']} ({event['duration']})\n"
                else:
                    response = (
                        f"You have no events scheduled for {schedule_info['date']}."
                    )

            self.jarvis.safe_ui_update(
                self.jarvis.ui.add_message, f"JARVIS: {response}", "jarvis"
            )
            self.jarvis.speak(response)

        except Exception as e:
            self.logger.error(f"Error handling schedule command: {str(e)}")
            self.jarvis.handle_error("I had trouble processing that schedule request.")

    def add_event(self, date, time, duration, description):
        """Add an event to the schedule"""
        try:
            if date not in self.schedule:
                self.schedule[date] = []

            event = {"time": time, "duration": duration, "description": description}

            self.schedule[date].append(event)
            self.schedule[date].sort(key=lambda x: x["time"])
            return True

        except Exception as e:
            self.logger.error(f"Error adding event: {str(e)}")
            return False

    def remove_event(self, date, time):
        """Remove an event from the schedule"""
        try:
            if date in self.schedule:
                for event in self.schedule[date]:
                    if event["time"] == time:
                        self.schedule[date].remove(event)
                        if not self.schedule[date]:
                            del self.schedule[date]
                        return True
            return False

        except Exception as e:
            self.logger.error(f"Error removing event: {str(e)}")
            return False

    def get_events(self, date):
        """Get events for a specific date"""
        try:
            return self.schedule.get(date, [])
        except Exception as e:
            self.logger.error(f"Error getting events: {str(e)}")
            return []


class VoiceEnhancer:
    """Handles enhanced voice interaction features"""

    def __init__(self, jarvis_instance):
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger("JARVIS.VoiceEnhancer")
        self.voice_patterns = self.load_voice_patterns()

    def load_voice_patterns(self):
        """Load voice interaction patterns"""
        return {
            "confirmation": [
                "Of course, sir.",
                "Right away, sir.",
                "Consider it done.",
                "I'm on it.",
            ],
            "clarification": [
                "Could you please clarify that, sir?",
                "I'm not quite sure I understood. Could you rephrase that?",
                "Would you mind providing more details?",
                "Could you be more specific?",
            ],
            "thinking": [
                "Let me process that...",
                "Computing response...",
                "Analyzing request...",
                "Processing...",
            ],
            "greeting": [
                "Welcome back, sir.",
                "At your service, sir.",
                "How can I assist you today, sir?",
                "Ready and operational, sir.",
            ],
        }

    def get_response_pattern(self, pattern_type):
        """Get a random response pattern"""
        try:
            patterns = self.voice_patterns.get(pattern_type, [])
            if patterns:
                return random.choice(patterns)
            return ""
        except Exception as e:
            self.logger.error(f"Error getting response pattern: {str(e)}")
            return ""

    def enhance_response(self, response, context=None):
        """Enhance response with personality and context"""
        try:
            if not response:
                return ""

            enhanced_response = response

            # Add thinking indicator for long responses
            if len(response) > 100:
                thinking = self.get_response_pattern("thinking")
                if thinking:
                    self.jarvis.safe_ui_update(
                        self.jarvis.ui.add_message, f"JARVIS: {thinking}", "jarvis"
                    )

            # Add confirmation for task-related responses
            if any(
                word in response.lower()
                for word in ["schedule", "remind", "note", "task"]
            ):
                confirmation = self.get_response_pattern("confirmation")
                if confirmation:
                    enhanced_response = f"{confirmation} {enhanced_response}"

            # Normalize text for better speech
            enhanced_response = self.normalize_speech(enhanced_response)

            return enhanced_response

        except Exception as e:
            self.logger.error(f"Error enhancing response: {str(e)}")
            return response

    def normalize_speech(self, text):
        """Normalize speech for better pronunciation"""
        try:
            # Common replacements for better speech
            replacements = {
                "JARVIS": "Jarvis",
                "AI": "A.I.",
                "UI": "U.I.",
                "CPU": "C.P.U.",
                "RAM": "ram",
                "GPU": "G.P.U.",
                "MHz": "megahertz",
                "GHz": "gigahertz",
                "KB": "kilobytes",
                "MB": "megabytes",
                "GB": "gigabytes",
                "TB": "terabytes",
            }

            for key, value in replacements.items():
                text = text.replace(key, value)

            return text

        except Exception as e:
            self.logger.error(f"Error normalizing speech: {str(e)}")
            return text

    def process_wake_word(self, audio_data):
        """Process wake word detection"""
        try:
            # Convert audio data to numpy array for processing
            audio_numpy = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate audio energy
            energy = np.abs(audio_numpy).mean()

            # Get threshold from config or use default
            threshold = self.jarvis.config.get("voice_settings", {}).get(
                "energy_threshold", 4000
            )

            # Check if energy is above threshold
            return energy > threshold

        except Exception as e:
            self.logger.error(f"Error processing wake word: {str(e)}")
            return False


class Jarvis:
    def __init__(self):
        # Initialize logging first
        self.logger = LoggerSetup.setup_logging()
        self.logger.info("Initializing JARVIS")

        # Initialize core components and state
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.active = True
        self.is_processing = False
        self.settings_updated = EventEmitter()

        # Initialize command patterns
        self.command_patterns = {
            "system": [
                "shutdown",
                "restart",
                "status",
                "mute",
                "unmute",
                "system",
                "settings",
                "configure",
            ],
            "query": [
                "what",
                "who",
                "when",
                "where",
                "why",
                "how",
                "tell me",
                "find",
                "search",
                "look up",
            ],
            "task": [
                "remind",
                "schedule",
                "note",
                "create",
                "set",
                "make",
                "add",
                "delete",
                "remove",
                "update",
            ],
        }

        # Setup all systems
        try:
            self.setup_command_handlers()  # Setup command handlers first
            self.setup_application()  # Then setup application
            self.initialize_systems()  # Initialize all systems
            self.start_background_processes()  # Start background processes
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    def setup_application(self):
        """Setup main application window and UI"""
        try:
            self.root = tk.Tk()
            self.root.title("J.A.R.V.I.S")
            self.root.geometry("1200x800")
            self.root.configure(bg="#1E1E1E")

            # Initialize UI
            self.ui = AdvancedUI(self.root)

            # Setup periodic UI updates
            self.setup_ui_updates()
        except Exception as e:
            self.logger.error(f"Error setting up application: {str(e)}")
            raise

    def setup_ui_updates(self):
        """Setup periodic UI updates"""

        def update_ui():
            if self.active:
                try:
                    if hasattr(self, "system_monitor"):
                        metrics = self.system_monitor.get_system_metrics()
                        self.ui.update_metrics(metrics)

                    while not self.response_queue.empty():
                        response = self.response_queue.get_nowait()
                        self.ui.add_message(response["message"], response["type"])
                        if response.get("speak", False):
                            self.speak(response["message"])

                except Exception as e:
                    self.logger.error(f"Error in UI update: {str(e)}")

                # Schedule next update if still active
                if self.active:
                    self.root.after(1000, update_ui)

        # Start UI updates
        self.root.after(1000, update_ui)

    def setup_command_handlers(self):
        """Setup command handling system"""
        self.command_handlers = {
            "system": self.handle_system_command,
            "query": self.handle_query_command,
            "task": self.handle_task_command,
            "general": self.handle_general_command,
        }

    def start_background_processes(self):
        """Start all background processes"""
        try:
            # Start command processing thread
            self.command_thread = threading.Thread(
                target=self.process_commands_loop, daemon=True
            )
            self.command_thread.start()
            self.logger.info("Command processing thread started")

            # Start wake word detection if available
            if hasattr(self, "wake_detector") and self.wake_detector:
                try:
                    self.wake_detector.start()
                    self.logger.info("Wake word detection started")
                except Exception as e:
                    self.logger.error(f"Failed to start wake word detector: {str(e)}")
                    self.services_status["wake_word"] = False
            else:
                self.logger.warning("Wake word detector not available")
                self.services_status["wake_word"] = False

            # Initialize thread status tracking
            self.background_threads = {
                "command_processing": True,
                "wake_detection": self.services_status.get("wake_word", False),
            }

        except Exception as e:
            self.logger.error(f"Error starting background processes: {str(e)}")
            raise

    def process_commands_loop(self):
        """Main command processing loop"""
        while self.active:
            try:
                # Get command from queue with timeout
                try:
                    command = self.command_queue.get(timeout=0.1)
                    if command:
                        command_type = self.classify_command(command["text"])
                        asyncio.run(self.process_command(command["text"], command_type))
                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"Error in command processing loop: {str(e)}")

            time.sleep(0.01)

    async def process_command(self, command_text, command_type="general"):
        """Process a single command"""
        try:
            self.is_processing = True
            self.safe_ui_update(self.ui.show_animation, "process")

            # Get thinking response if voice enhancer is available
            if hasattr(self, "voice_enhancer"):
                thinking = self.voice_enhancer.get_response_pattern("thinking")
                self.safe_ui_update(
                    self.ui.add_message, f"JARVIS: {thinking}", "jarvis"
                )

            # Get appropriate handler
            handler = self.command_handlers.get(
                command_type, self.handle_general_command
            )

            # Process command
            response = await handler(command_text)

            if response:
                # Enhance response if voice enhancer is available
                if hasattr(self, "voice_enhancer"):
                    enhanced_response = self.voice_enhancer.enhance_response(response)
                else:
                    enhanced_response = response

                # Store in conversation memory if available
                if hasattr(self, "conversation_memory"):
                    self.conversation_memory.add_interaction(
                        command_text, enhanced_response, {"type": command_type}
                    )

                # Deliver response
                self.safe_ui_update(
                    self.ui.add_message, f"JARVIS: {enhanced_response}", "jarvis"
                )
                self.speak(enhanced_response)

        except Exception as e:
            self.logger.error(f"Error processing command: {str(e)}")
            self.handle_error("I encountered an error processing your request.")
        finally:
            self.is_processing = False

    def classify_command(self, command):
        """Classify the type of command"""
        try:
            cmd_lower = command.lower()

            # Check system commands first
            if any(word in cmd_lower for word in self.command_patterns["system"]):
                return "system"

            # Check queries
            if any(word in cmd_lower for word in self.command_patterns["query"]):
                return "query"

            # Check tasks
            if any(word in cmd_lower for word in self.command_patterns["task"]):
                return "task"

            # Default to general conversation
            return "general"

        except Exception as e:
            self.logger.error(f"Error classifying command: {str(e)}")
            return "general"

    def safe_ui_update(self, method, *args, **kwargs):
        """Safely update UI from any thread"""
        if hasattr(self, "root") and self.active:
            try:
                self.root.after(0, lambda: method(*args, **kwargs))
            except Exception as e:
                self.logger.error(f"Error in UI update: {str(e)}")

    def handle_wake_word(self):
        """Handle wake word detection"""
        if not self.is_processing:
            try:
                self.is_processing = True
                self.safe_ui_update(self.ui.show_animation, "wake")

                # Get contextual greeting based on time of day
                hour = datetime.now().hour
                if 5 <= hour < 12:
                    time_context = "Good morning"
                elif 12 <= hour < 17:
                    time_context = "Good afternoon"
                else:
                    time_context = "Good evening"

                welcome = self.personality.format_response(
                    "greeting", f"{time_context}, sir. How may I assist you?"
                )

                # Update UI with message
                self.safe_ui_update(self.ui.add_message, f"JARVIS: {welcome}", "jarvis")

                # Speak the welcome message
                self.speak(welcome)

                # Start listening for command after speaking
                def start_listening():
                    self.is_processing = False
                    self.listen_for_command()

                # Schedule listening after a short delay
                self.root.after(1000, start_listening)

            except Exception as e:
                self.logger.error(f"Error handling wake word: {str(e)}")
                self.is_processing = False

    def listen_for_command(self):
        """Listen for user command"""
        if self.services_status.get("voice", False):
            try:
                with sr.Microphone() as source:
                    self.safe_ui_update(self.ui.update_status, "Listening...")

                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                    # Update UI to show listening state
                    self.safe_ui_update(self.ui.show_animation, "listening")

                    # Get audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.config["system_settings"]["record_timeout"],
                        phrase_time_limit=self.config["system_settings"][
                            "phrase_timeout"
                        ],
                    )

                    # Process the audio
                    self.process_audio(audio)

            except sr.WaitTimeoutError:
                self.safe_ui_update(
                    self.ui.add_message,
                    "JARVIS: I didn't hear anything. Please try again.",
                    "jarvis",
                )
                self.speak("I didn't hear anything. Please try again.")
            except Exception as e:
                self.logger.error(f"Error listening for command: {str(e)}")
            finally:
                self.safe_ui_update(self.ui.update_status, "Ready")
                self.safe_ui_update(self.ui.show_animation, "standby")

    def process_audio(self, audio):
        """Process recorded audio"""
        try:
            text = self.recognizer.recognize_google(audio)

            if text:
                # Remove wake word if present
                text = text.lower().replace("hey jarvis", "").strip()
                if text:
                    # Show processing animation
                    self.safe_ui_update(self.ui.show_animation, "process")

                    # Add to command queue
                    self.command_queue.put({"text": text, "type": "voice"})

                    # Show in UI
                    self.safe_ui_update(self.ui.add_message, f"You: {text}", "user")

        except sr.UnknownValueError:
            self.speak("I didn't catch that. Could you repeat?")
        except sr.RequestError:
            self.safe_ui_update(
                self.ui.add_message,
                "JARVIS: There seems to be an issue with the speech recognition service.",
                "jarvis",
            )
            self.speak("I'm having trouble with speech recognition. Please try again.")
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")

    def speak(self, text):
        """Enhanced text to speech output"""
        try:
            # Clean up text and fix pronunciation
            text = text.replace("JARVIS:", "").strip()
            text = text.replace("JARVIS", "Jarvis")

            if self.services_status.get("voice", False) and not self.ui.mute_var.get():

                def speak_thread():
                    try:
                        # Initialize audio only when needed
                        if pygame.mixer.get_init() is None:
                            pygame.mixer.init(
                                frequency=44100, size=-16, channels=2, buffer=2048
                            )

                        # Create temp directory if it doesn't exist
                        os.makedirs("temp", exist_ok=True)
                        temp_file = os.path.join(
                            "temp", f"response_{int(time.time())}.mp3"
                        )

                        # Generate speech
                        tts = gTTS(
                            text=text,
                            lang=self.config["voice_settings"]["language"],
                            slow=False,
                        )
                        tts.save(temp_file)

                        try:
                            pygame.mixer.music.load(temp_file)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                pygame.time.Clock().tick(10)
                        finally:
                            # Clean up
                            pygame.mixer.music.unload()
                            if os.path.exists(temp_file):
                                os.remove(temp_file)

                    except Exception as e:
                        self.logger.error(f"Speech error in thread: {str(e)}")

                threading.Thread(target=speak_thread, daemon=True).start()

        except Exception as e:
            self.logger.error(f"Speech error: {str(e)}")

    async def handle_system_command(self, command):
        """Handle system-related commands"""
        try:
            cmd_lower = command.lower()

            if "shutdown" in cmd_lower or "goodbye" in cmd_lower:
                return await self.handle_shutdown()
            elif "restart" in cmd_lower:
                return await self.handle_restart()
            elif "status" in cmd_lower:
                return await self.show_system_status()
            elif "mute" in cmd_lower or "unmute" in cmd_lower:
                return self.toggle_mute()
            elif "settings" in cmd_lower:
                return self.show_settings()
            else:
                return "I'm not sure how to handle that system command, sir."

        except Exception as e:
            self.logger.error(f"Error in system command: {str(e)}")
            return "I encountered an error processing that system command."

    async def handle_query_command(self, command):
        """Handle information queries"""
        try:
            prompt = (
                "You are JARVIS, responding to an information query. "
                "Be precise and informative while maintaining your personality. "
                f"\nQuery: {command}"
            )

            response = await self.get_claude_response(prompt)
            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return "I'm having trouble accessing that information at the moment."

    async def handle_task_command(self, command):
        """Handle task execution commands"""
        try:
            cmd_lower = command.lower()

            if "remind" in cmd_lower:
                return await self.handle_reminder(command)
            elif "schedule" in cmd_lower:
                return await self.handle_schedule(command)
            elif "note" in cmd_lower:
                return await self.handle_note(command)
            else:
                return await self.handle_general_command(command)

        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            return "I encountered an error while processing that task."

    async def handle_general_command(self, command):
        """Handle general conversation and commands"""
        try:
            prompt = self.personality.create_contextual_prompt(command)
            response = await self.get_claude_response(prompt)
            return response

        except Exception as e:
            self.logger.error(f"Error processing command: {str(e)}")
            return "I'm having trouble processing that request."

    def handle_error(self, message):
        """Handle errors with user feedback"""
        try:
            error_response = self.personality.format_response("error", message)
            self.safe_ui_update(
                self.ui.add_message, f"JARVIS: {error_response}", "jarvis"
            )
            self.speak(error_response)
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")

    async def handle_reminder(self, command):
        """Handle reminder creation"""
        try:
            prompt = (
                "Extract reminder details from the following command. "
                "Respond with only the extracted information in JSON format "
                "with 'time' and 'task' fields.\n\n"
                f"Command: {command}"
            )

            response = await self.get_claude_response(prompt)
            details = json.loads(response)
            success = self.schedule_reminder(details["time"], details["task"])

            if success:
                return f"I'll remind you to {details['task']} at {details['time']}"
            else:
                return (
                    "I couldn't schedule that reminder. Please check the time format."
                )

        except Exception as e:
            self.logger.error(f"Error setting reminder: {str(e)}")
            return "I had trouble processing that reminder."

    def schedule_reminder(self, time_str, task):
        """Schedule a reminder"""
        try:
            reminder_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

            def reminder_callback():
                response = f"Sir, you asked me to remind you to {task}"
                self.safe_ui_update(
                    self.ui.add_message, f"JARVIS: Reminder - {task}", "jarvis"
                )
                self.speak(response)

            now = datetime.now()
            delay = (reminder_time - now).total_seconds() * 1000

            if delay > 0:
                self.root.after(int(delay), reminder_callback)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error scheduling reminder: {str(e)}")
            return False

    async def handle_schedule(self, command):
        """Handle schedule management"""
        try:
            prompt = (
                "Extract scheduling information from the command below. "
                "Return a JSON object with 'action' (add/remove/view), "
                "'date', 'time', 'duration', and 'description' fields.\n\n"
                f"Command: {command}"
            )

            schedule_info = await self.get_claude_response(prompt)
            details = json.loads(schedule_info)

            if details["action"] == "add":
                event_time = f"{details['date']} {details['time']}"
                success = self.schedule_event(
                    event_time, details["duration"], details["description"]
                )
                if success:
                    return f"I've scheduled {details['description']} for {event_time}."
                else:
                    return (
                        "I couldn't schedule that event. Please check the time format."
                    )

            elif details["action"] == "view":
                return await self.get_schedule(details["date"])

            else:
                return "I'm not sure how to handle that schedule request."

        except Exception as e:
            self.logger.error(f"Error handling schedule command: {str(e)}")
            return "I had trouble processing that schedule request."

    def schedule_event(self, time_str, duration, description):
        """Schedule an event"""
        try:
            event_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

            def event_callback():
                response = f"Sir, you have an event: {description}"
                self.safe_ui_update(
                    self.ui.add_message, f"JARVIS: Event - {description}", "jarvis"
                )
                self.speak(response)

            now = datetime.now()
            delay = (event_time - now).total_seconds() * 1000

            if delay > 0:
                self.root.after(int(delay), event_callback)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error scheduling event: {str(e)}")
            return False

    async def handle_note(self, command):
        """Handle note taking"""
        try:
            note_content = (
                command.replace("take a note", "")
                .replace("make a note", "")
                .replace("create a note", "")
                .strip()
            )

            if note_content:
                notes_dir = "notes"
                os.makedirs(notes_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                note_file = f"{notes_dir}/note_{timestamp}.txt"

                with open(note_file, "w") as f:
                    f.write(note_content)

                return "I've made a note of that, sir."
            else:
                return "I couldn't determine what to note down. Please try again."

        except Exception as e:
            self.logger.error(f"Error taking note: {str(e)}")
            return "I encountered an error while taking that note."

    async def handle_shutdown(self):
        """Handle system shutdown"""
        response = "Initiating shutdown sequence. Goodbye, sir."
        self.root.after(2000, self.shutdown)
        return response

    def shutdown(self):
        """Perform actual shutdown"""
        try:
            self.cleanup()
            self.root.quit()
        except Exception as e:
            self.logger.error(f"Error in shutdown: {str(e)}")

    async def handle_restart(self):
        """Handle system restart"""
        try:
            response = "Initiating restart sequence. Please wait..."
            self.safe_ui_update(self.ui.add_message, f"JARVIS: {response}", "jarvis")

            # Cleanup current systems
            self.cleanup()

            # Reinitialize systems
            self.initialize_systems()
            self.start_background_processes()

            return "Restart complete. All systems operational."

        except Exception as e:
            self.logger.error(f"Error during restart: {str(e)}")
            return "Error during restart sequence."

    async def show_system_status(self):
        """Show system status"""
        try:
            metrics = self.system_monitor.get_system_metrics()
            active_services = [k for k, v in self.services_status.items() if v]

            status_msg = (
                f"System Status Report:\n"
                f"CPU Usage: {metrics['cpu']['current']}%\n"
                f"Memory Usage: {metrics['memory']['current']}%\n"
                f"Active Services: {', '.join(active_services)}\n"
                f"Network Status: {metrics['network']['status']}\n"
                f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return status_msg

        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return "I'm having trouble accessing system status."

    def toggle_mute(self):
        """Toggle voice output"""
        try:
            self.ui.mute_var.set(not self.ui.mute_var.get())
            status = "muted" if self.ui.mute_var.get() else "unmuted"
            return f"Voice output {status}."

        except Exception as e:
            self.logger.error(f"Error toggling mute: {str(e)}")
            return "Error toggling voice output."

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources...")
        self.active = False

        if hasattr(self, "wake_detector"):
            self.wake_detector.stop()

        if hasattr(self, "command_thread"):
            self.command_thread.join(timeout=1.0)

        if pygame.mixer.get_init():
            pygame.mixer.quit()

        # Clean up temp files
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass

        self.logger.info("Cleanup completed")

    def run(self):
        """Run the assistant"""
        try:
            # Show welcome message
            self.ui.add_message("J.A.R.V.I.S Online. At your service, sir.", "jarvis")
            self.speak("Jarvis Online. At your service, sir.")

            # Start main loop
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def initialize_audio(self):
        """Initialize audio system"""
        try:
            if pygame.mixer.get_init() is not None:
                pygame.mixer.quit()

            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            return True
        except Exception as e:
            self.logger.error(f"Audio initialization error: {str(e)}")
            return False


# Main execution
if __name__ == "__main__":
    try:
        ensure_directories()

        print(
            """
    ╔══════════════════════════════════════════╗
    ║                J.A.R.V.I.S               ║
    ║        Personal AI Assistant v1.0        ║
    ╚══════════════════════════════════════════╝
        """
        )

        # Initialize and run Jarvis
        jarvis = Jarvis()
        jarvis.run()

    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}", exc_info=True)
        print(f"Critical Error: {str(e)}")

import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import requests
from googletrans import Translator
import sys

# Claude AI API Key
CLAUDE_API_KEY = "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"


# Function to interact with Claude AI
def ask_claude(prompt):
    """Send user queries to Claude AI and get a response."""
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "Authorization": f"Bearer {CLAUDE_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": f"{prompt}\n\n",
        "model": "claude-v1",
        "max_tokens_to_sample": 300,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["completion"].strip()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "క్షమించండి, నేను ప్రాసెస్ చేయలేకపోయాను."  # Telugu for error


# Translator setup
translator = Translator()


def translate_to_telugu(text):
    """Translate text to Telugu."""
    return translator.translate(text, dest="te").text


def translate_to_english(text):
    """Translate text to English."""
    return translator.translate(text, dest="en").text


def speak_telugu(text, file_name="response.mp3"):
    """Convert Telugu text to speech using gTTS."""
    tts = gTTS(text=text, lang="te")
    tts.save(file_name)
    playsound.playsound(file_name)
    os.remove(file_name)


class VirtualAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.language = "te"  # Default language is Telugu
        self.assistant_name = "Jarvis"

    def speak(self, text):
        """Speak the text in Telugu."""
        if self.language == "te":
            print(f"Speaking (Telugu): {text}")
            speak_telugu(text)
        else:
            print(f"Speaking (English): {text}")
            speak_telugu(text, lang="en")  # Can be modified for English if needed

    def listen(self):
        """Listen to user input and return text."""
        with sr.Microphone() as source:
            print("Listening...")
            self.speak("మీ ఆదేశం చెప్పండి.")  # Telugu for "Please say your command."
            audio = self.recognizer.listen(source)
            try:
                if self.language == "te":
                    text = self.recognizer.recognize_google(audio, language="te-IN")
                    print(f"You said (Telugu): {text}")
                else:
                    text = self.recognizer.recognize_google(audio, language="en-IN")
                    print(f"You said (English): {text}")

                if self.language == "te":
                    text = translate_to_english(
                        text
                    )  # Translate Telugu input to English for processing
                return text.lower()
            except sr.UnknownValueError:
                self.speak(
                    "క్షమించండి, నేను పట్టుకోలేకపోయాను. దయచేసి మళ్లీ చెప్పండి."
                )  # Telugu for "Sorry, I couldn't catch that."
                return ""
            except sr.RequestError:
                self.speak(
                    "క్షమించండి, మైక్ లేదా సేవలో సమస్య ఉంది."
                )  # Telugu for "Sorry, there's an issue with the mic or service."
                return ""

    def process_command(self, command):
        """Process user commands using Claude AI."""
        if "exit" in command or "bye" in command:
            self.speak(
                "వీడ్కోలు! మీతో మాట్లాడడం ఆనందంగా ఉంది."
            )  # Telugu for "Goodbye! It was nice talking to you."
            sys.exit()

        # Get response from Claude AI
        response = ask_claude(command)
        print(f"Claude AI Response: {response}")

        # Translate response to Telugu and speak
        if self.language == "te":
            response = translate_to_telugu(response)
        self.speak(response)

    def run(self):
        """Main loop of the assistant."""
        self.speak(
            f"హలో! నేను {self.assistant_name}. నేను మీకు ఎలా సహాయం చేయగలను?"
        )  # Telugu for "Hello! I am Jarvis. How can I assist you?"
        while True:
            command = self.listen()
            if command:
                self.process_command(command)


if __name__ == "__main__":
    assistant = VirtualAssistant()
    assistant.run()

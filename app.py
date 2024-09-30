import os
from dotenv import load_dotenv
import tkinter as tk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import threading
import tempfile
import time
import whisper
from gtts import gTTS
import pygame
import warnings
import openai

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
print(api_key)  # For testing, remove in production

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Whisper model
whisper_model = whisper.load_model("base")

# Initialize global variables
recording = False
recording_start_time = 0
audio_data = []
recording_lock = threading.Lock()

# Function to update the timer during recording
def update_timer():
    while recording:
        elapsed_time = int(time.time() - recording_start_time)
        timer_label.config(text=f"Recording... {elapsed_time} sec")
        time.sleep(1)

# Function to start recording
def start_recording():
    global recording, recording_start_time, audio_data
    fs = 44100  # Sample rate
    recording = True
    recording_start_time = time.time()
    audio_data = []

    start_record_button.config(state=tk.DISABLED)
    stop_record_button.config(state=tk.NORMAL)
    text_entry.delete("1.0", tk.END)
    text_entry.insert(tk.END, "Recording started...")

    # Start timer in a separate thread
    timer_thread = threading.Thread(target=update_timer)
    timer_thread.start()

    # Record audio in chunks until stopped
    def record_audio():
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
            while recording:
                audio_data.append(stream.read(1024)[0])  # Read small chunks of audio

    threading.Thread(target=record_audio).start()

# Function to stop recording
def stop_recording():
    global recording
    recording = False
    sd.stop()

    start_record_button.config(state=tk.NORMAL)
    stop_record_button.config(state=tk.DISABLED)
    timer_label.config(text="Recording finished")

    # Combine all audio chunks and save as a single file
    audio_data_combined = np.concatenate(audio_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        audio_file = temp_audio_file.name
        scipy.io.wavfile.write(audio_file, 44100, audio_data_combined)

    # Transcribe the audio directly
    transcribe_audio(audio_file)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    try:
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        result = whisper_model.transcribe(audio)
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, result['text'])
    except Exception as e:
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, f"Error: {str(e)}")

# Function to convert text to speech
def text_to_speech():
    text = text_entry.get("1.0", tk.END).strip()  # Get text from the Text widget
    if text:
        try:
            tts = gTTS(text=text, lang='en')
            audio_file = 'output.mp3'
            tts.save(audio_file)

            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait until the music finishes playing
            while pygame.mixer.music.get_busy():
                pass

            pygame.mixer.music.stop()  # Stop the mixer
            pygame.mixer.quit()  # Quit the mixer to release resources
            os.remove(audio_file)  # Delete the audio file after playing
        except Exception as e:
            text_entry.delete("1.0", tk.END)
            text_entry.insert(tk.END, f"Error: {str(e)}")

# Function to clear the text and reset the UI
def clear_text():
    text_entry.delete("1.0", tk.END)
    timer_label.config(text="")

# Function to get a response from OpenAI
def get_openai_response():
    user_input = text_entry.get("1.0", tk.END).strip()
    if user_input:
        try:
            # Use the new API call format
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Specify the model
                messages=[{"role": "user", "content": user_input}],
                api_key=api_key  # Use the API key here
            )
            # Extract the assistant's reply from the response
            assistant_response = response['choices'][0]['message']['content']
            text_entry.delete("1.0", tk.END)
            text_entry.insert(tk.END, assistant_response)
        except Exception as e:
            text_entry.delete("1.0", tk.END)
            text_entry.insert(tk.END, f"Error: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Speech to Text and Text to Speech App")
root.geometry("500x600")  # Set window size
root.configure(bg="#f0f0f0")  # Set background color

# Create and place the input text box
text_entry = tk.Text(root, height=10, width=50, font=("Arial", 14), bg="white")
text_entry.pack(pady=20)

# Bind mouse actions for cut, copy, and paste
def cut_text(event=None):
    text_entry.event_generate("<<Cut>>")

def copy_text(event=None):
    text_entry.event_generate("<<Copy>>")

def paste_text(event=None):
    text_entry.event_generate("<<Paste>>")

text_entry.bind("<Control-x>", cut_text)  # Cut with Ctrl+X
text_entry.bind("<Control-c>", copy_text)  # Copy with Ctrl+C
text_entry.bind("<Control-v>", paste_text)  # Paste with Ctrl+V

# Create a context menu
def show_context_menu(event):
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="Cut", command=cut_text)
    context_menu.add_command(label="Copy", command=copy_text)
    context_menu.add_command(label="Paste", command=paste_text)
    context_menu.post(event.x_root, event.y_root)

text_entry.bind("<Button-3>", show_context_menu)  # Right-click to show context menu

# Create a frame to hold the buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Function to change button background on hover
def on_enter(button):
    button['bg'] = '#d3d3d3'  # Light gray

def on_leave(button):
    button['bg'] = '#f0f0f0'  # Default light grey

# Create and place the Start Recording button
start_record_button = tk.Button(button_frame, text="Start Recording", command=start_recording, bg="#f0f0f0", fg="black", font=("Arial", 14))
start_record_button.pack(pady=5)
start_record_button.bind("<Enter>", lambda e: on_enter(start_record_button))
start_record_button.bind("<Leave>", lambda e: on_leave(start_record_button))

# Create and place the Stop Recording button (disabled initially)
stop_record_button = tk.Button(button_frame, text="Stop Recording", command=stop_recording, state=tk.DISABLED, bg="#f0f0f0", fg="black", font=("Arial", 14))
stop_record_button.pack(pady=5)
stop_record_button.bind("<Enter>", lambda e: on_enter(stop_record_button))
stop_record_button.bind("<Leave>", lambda e: on_leave(stop_record_button))

# Create and place the Text-to-Speech button
text_to_speech_button = tk.Button(button_frame, text="Text to Speech", command=text_to_speech, bg="#f0f0f0", fg="black", font=("Arial", 14))
text_to_speech_button.pack(pady=5)
text_to_speech_button.bind("<Enter>", lambda e: on_enter(text_to_speech_button))
text_to_speech_button.bind("<Leave>", lambda e: on_leave(text_to_speech_button))

# Create and place the Clear button
clear_button = tk.Button(button_frame, text="Clear", command=clear_text, bg="#f0f0f0", fg="black", font=("Arial", 14))
clear_button.pack(pady=5)
clear_button.bind("<Enter>", lambda e: on_enter(clear_button))
clear_button.bind("<Leave>", lambda e: on_leave(clear_button))

# Create and place the Get Response button
get_response_button = tk.Button(button_frame, text="Get Response", command=get_openai_response, bg="#f0f0f0", fg="black", font=("Arial", 14))
get_response_button.pack(pady=5)
get_response_button.bind("<Enter>", lambda e: on_enter(get_response_button))
get_response_button.bind("<Leave>", lambda e: on_leave(get_response_button))

# Create and place the timer label
timer_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 14))
timer_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

from pydub import AudioSegment
import io
import numpy as np
from utils import load_config
import librosa
from transformers import pipeline
config = load_config()

def convert_bytes_to_array(audio_bytes):
    # Use BytesIO to handle the in-memory audio bytes
    audio_bytes = io.BytesIO(audio_bytes)
    
    # Load the audio using pydub (handles multiple formats like mp3, wav, etc.)
    audio = AudioSegment.from_file(audio_bytes)
    
    # Convert to numpy array and normalize to float32 range (-1.0 to 1.0)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 2**15
    sample_rate = audio.frame_rate
    
    return audio_array, sample_rate

def transcribe_audio(audio_bytes):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # Initialize the Whisper model pipeline for automatic speech recognition
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=config["whisper_model"],
        chunk_length_s=30,
        device=device,
    )

    # Convert the audio bytes to an array and get the sample rate
    audio_array, sample_rate = convert_bytes_to_array(audio_bytes)
    
    # Whisper models typically work with 16kHz audio, resample if needed
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    
    # Make predictions using the model
    prediction = pipe(audio_array, batch_size=1)["text"]

    return prediction

# from transformers import pipeline
# import librosa
# import io
# from utils import load_config
# config = load_config()

# def convert_bytes_to_array(audio_bytes):
#     audio_bytes = io.BytesIO(audio_bytes)
#     audio, sample_rate = librosa.load(audio_bytes)
#     print(sample_rate)
#     return audio

# def transcribe_audio(audio_bytes):
#     #device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     pipe = pipeline(
#         task="automatic-speech-recognition",
#         model=config["whisper_model"],
#         chunk_length_s=30,
#         device=device,
#     )   

#     audio_array = convert_bytes_to_array(audio_bytes)
#     prediction = pipe(audio_array, batch_size=1)["text"]

#     return prediction

import os
import torch
import torchaudio
import numpy as np
import time
import shutil
from datetime import datetime
from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode
from hubert.customtokenizer import CustomTokenizer

BASE_DIR = '/content/voiceclone'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
COMPLETED_DIR = os.path.join(BASE_DIR, 'completed')
INPUT_DIR = os.path.join(BASE_DIR, 'input')

def clone_voice(audio_filepath, text_prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save the features to an .npz file
    filename_without_extension = os.path.splitext(os.path.basename(audio_filepath))[0]
    output_path = os.path.join(BASE_DIR, 'bark', 'assets', 'prompts', filename_without_extension + '.npz')
    
    # Check if the .npz file already exists
    if not os.path.exists(output_path):
        print("Creating NPZ File")
        # Load models
        model = load_codec_model(use_gpu=True if device == 'cuda' else False)
        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed()
        hubert_model = CustomHubert(checkpoint_path=os.path.join(BASE_DIR, 'data/models/hubert/hubert.pt')).to(device)
        tokenizer = CustomTokenizer.load_from_checkpoint(os.path.join(BASE_DIR, 'data/models/hubert/tokenizer.pth'), map_location=torch.device('cpu')).to(device)

        # Process audio sample
        wav, sr = torchaudio.load(audio_filepath)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.to(device)
        semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

        # Move codes and semantic tokens to CPU
        codes = codes.cpu().numpy()
        semantic_tokens = semantic_tokens.cpu().numpy()
        np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    else:
        print("Using existing NPZ file")
    preload_models(text_use_gpu=True, coarse_use_gpu=True, fine_use_gpu=True, codec_use_gpu=True, force_reload=False, path=os.path.join(BASE_DIR, "models"))

    # Use the .npz file as the history_prompt
    voice_name = output_path
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    
    return audio_array

if __name__ == "__main__":
    # Ensure the directories exist
    for directory in [OUTPUT_DIR, COMPLETED_DIR, INPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)



    for filename in os.listdir(BASE_DIR):  # Search in the base directory
        if filename.endswith('.txt'):
            print("found text to generate")
            print(filename)
            with open(os.path.join(BASE_DIR, filename), 'r') as file:
                text_to_speak = file.read().replace('\n', ' ')  # Load the content and strip line returns
            # Set the input wav file based on the txt file's name up to the first hyphen
            audio_sample_base = filename.split('-')[0]
            audio_sample_path = os.path.join(INPUT_DIR, audio_sample_base + '.wav')
            print(audio_sample_path)
            print(text_to_speak)
            cloned_audio = clone_voice(audio_sample_path, text_to_speak)

            # Save the cloned audio with the same name as the txt file but with .wav extension
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            print(output_path)
            from scipy.io.wavfile import write as write_wav
            write_wav(output_path, SAMPLE_RATE, cloned_audio)
                
            time.sleep(5)
            # Move the processed txt file to the completed directory
            try:
                shutil.move(os.path.join(BASE_DIR, filename), os.path.join(COMPLETED_DIR, filename))
            except Exception as e:
                print(f"Error moving {os.path.join(BASE_DIR, filename)} to {os.path.join(COMPLETED_DIR, filename)}. Error: {e}")

            # If no txt files are found in the script directory, look in the Q:\ directory


        time.sleep(5)  # Wait for 5 seconds before searching again

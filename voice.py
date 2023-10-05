import torch
import torchaudio
import numpy as np
import os
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

def clone_voice(audio_filepath, text_prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save the features to an .npz file
    filename_without_extension = os.path.splitext(os.path.basename(audio_filepath))[0]
    output_path = os.path.join('bark', 'assets', 'prompts', filename_without_extension + '.npz')
    print(output_path)
    
    # Check if the .npz file already exists
    if not os.path.exists(output_path):
        print("Creeating NPZ File")
        # Load models
        model = load_codec_model(use_gpu=True if device == 'cuda' else False)
        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed()
        hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)
        tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth', map_location=torch.device('cpu')).to(device)

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
    preload_models(text_use_gpu=True, coarse_use_gpu=True, fine_use_gpu=True, codec_use_gpu=True, force_reload=False, path="models")

    # Use the .npz file as the history_prompt
    voice_name = output_path
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    
    return audio_array


if __name__ == "__main__":
    output_directory = 'output'
    completed_dir = 'completed'
    input_dir = 'input'

    # Ensure the completed directory exists
    if not os.path.exists(completed_dir):
        os.makedirs(completed_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    while True:  # Infinite loop to keep searching for txt files
        k=0
        for filename in os.listdir('.'):  # Search in the current directory
            if filename.endswith('.txt'):
                print("found text to generate")
                print(filename)
                with open(filename, 'r') as file:
                    text_to_speak = file.read().replace('\n', ' ')  # Load the content and strip line returns
                print(k)
                # Set the input wav file based on the txt file's name up to the first hyphen
                audio_sample_base = filename.split('-')[0]
                audio_sample_path = os.path.join('input', audio_sample_base + '.wav')
                print(audio_sample_path)
                print(text_to_speak)
                cloned_audio = clone_voice(audio_sample_path, text_to_speak)

                # Save the cloned audio with the same name as the txt file but with .wav extension
                output_filename = os.path.splitext(filename)[0] + '.wav'
                output_path = os.path.join(output_directory, output_filename)
                print(output_path)
                from scipy.io.wavfile import write as write_wav
                write_wav(output_path, SAMPLE_RATE, cloned_audio)
                
                time.sleep(5)
                # Move the processed txt file to the completed directory
                #shutil.move(filename, os.path.join(completed_dir, filename))
                try:
                    shutil.move(filename, os.path.join(completed_dir, filename))
                except Exception as e:
                    print(f"Error moving {source_path} to {destination_path}. Error: {e}")
                

                # If no txt files are found in the script directory, look in the Q:\ directory
            k=k+1
        print("restarting")
        time.sleep(5)  # Wait for 5 seconds before searching again

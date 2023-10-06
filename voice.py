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
import re

BASE_DIR = '/content/voiceclone'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
COMPLETED_DIR = os.path.join(BASE_DIR, 'completed')
INPUT_DIR = os.path.join(BASE_DIR, 'input')

def split_and_recombine_text(text, desired_length=100, max_length=150):
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                d = pos - split_pos[-1]
                seek(-d)
            else:
                while c not in "!?.\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]
    return rv

def clone_voice(audio_filepath, text_prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filename_without_extension = os.path.splitext(os.path.basename(audio_filepath))[0]
    output_path = os.path.join(BASE_DIR, 'bark', 'assets', 'prompts', filename_without_extension + '.npz')
    
    if not os.path.exists(output_path):
        print("Creating NPZ File")
        model = load_codec_model(use_gpu=True if device == 'cuda' else False)
        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed()
        hubert_model = CustomHubert(checkpoint_path=os.path.join(BASE_DIR, 'data/models/hubert/hubert.pt')).to(device)
        tokenizer = CustomTokenizer.load_from_checkpoint(os.path.join(BASE_DIR, 'data/models/hubert/tokenizer.pth'), map_location=torch.device('cpu')).to(device)

        wav, sr = torchaudio.load(audio_filepath)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.to(device)
        semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        with torch.no_grad():
            encoded_frames = model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
        codes = codes.cpu().numpy()
        semantic_tokens = semantic_tokens.cpu().numpy()
        np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    else:
        print("Using existing NPZ file")

    preload_models(text_use_gpu=True, coarse_use_gpu=True, fine_use_gpu=True, codec_use_gpu=True, force_reload=False, path=os.path.join(BASE_DIR, "models"))
    voice_name = output_path
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    return audio_array

if __name__ == "__main__":
    for directory in [OUTPUT_DIR, COMPLETED_DIR, INPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for filename in os.listdir(BASE_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(BASE_DIR, filename), 'r') as file:
                text_to_speak = file.read().replace('\n', ' ')
            audio_sample_base = filename.split('-')[0]
            audio_sample_path = os.path.join(INPUT_DIR, audio_sample_base + '.wav')

            texts = split_and_recombine_text(text_to_speak)
            all_parts = []
            for text_chunk in texts:
                audio_chunk = clone_voice(audio_sample_path, text_chunk)
                all_parts.append(audio_chunk)

            cloned_audio = np.concatenate(all_parts, axis=-1)
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            from scipy.io.wavfile import write as write_wav
            write_wav(output_path, SAMPLE_RATE, cloned_audio)
                
            time.sleep(5)
            try:
                shutil.move(os.path.join(BASE_DIR, filename), os.path.join(COMPLETED_DIR, filename))
            except Exception as e:
                print(f"Error moving {os.path.join(BASE_DIR, filename)} to {os.path.join(COMPLETED_DIR, filename)}. Error: {e}")

            time.sleep(5)

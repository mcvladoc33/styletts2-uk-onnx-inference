# --- ПОВНЕ ПРИДУШЕННЯ ВСЬОГО ---
import sys, os

# Радикальне придушення на рівні системи до будь-яких імпортів
# Перенаправляємо stderr у "діру", щоб вбити будь-який шум
sys.stderr = open(os.devnull, 'w')

# Змінні оточення для мовчання бібліотек
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_DOWNLOADS_WARNING'] = '1'

import torch

# Повертаємо stderr назад, щоб ми бачили наші власні принти та помилки
sys.stderr = sys.__stdout__

import warnings
import logging
import numpy as np
import onnxruntime as ort
import sounddevice as sd
import builtins
import time
from styletts2_inference.models import StyleTTS2Tokenizer
from ukrainian_word_stress import Stressifier
from ipa_uk import ipa as uk_to_ipa

# Придушення внутрішніх попереджень Python
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Патч кодування Windows для кирилиці
original_open = builtins.open


def utf8_open(*args, **kwargs):
    if len(args) > 1 and 'b' in args[1]: return original_open(*args, **kwargs)
    if 'mode' in kwargs and 'b' in kwargs['mode']: return original_open(*args, **kwargs)
    kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


builtins.open = utf8_open

# --- ІНІЦІАЛІЗАЦІЯ ---
stressifier = Stressifier()
tokenizer = StyleTTS2Tokenizer(hf_path="patriotyk/styletts2_ukrainian_multispeaker_hifigan")
session = ort.InferenceSession("models/styletts2.onnx")

# Завантаження вектора стилю
style_vector = torch.load("./voices/Інна Гелевера.pt", map_location='cpu').detach().numpy()
if len(style_vector.shape) == 1:
    style_vector = np.expand_dims(style_vector, axis=0)


def generate_and_play(text):
    text_stressed = stressifier(text)
    text_ipa = uk_to_ipa(text_stressed)

    start_time = time.time()

    tokens = tokenizer.encode(text_ipa)
    tokens_onnx = np.concatenate([[0], tokens.numpy()]).astype(np.int64)

    inputs = {
        'tokens': tokens_onnx,
        'speed': np.array(1.0, dtype=np.float32),
        's_prev': style_vector.astype(np.float32)
    }

    wav = session.run(None, inputs)[0].flatten()
    wav = np.clip(wav, -0.99, 0.99)
    if np.max(np.abs(wav)) > 0: wav = wav / np.max(np.abs(wav))

    print(f"✅ Готово! (Час: {time.time() - start_time:.3f} сек)")
    sd.play(wav.astype(np.float32), 24000)
    sd.wait()


if __name__ == "__main__":
    print("\n🚀 Система готова. Введіть текст (або 'exit' для виходу):")
    while True:
        try:
            user_input = input("\n>>> ")
            if user_input.lower() == 'exit':
                print("👋 Завершення роботи...")
                break
            generate_and_play(user_input)
        except KeyboardInterrupt:
            print("\n👋 Завершення роботи...")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")
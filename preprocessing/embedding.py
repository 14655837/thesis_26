import os
import clip
import torch
import open_clip
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from transformers import ClapModel, ClapProcessor
import librosa
import duckdb
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Open the model for embedding images, so the function that embedds images can be called one for one
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def open_csv(path):
    """Opens the csv of the give path"""
    df = pd.read_csv(path)
    return df

def my_id_to_car_id(df):
    num_rows = len(df.index)
    df['my_id'] = [get_id_from_path_as_str(df['image_path'][i]) for i in range(num_rows)]
    new_df = df[['my_id', 'car_id']]
    return new_df

def windowspath_to_relative(windows_path):
    parts = windows_path.parts  # e.g. ('C:\\', 'Users', ..., 'files', 'cars', 'data', 'all_car_images', 'img.jpg')
    cutoff = parts.index("files")
    return "/".join(parts[cutoff:])

def rewrite_dict_path_to_str(my_dict):
    return {windowspath_to_relative(k): v for k, v in my_dict.items()}

def path_to_car_id_in_df(df, vector_dict):
    num_rows = len(df.index)
    df['vectors'] = [vector_dict[df['image_path'][i]] for i in range(num_rows)]
    new_df = df[['car_id', 'vectors']]
    return new_df

def path_to_image_id_in_df(df, vector_dict):
    num_rows = len(df.index)
    df['vectors'] = [vector_dict[df['image_path'][i]] for i in range(num_rows)]
    new_df = df[['image_id', 'vectors']]
    return new_df

def is_my_id_good(df) -> None:
    inconsistent = (
        df.groupby("my_id")["car_id"]
        .nunique()
        .loc[lambda x: x > 1]
    )

    if inconsistent.empty:
        print("All good: every my_id maps to exactly one car_id.")
    else:
        print(f"Inconsistent my_ids:\n{inconsistent}")

def get_id_from_path(image_path):
    filename = Path(image_path).stem  # e.g. "178_MWLH7147_03012020_091505image270933"
    return filename.split("_")[0] 

def get_id_from_path_as_str(image_path: str) -> int:
    filename = image_path.split("/")[-1]  # e.g. "0_YKMS3041_01012020_172204image853193.jpg"
    return int(filename.split("_")[0])         # e.g. "0"

def embed_text(text: str):
    tokens = tokenizer([text])  # tokenizer expects a list
    with torch.no_grad():
        return model.encode_text(tokens).squeeze().numpy()

def embed_image(image_path):
    try:
        image = preprocess_val(Image.open(image_path)).unsqueeze(0)
        with torch.no_grad():
            return model.encode_image(image).squeeze().numpy()
    except Exception as e:
        print(f"Skipping {image_path.name}: {e}")
        return None
    
def embed_audio(audio_dir) -> dict:
    model_audio = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor_audio = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    all_embeddings = dict()
    for path in audio_dir:
        audio_array, _ = librosa.load(path, sr=48000, mono=True)
        inputs = processor_audio(audio=audio_array, return_tensors="pt", sampling_rate=48000)
        audio_embed = model_audio.get_audio_features(**inputs)
        vector = audio_embed.pooler_output.squeeze().detach().numpy()  # shape: (512,)
        all_embeddings[path] = vector
    return all_embeddings
    
def avg_vectors(vectors):
    all_vectors_single = {
    id_: np.mean(vecs, axis=0)
        for id_, vecs in vectors.items()
    }
    return all_vectors_single

def embed_images(image_dir) -> None:
    all_vectors = defaultdict(list)
    for path in image_dir:
        all_vectors[get_id_from_path(path)].append(embed_image(path))
    sorted_vectors = dict(sorted(all_vectors.items(), key=lambda x: int(x[0])))
    sorted_vectors = avg_vectors(sorted_vectors)

def create_dict_car_id_vec(df, vectors) -> dict:
    new_df = path_to_car_id_in_df(df, vectors)
    num_rows = len(new_df.index)
    return {new_df['car_id'][i]: new_df['vectors'][i] for i in range(num_rows)}

def create_dict_image_id_vec(df, vectors) -> dict:
    new_df = path_to_image_id_in_df(df, vectors)
    num_rows = len(new_df.index)
    return {new_df['image_id'][i]: new_df['vectors'][i] for i in range(num_rows)}

def dump_new_image_id_vec_dict(df, vectors) -> None:
    new_dict = create_dict_image_id_vec(df, vectors)
    with open("cars_image_id_vec_dict.pkl", "wb") as f:
        pickle.dump(new_dict, f)

def dump_new_car_id_vec_dict(df, vectors) -> None:
    new_dict = create_dict_car_id_vec(df, vectors)
    with open("cars_id_vec_dict.pkl", "wb") as f:
        pickle.dump(new_dict, f)

def save_pickle(data, name: str) -> None:
    if ".pkl" not in name:
        print("Wrong input with name, not ending at .pkl")
        return

    with open(name, "wb") as f:
        pickle.dump(data, f)

def create_audio_dir(audio_path):
    return list(audio_path.glob("*.wav"))

def create_image_dir(image_path) -> list:
    return list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")) + list(image_path.glob("*.jpeg"))

def path_to_audio_id_in_df(df, vector_dict):
    num_rows = len(df.index)
    df['vectors'] = [vector_dict[df['audio_path'][i]] for i in range(num_rows)]
    new_df = df[['audio_id', 'vectors']]
    return new_df

def create_dict_audio_id_vec(df, vectors) -> dict:
    new_df = path_to_audio_id_in_df(df, vectors)
    num_rows = len(new_df.index)
    return {new_df['audio_id'][i]: new_df['vectors'][i] for i in range(num_rows)}

if __name__ == '__main__':
    print("Running embedding.py!")
    print("______________________")
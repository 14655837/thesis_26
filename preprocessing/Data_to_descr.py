from litellm import completion
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import base64
import os
import pickle
import duckdb

def create_image_dir(image_path) -> list:
    """Creates a list of all the images in a folder, through an imput of a path to a folder with images"""
    return list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")) + list(image_path.glob("*.jpeg"))

def create_audio_dir(audio_path):
    """Creates a list of all the audio files in a folder, through an imput of a path to a folder with audio"""
    return list(audio_path.glob("*.wav"))

def encode_data(data_path):
    """Encode images and audio files so they can be processed by an LLM"""
    with open(data_path, "rb") as f:
        encoded_data = base64.b64encode(f.read()).decode("utf-8")
    return encoded_data

def create_description(data_input: dict, type_of_data: str) -> list:
    """
    Function to create descriptions of all the data in the data_input, which should
    be audio or image data.

    data_input is suppossed to be a dict with paths of all the data files that
        need to be desrcibed in text form. With the is as key and the path as item.
    type_of_data is supposed to be a string that says what type of files they are
        so it can be put in the message e.g. image or audio
    """
    if type_of_data not in ["audio", "image"]:
        print("Wrong type_of_data")
        return []

    message = (
        f"Describe this {type_of_data} in exactly 75 words. "
        "Focus on all visible/audible details that could be relevant for filtering or classification. "
        "Return only the description, nothing else."
    )
    
    if type_of_data == "audio":
        model = "gpt-audio"
    else:
        model = "gpt-5-mini"

    output_dict = dict()
    for id, path in data_input.items():
        data = encode_data(path)
        if type_of_data == "audio":
            media_block = {
                "type": "input_audio",
                "input_audio": {"data": data, "format": "wav"}
            }
        else:
            media_block = {
                "type": "image_url",
                "image_url": {"url": f"data:{"image/jpeg"};base64,{data}"}
            }

        response = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    media_block,
                    {"type": "text", "text": message}
                ]
            }]
        )
        output = response.choices[0].message.content.strip()
        if output is None:
            print("Something went wrong")
        output_dict[id] = output
    
    return output_dict

def save_data(data, save_path: str) -> None:
    """Saves the data in a pickle file"""
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    
def filter_data_by_db(images: list[Path], db_path: str, table_name: str, col_img: str, col_id) -> dict:
    """Return only images whose filename appears in the database column."""
    
    con = duckdb.connect(db_path)
    
    # Get all filenames from the db column (strip to just the filename)
    df = con.execute(f"SELECT {col_img}, {col_id} FROM {table_name}").fetchdf()
    
    # Select only the part of the file name, so everything behind the last slash
    df[col_img] = df[col_img].str.split("/").str[-1]
    filename_to_id = dict(zip(df[col_img], df[col_id]))

    print(len(filename_to_id) == len(df))
    con.close()

    return {filename_to_id[file_path.name]: file_path for file_path in images if file_path.name in filename_to_id}

def add_descr_to_duckdb(descr_dict, duckdb_file, table_name: str):
    """Adds the created LLM description to a duckdb database"""
    con = duckdb.connect(duckdb_file)

    con.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS description VARCHAR")

    # Update each row with its description
    for car_id, description in descr_dict.items():
        con.execute(
            f"UPDATE {table_name} SET description = ? WHERE car_id = ?",
            [description, car_id]
        )

    con.close()

    con = duckdb.connect(duckdb_file)
    df = con.execute(f"SELECT car_id, description FROM {table_name} LIMIT 5").fetchdf()
    print(df)

    # Check
    counts = con.execute(f"""
        SELECT 
            COUNT(*) as total,
            COUNT(description) as filled,
            COUNT(*) - COUNT(description) as missing
        FROM {table_name}
    """).fetchdf()
    print(counts)
    con.close()

if __name__ == '__main__':
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    
    db_path = r"C:\Users\Mikev\study\scriptie\medical_database_tdb.duckdb"

    image_dir = Path(r"C:\Users\Mikev\study\scriptie\medical-20260627T215421Z-3-001\medical\raw_data\raw_data\all_skin_images")
    images = create_image_dir(image_dir)
    imgs = filter_data_by_db(images, db_path, "skin_images", "image_path", 'patient_id')
    descr_dict = create_description(imgs, "image")
    save_data(descr_dict, "all_skin_img_v2.pkl")
import getpass
import os
import sys
import time
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pickle

sql1 = "SELECT DISTINCT car_id FROM car_complaints WHERE NLfilter(summary, 'You are be given a textual complaint " \
"entailing that the car was in a crash/accident/collision. Complaint: {summary}.')"
sql2 = """SELECT DISTINCT cars.car_id
FROM cars, car_audio
WHERE cars.car_id = car_audio.car_id
AND cars.fuel_type = 'Electric'
AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the car from the recording has a dead battery, false otherwise.')
"""
sql3 = """SELECT cars.car_id
FROM cars, car_images
WHERE cars.car_id = car_images.car_id
AND cars.transmission = 'Manual'
AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is not damaged.')
LIMIT 10
"""

sql4 = """SELECT cars.car_id
FROM cars, car_complaints
WHERE cars.car_id = car_complaints.car_id
AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with engine / connected to engine. Complaint: {summary}.')
"""

sql4_l10 = """SELECT cars.car_id
FROM cars, car_complaints
WHERE cars.car_id = car_complaints.car_id
AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with engine / connected to engine. Complaint: {summary}.')
LIMIT 10
"""

sql5 = """SELECT DISTINCT cars.car_id
FROM cars, car_images, car_audio
WHERE cars.car_id = car_images.car_id
AND cars.car_id = car_audio.car_id
AND cars.transmission = 'Automatic'
AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the recording captures an audio of a damaged car.')
AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is damaged.')
LIMIT 100
"""

sql6 = """WITH sick_audio AS (
    SELECT car_audio.car_id
    FROM cars, car_audio
    WHERE cars.car_id = car_audio.car_id
    AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the recording captures an audio of a damaged car.')
),
sick_image AS (
    SELECT car_images.car_id
    FROM cars, car_images
    WHERE cars.car_id = car_images.car_id
    AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is damaged.')
),
sick_text AS (
    SELECT car_complaints.car_id
    FROM cars, car_complaints
    WHERE cars.car_id = car_complaints.car_id
    AND NLfilter(car_complaints.summary, 'You are be given a textual complaint entailing that the car was in on fire or burned. Complaint: {summary}.')
),
combined AS (
    SELECT DISTINCT
        cars.car_id,
        cars.year,
        car_complaints.complaint_id,
        car_images.image_id,
        car_audio.audio_id,
        CASE WHEN a.car_id IS NOT NULL THEN 1 WHEN car_audio.audio_id IS NOT NULL THEN 0 ELSE NULL END AS is_sick_audio,
        CASE WHEN s.car_id IS NOT NULL THEN 1 WHEN car_complaints.complaint_id IS NOT NULL THEN 0 ELSE NULL END AS is_sick_text,
        CASE WHEN x.car_id IS NOT NULL THEN 1 WHEN car_images.image_id IS NOT NULL THEN 0 ELSE NULL END AS is_sick_image
    FROM cars
    LEFT JOIN car_complaints ON cars.car_id = car_complaints.car_id
    LEFT JOIN car_images ON cars.car_id = car_images.car_id
    LEFT JOIN car_audio ON cars.car_id = car_audio.car_id
    LEFT JOIN sick_audio a ON cars.car_id = a.car_id
    LEFT JOIN sick_text s ON cars.car_id = s.car_id
    LEFT JOIN sick_image x ON cars.car_id = x.car_id
)
SELECT * FROM combined AS subq
WHERE (is_sick_audio = 1 OR is_sick_text = 1 OR is_sick_image = 1)
AND (is_sick_audio = 0 OR is_sick_text = 0 OR is_sick_image = 0)"""
sql7 = """WITH sick_audio AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_audio
    WHERE cars.car_id = car_audio.car_id
    AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes.')
),
sick_text AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_complaints
    WHERE cars.car_id = car_complaints.car_id
    AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint: {summary}.')
),
sick_image AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_images
    WHERE cars.car_id = car_images.car_id
    AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is dented.')
)
SELECT car_id FROM sick_audio
UNION DISTINCT
SELECT car_id FROM sick_text
UNION DISTINCT
SELECT car_id FROM sick_image
"""
sql7_l10 = """WITH sick_audio AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_audio
    WHERE cars.car_id = car_audio.car_id
    AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes.')
),
sick_text AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_complaints
    WHERE cars.car_id = car_complaints.car_id
    AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint: {summary}.')
),
sick_image AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_images
    WHERE cars.car_id = car_images.car_id
    AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is dented.')
)
SELECT car_id FROM sick_audio
UNION DISTINCT
SELECT car_id FROM sick_text
UNION DISTINCT
SELECT car_id FROM sick_image
LIMIT 10
"""
sql7_l25 = """WITH sick_audio AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_audio
    WHERE cars.car_id = car_audio.car_id
    AND NLfilter(car_audio.audio_path, 'You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes.')
),
sick_text AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_complaints
    WHERE cars.car_id = car_complaints.car_id
    AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint: {summary}.')
),
sick_image AS (
    SELECT DISTINCT cars.car_id
    FROM cars, car_images
    WHERE cars.car_id = car_images.car_id
    AND NLfilter(car_images.image_path, 'You are given an image of a vehicle or its parts. Return true if car is dented.')
)
SELECT car_id FROM sick_audio
UNION DISTINCT
SELECT car_id FROM sick_text
UNION DISTINCT
SELECT car_id FROM sick_image
LIMIT 25
"""
sql8 = """SELECT car_id
FROM car_images
WHERE NLfilter(image_path, 'You are given an image of a vehicle or its parts. Return true if car has both, puncture and paint scratches.')
LIMIT 100
"""
sql8_l10 = """SELECT car_id
FROM car_images
WHERE NLfilter(image_path, 'You are given an image of a vehicle or its parts. Return true if car has both, puncture and paint scratches.')
LIMIT 10
"""
sql9 = """SELECT DISTINCT cars.car_id
FROM cars, car_images, car_audio
WHERE cars.car_id = car_images.car_id
AND cars.car_id = car_audio.car_id
AND NLfilter(car_images.image_path, car_audio.audio_path, 'You are given an image of a vehicle and an audio recording of car diagnostics. Return true if car is torn according to image and has bad ignition according to audio.')
"""
# Limit 2 is added to the first sem_sql
sem_sql_ecomm = """SELECT
  id
FROM styles_details
WHERE true
  AND NLfilter(
    full_product_description,
    'The product is a backpack from Reebok'
  ) LIMIT 10;
"""
sem_sql_ecomm2 = """SELECT
  image_mapping.id as id
FROM styles_details
JOIN image_mapping on styles_details.id = image_mapping.id
WHERE true
  AND NLfilter(
    local_image_path,
    'The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver.'
  );
"""
sem_sql_ecomm7 = """SELECT
  p1.id || '-' || p2.id AS "id"
FROM styles_details p1, styles_details p2
WHERE true
  AND NLjoin(p1.full_product_description, p2.full_product_description, 'You will be given two product descriptions. Do both product descriptions describe products of the same category from the same brand, e.g., both are t-shirts from Adidas?')
  AND p1.price <= 500
  AND p2.price <= 500
;
"""
sem_sql_ecomm8 = """SELECT
  styles_details.id || '-' || image_mapping.id as id
FROM styles_details, image_mapping
WHERE true
  AND NLjoin(styles_details.full_product_description, image_mapping.local_image_path, 'The image fits the description')
  AND character_length(styles_details.productDescriptors.description.value) >= 3000
;
"""
sem_sql_ecomm9 = """SELECT
  img1.id || '-' || img2.id AS id
FROM styles_details s1, styles_details s2, image_mapping img1, image_mapping img2
WHERE true
  -- Pre-filter both image tables with conditions on styles_details
  AND s1.baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
  AND s1.colour1 = ''
  AND s1.colour2 = ''
  AND s1.price < 800
  AND s1.id = img1.id
  AND s2.baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
  AND s2.colour1 = ''
  AND s2.colour2 = ''
  AND s2.price < 800
  AND s2.id = img2.id
  -- Semantic join
  AND img1.id <> img2.id
  AND NLjoin(img1.local_image_path, img2.local_image_path,
     '''
     Determine whether both images display objects of the same category
     (e.g., both are shoes, both are bags, etc.) and whether these objects
     share the same dominant surface color. Disregard any logos, text, or
     printed graphics on the objects. There might be other objects in the
     images. Only focus on the main object. Base your comparison solely on
     object type and overall surface color.
     '''
)
;"""
medical_sql1 = "select patients.patient_id from patients, symptoms_texts " \
    "where patients.patient_id=symptoms_texts.patient_id and " \
    "NLfilter(symptoms_texts.symptoms, 'Patient has an allergy.')"

medical_sql2 = "select distinct patients.patient_id from patients, lung_audio " \
    "where patients.patient_id = lung_audio.patient_id and patients.smoking_history != 'Current' and " \
    "NLfilter(lung_audio.path, 'This audio recording of human lungs captures healthy lungs, without diseases.')"

medical_sql3 = "select patients.patient_id " \
    "from patients, x_ray_images " \
    "where patients.patient_id = x_ray_images.patient_id and patients.did_family_have_cancer = 1 and NLfilter(x_ray_images.image_path, " \
    "'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.') " \
    "limit 5"

medical_sql4 = "SELECT patients.patient_id " \
    "FROM patients " \
    "JOIN symptoms_texts " \
    "WHERE patients.patient_id = symptoms_texts.patient_id AND NLfilter(symptoms_texts.symptoms, 'Patient has a skin acne.')"

medical_sql5 = "SELECT DISTINCT patients.patient_id " \
    "FROM patients, lung_audio, x_ray_images " \
    "WHERE patients.patient_id = x_ray_images.patient_id AND patients.patient_id = lung_audio.patient_id AND patients.smoking_history = 'Current' AND " \
    "NLfilter(lung_audio.path,'This human lung audio recording captures an audio of sick lungs, with diseases.') AND " \
    "NLfilter(x_ray_images.image_path,'This X-ray image of human lungs shows that there are lung problems (considered sick/disease).') "

medical_sql6 = "WITH sick_audio AS( " \
    "SELECT two_more_modalities.patient_id as patient_id " \
    "FROM two_more_modalities " \
    "WHERE two_more_modalities.bell_audio_id IS NOT NULL AND " \
    "(NLfilter(two_more_modalities.extended_audio, 'This audio recording of human lungs captures sick lungs, with diseases.') OR " \
    "NLfilter(two_more_modalities.bell_audio, 'This audio recording of human lungs captures sick lungs, with diseases.') OR " \
    "NLfilter(two_more_modalities.diaphragm_audio, 'This audio recording of human lungs captures sick lungs, with diseases.')) " \
    "), " \
    "sick_image AS( " \
    "SELECT two_more_modalities.patient_id as patient_id " \
    "FROM two_more_modalities " \
    "WHERE two_more_modalities.xray_id IS NOT NULL AND NLfilter(two_more_modalities.image_path, " \
    "'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.') " \
    "), " \
    "sick_text AS ( " \
    "SELECT two_more_modalities.patient_id as patient_id " \
    "FROM two_more_modalities " \
    "WHERE two_more_modalities.symptom_id IS NOT NULL AND NLfilter(two_more_modalities.symptoms, 'This patient is sick.') " \
    ") " \
    "SELECT patient_id FROM ( " \
    "SELECT two_more_modalities.patient_id " \
    "FROM two_more_modalities " \
    "LEFT JOIN sick_audio ON two_more_modalities.patient_id = sick_audio.patient_id " \
    "LEFT JOIN sick_text ON two_more_modalities.patient_id = sick_text.patient_id " \
    "LEFT JOIN sick_image ON two_more_modalities.patient_id = sick_image.patient_id) " \
    "WHERE (is_sick_audio = 1 OR is_sick_text = 1 OR is_sick_image = 1) " \
    "AND " \
    "(is_sick_audio = 0 OR is_sick_text = 0 OR is_sick_image = 0)"

medical_sql7 = "WITH sick_audio AS ( " \
    "select distinct patients.patient_id from patients, lung_audio " \
    "where patients.patient_id = lung_audio.patient_id and patients.smoking_history != 'Current' and " \
    "NLfilter(lung_audio.path, 'This audio recording of human lungs captures sick lungs, with diseases.') " \
    "), " \
    "sick_text AS ( " \
    "select patients.patient_id from patients, symptoms_texts " \
    "where patients.patient_id=symptoms_texts.patient_id and " \
    "NLfilter(symptoms_texts.symptoms, 'This patient is sick.') " \
    "), " \
    "sick_image AS( " \
    "select patients.patient_id " \
    "from patients, x_ray_images " \
    "where patients.patient_id = x_ray_images.patient_id and NLfilter(x_ray_images.image_path, " \
    "'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.') " \
    "), " \
    "sick_cancer AS( " \
    "select patients.patient_id " \
    "from patients, skin_images " \
    "where patients.patient_id = skin_images.patient_id and NLfilter(skin_images.image_path, " \
    "'This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image.') " \
    ") " \
    "SELECT sick_audio.patient_id FROM sick_audio " \
    "UNION DISTINCT " \
    "SELECT sick_text.patient_id FROM sick_text " \
    "UNION DISTINCT " \
    "SELECT sick_image.patient_id FROM sick_image " \
    "UNION DISTINCT " \
    "SELECT sick_cancer.patient_id FROM sick_cancer"

medical_sql7_l10 = "WITH sick_audio AS ( " \
    "select distinct patients.patient_id from patients, lung_audio " \
    "where patients.patient_id = lung_audio.patient_id and patients.smoking_history != 'Current' and " \
    "NLfilter(lung_audio.path, 'This audio recording of human lungs captures sick lungs, with diseases.') " \
    "), " \
    "sick_text AS ( " \
    "select patients.patient_id from patients, symptoms_texts " \
    "where patients.patient_id=symptoms_texts.patient_id and " \
    "NLfilter(symptoms_texts.symptoms, 'This patient is sick.') " \
    "), " \
    "sick_image AS( " \
    "select patients.patient_id " \
    "from patients, x_ray_images " \
    "where patients.patient_id = x_ray_images.patient_id and NLfilter(x_ray_images.image_path, " \
    "'This X-ray image of human lungs shows that there are lung problems (considered sick/disease) according to the X-ray image.') " \
    "), " \
    "sick_cancer AS( " \
    "select patients.patient_id " \
    "from patients, skin_images " \
    "where patients.patient_id = skin_images.patient_id and NLfilter(skin_images.image_path, " \
    "'This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image.') " \
    ") " \
    "SELECT sick_audio.patient_id FROM sick_audio " \
    "UNION DISTINCT " \
    "SELECT sick_text.patient_id FROM sick_text " \
    "UNION DISTINCT " \
    "SELECT sick_image.patient_id FROM sick_image " \
    "UNION DISTINCT " \
    "SELECT sick_cancer.patient_id FROM sick_cancer " \
    "LIMIT 10"

medical_sql8 = "select patients.patient_id " \
    "from patients, skin_images " \
    "where patients.patient_id = skin_images.patient_id and patients.did_family_have_cancer = 1 and NLfilter(skin_images.image_path, " \
    "'This image shows a malignant human skin mole (considered abnormal/cancerous/sick) according to the image.') " \
    "limit 100"

medical_sql9 = "SELECT patients.patient_id " \
    "FROM patients, skin_images, x_ray_images " \
    "WHERE patients.patient_id = skin_images.patient_id AND patients.patient_id = x_ray_images.patient_id AND " \
    "NLjoin(skin_images.image_path, x_ray_images.image_path, 'Both images indicate diseases, one image shows malignant human skin mole, and another image shows sick human lungs with diseases.')"

# DUCKDB_FILEPATH = r"C:\Users\Mikev\study\scriptie\cars_database3.db"
#DUCKDB_FILEPATH = r"C:\Users\Mikev\study\scriptie\cars_database3_with_descr.db"
DUCKDB_FILEPATH = r"C:\Users\Mikev\study\scriptie\embedding\fashion_500_with_descriptions.db"
#DUCKDB_FILEPATH = r"C:\Users\Mikev\study\scriptie\cars_database3_with_descr_and_vec.db"
#DUCKDB_FILEPATH = r"C:\Users\Mikev\study\scriptie\medical_database_tdb_with_descriptions.duckdb"

def change_path(current_path: str, new_path: str):
    """Changes the system path to another one, so another code base can be used.
    
    E.g. C:\\thalamusdb_Batch\\src to C:\\thalamusdb_embed_certain_rows\\src"""
    if current_path in sys.path:
        sys.path.remove(current_path)
    else:
        print("Wrong current path given, give a correct one!")
        return

    if new_path not in sys.path:
        sys.path.append(new_path)
    else:
        print("Path already in system!")

    # Remove cached modules
    modules_to_delete = [m for m in sys.modules if m.startswith("tdb")]
    for m in modules_to_delete:
        del sys.modules[m]

def only_remove_path_and_modules(current_path: str):
    """Removes a system path"""
    if current_path in sys.path:
        sys.path.remove(current_path)
    else:
        print("Wrong current path given, give a correct one!")
        return
    
    # Remove cached modules
    modules_to_delete = [m for m in sys.modules if m.startswith("tdb")]
    for m in modules_to_delete:
        del sys.modules[m]

def get_approach_name(path: str) -> str:
    """This function takes the second last part between 2 '\\' of a path and returns that
    text. For example, it returns thalamusdb_Batch our of
    C:\\Users\\Mikev\\study\\scriptie\\backup\\thalamusdb_Batch\\src"""
    parts = path.replace("\\", "/").split("/")
    parts = [p for p in parts if p]  # remove empty strings
    return parts[-2]

def remove_limit(sql: str) -> str:
    """The function removes the LIMIT from a sql string."""
    sql = re.sub(r'\bLIMIT\s+\d+\s*', '', sql, flags=re.IGNORECASE)
    return ' '.join(sql.split())

def create_compare_list(sem_sql: str) -> list:
    """This function creates a compare list. This means that a query is runned
    on the original code base, but without the LIMIT, so that all the possible results
    are found"""
    #Re import
    from tdb.data.relational import Database
    from tdb.execution.constraints import Constraints
    from tdb.execution.engine import ExecutionEngine
    from tdb.queries.query import Query

    db = Database(DUCKDB_FILEPATH)
    engine = ExecutionEngine(db, 20, r"C:\Users\Mikev\study\scriptie\model_only_openAPI.json")
    constraints = Constraints()

    sem_sql_no_limit = remove_limit(sem_sql)

    query = Query(db, sem_sql_no_limit)
    results, _, _ = engine.run(query, constraints)
    return list(results.iloc[:,0])

def run_n_times(sem_sql: str, n: int, compare_list = None):
    """Runs A query n times for a code base. For every run the output is checked
     with the compare list and the measurements are returned."""
    #Re import
    from tdb.data.relational import Database
    from tdb.execution.constraints import Constraints
    from tdb.execution.engine import ExecutionEngine
    from tdb.queries.query import Query

    db = Database(DUCKDB_FILEPATH)
    engine = ExecutionEngine(db, 20, r"C:\Users\Mikev\study\scriptie\model_only_openAPI.json")
    constraints = Constraints()
    
    query = Query(db, sem_sql)

    total_times = []
    processed_tasks_list = []
    n_LLM_calls_list = []
    n_input_tokens_list = []
    n_output_tokens_list = []
    cpu_times = []
    precision_list = []
    recall_list = []
    f1_list = []
    compare_list_length = []
    length_result_list = []

    for _ in range(n):
        start = time.process_time()
        results, counter, total_time = engine.run(query, constraints)
        cpu_time = time.process_time() - start

        length_result_list.append(len(results))
        if compare_list is not None:
            compare_list_length.append(len(compare_list))
            if len(compare_list) > 0:
                result_list = results.iloc[:, 0].tolist()
                if len(result_list) > 0:
                    true_positives = sum(1 for id in result_list if id in compare_list)

                    precision = true_positives / len(result_list) * 100
                    recall = true_positives / len(compare_list) * 100
                    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

                    precision_list.append(f"{precision} %")
                    recall_list.append(f"{recall} %")
                    f1_list.append(f"{f1_score} %")
                else:
                    precision_list.append("NA")
                    recall_list.append("NA")
                    f1_list.append("NA")
            else:
                precision_list.append("No compare empty")
                recall_list.append("No compare empty")
                f1_list.append("No compare empty")

        else:
            precision_list.append("No compare list")
            recall_list.append("No compare list")
            f1_list.append("No compare list")
            compare_list_length.append(0)
        total_times.append(total_time)
        processed_tasks_list.append(counter.processed_tasks)
        n_LLM_calls_list.append(sum(c.LLM_calls for c in counter.model2counters.values()))
        n_input_tokens_list.append(sum(c.input_tokens for c in counter.model2counters.values()))
        n_output_tokens_list.append(sum(c.output_tokens for c in counter.model2counters.values()))
        cpu_times.append(cpu_time)

        # a pause to avoid reaching the limit of liteLLM
        time.sleep(10)

    return (total_times, 
            processed_tasks_list, 
            n_LLM_calls_list, 
            n_input_tokens_list, 
            n_output_tokens_list, 
            cpu_times, 
            precision_list, 
            recall_list, 
            f1_list,
            compare_list_length,
            length_result_list
    )

def add_avg(measures: list) -> list:
    """Calculates the average of a list and adds it at the end of the list."""
    """adds the average at the end of the list"""
    return measures + [sum(measures) / len(measures)]

def create_df(total_times, processed_tasks_list, n_LLM_calls_list, n_input_tokens_list, n_output_tokens_list, cpu_times, precision_list, recall_list, f1_list, compare_list_length, length_result_list):
    """"""
    if not (len(total_times) == len(processed_tasks_list) == len(n_LLM_calls_list) == len(n_input_tokens_list) == len(n_output_tokens_list) == len(cpu_times) == len(precision_list) == len(recall_list) == len(f1_list)):
        print("lists aren't equal length")
    
    length_df = len(total_times)
    run_count = [str(i) for i in range(length_df)] + ["AVG"]
    total_times = add_avg(total_times)
    processed_tasks_list = add_avg(processed_tasks_list)
    n_LLM_calls_list = add_avg(n_LLM_calls_list)
    n_input_tokens_list = add_avg(n_input_tokens_list)
    n_output_tokens_list = add_avg(n_output_tokens_list)
    cpu_times = add_avg(cpu_times)
    precision_list = precision_list + ["nvt"]
    recall_list = recall_list + ["nvt"]
    f1_list = f1_list + ["nvt"]
    compare_list_length = add_avg(compare_list_length)
    length_result_list = add_avg(length_result_list)

    dict_df = {"Index": run_count, "total times": total_times, "cpu times": cpu_times,  "n LLM calls": n_LLM_calls_list,
               "n input tokes": n_input_tokens_list, "n output tokes": n_output_tokens_list, "processed tasks": processed_tasks_list,
               "precision": precision_list, "recall": recall_list, "f1 score": f1_list, "compare_list_length": compare_list_length,
               "length_result_list": length_result_list}
    df = pd.DataFrame(data = dict_df)
    return df

def add_avg_tab_dfs(dfs: dict):
    """Extract all the averages out of all the variants and put them in one df,
    so it becomes one tab in the excel later where the user can see the avg in
    one view"""
    avgs = dict()
    for k, df in dfs.items():
        avg = df[df['Index'] == 'AVG'].iloc[0]
        avgs[k] = avg
    new_df = pd.DataFrame(data = avgs)
    new_df = new_df.T
    new_dfs = {'Averages': new_df}
    for k, df in dfs.items():
        new_dfs[k] = df
    return new_dfs

def save_in_excel_n_worksheets(dfs: dict, file_name: str) -> None:
    """Saves the dict inside an excel sheet in n different sheets."""
    base_path = r"C:\Users\Mikev\study\scriptie\resluts\car_results"
    file_path = base_path + "\\" + file_name
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for k, df in dfs.items():
            df.to_excel(writer, sheet_name = k)

def run_all_one_sql(new_dbs: list[str], sem_sql: str, name_sql: str, n: int, og_db = None, compare_l = None):
    """This function runs all the approaches (code bases) for one sql and saves them together, with averages and
    accuracy compared to a original run inexcel."""
    if og_db == None:
        # Since in this approach every base case is the normal thalamusdb, this is 
        # always taken when given None
        og_db = r"C:\Users\Mikev\study\scriptie\thalamusdb\src"
    
    sys.path.append(og_db)

    sys.path.append(r"C:\Users\Mikev\study\scriptie\cars_data_complete")

    dfs = dict()

    if compare_l != None:
        cur_compare_list = compare_l
    else:
        cur_compare_list = create_compare_list(sem_sql = sem_sql) 

    cur_path = og_db
    total_times, processed_tasks_list, n_LLM_calls_list, n_input_tokens_list, n_output_tokens_list, cpu_times, precision, recall, f1, compare_list_length, length_result_list = run_n_times(sem_sql=sem_sql, n=n, compare_list=cur_compare_list)
    df_name = "Base case"
    dfs[df_name] = create_df(total_times, processed_tasks_list, n_LLM_calls_list, n_input_tokens_list, n_output_tokens_list, cpu_times, precision, recall, f1, compare_list_length, length_result_list)
    # a pause to avoid reaching the limit of liteLLM
    time.sleep(60)
    
    for approach in new_dbs:
        df_name = get_approach_name(approach)
        print(df_name)
        change_path(current_path = cur_path, new_path = approach)
        total_times, processed_tasks_list, n_LLM_calls_list, n_input_tokens_list, n_output_tokens_list, cpu_times, precision, recall, f1, compare_list_length, length_result_list = run_n_times(sem_sql=sem_sql, n=n, compare_list=cur_compare_list)
        dfs[df_name] = create_df(total_times, processed_tasks_list, n_LLM_calls_list, n_input_tokens_list, n_output_tokens_list, cpu_times, precision, recall, f1, compare_list_length, length_result_list)
        cur_path = approach

        # a pause to avoid reaching the limit of liteLLM
        time.sleep(60)

    dfs_with_avg = add_avg_tab_dfs(dfs)
    save_in_excel_n_worksheets(dfs = dfs_with_avg, file_name = ("r_" + name_sql + ".xlsx"))

    # removing the path and modules, so the user doesn't have to restart the notebook to run again
    # This is only relevant when using it in a Jupyer Notebook
    only_remove_path_and_modules(cur_path)
        

if __name__ == '__main__':
    # Example of main:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

    path1 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_Batch\src"
    path3 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_embed_certain_rows\src"
    path5 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_earlier_join\src"
    path6 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_order_by_LLM\src"
    path7 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_all_improvements\src"
    path8 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_LLM_descr\src"
    path9 = r"C:\Users\Mikev\study\scriptie\backup\thalamusdb_LLM_descr_only_image\src"

    new_approaches = [path1, path3, path5, path8, path7]

    with open(r"C:\Users\Mikev\study\scriptie\compare_list_500\ecomm_sql_1.pkl", "rb") as f:
        c_list = pickle.load(f)
    
    run_all_one_sql(new_approaches, sem_sql_ecomm, "ecomm_sql1_p2", 5, compare_l = c_list)

    time.sleep(120)

    with open(r"C:\Users\Mikev\study\scriptie\compare_list_500\ecomm_sql_2.pkl", "rb") as f:
        c_list = pickle.load(f)
    
    run_all_one_sql(new_approaches, sem_sql_ecomm2, "ecomm_sql2_p2", 5, compare_l = c_list)

    time.sleep(120)

    with open(r"C:\Users\Mikev\study\scriptie\compare_list_500\ecomm_sql_7.pkl", "rb") as f:
        c_list = pickle.load(f)
    
    run_all_one_sql(new_approaches, sem_sql_ecomm7, "ecomm_sql-p2", 5, compare_l = c_list)

    time.sleep(120)

    with open(r"C:\Users\Mikev\study\scriptie\compare_list_500\ecomm_sql_8.pkl", "rb") as f:
        c_list = pickle.load(f)
    
    run_all_one_sql(new_approaches, sem_sql_ecomm8, "ecomm_sql8_p2", 5, compare_l = c_list)

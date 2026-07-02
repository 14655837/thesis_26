'''
Created on Jul 16, 2025

@author: immanueltrummer

@rewrite: Jiale Lao
Rewritten to use multi-threading (ThreadPoolExecutor) instead of multi-processing.
'''
import litellm
from concurrent.futures import ThreadPoolExecutor

from litellm import completion
from tdb.operators.semantic_operator import SemanticOperator

import numpy as np

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _filter_completion_wrapper(item_text, kwargs):
    """Invoke completion function with given keyword arguments.

    Args:
        item_text (str): Text representation of the item.
        kwargs (dict): Keyword arguments for the completion function.

    Returns:
        tuple: (item_text, kwargs, LLM response).
    """
    # Ensure parameters are dropped for logging where applicable
    litellm.drop_params = True
    response = completion(**kwargs)
    return item_text, kwargs, response


class UnaryFilter(SemanticOperator):
    """Base class for unary filters specified in natural language."""

    def __init__(
            self, db, operator_ID, batch_size,
            config_path, query, predicate):
        """
        Initializes the unary filter.

        Args:
            db: Database containing the filtered table.
            operator_ID (str): Unique identifier for the operator.
            batch_size (int): Number of items to process per call.
            config_path (str): Path to the configuration file for models.
            query: Query containing the predicate.
            predicate: predicate expressed in natural language.
        """
        super().__init__(db, operator_ID, batch_size, config_path)
        self.query = query
        self.filtered_table = predicate.table
        self.filtered_alias = predicate.alias
        self.filtered_column = predicate.column
        self.filter_condition = predicate.condition
        self.filter_sql = predicate.sql
        self.tmp_table = f'ThalamusDB_{self.operator_ID}'
        self.order_sql = None
        self.iteration = 0

    def _evaluate_predicate_parallel(self, item_texts):
        """Evaluates the filter conditions using the LLM concurrently (threads).

        Args:
            item_texts: List of items to evaluate.

        Returns:
            List of tuples (item_text, result) where result is True or False.
        """
        # Prepare keyword inputs for completion function
        inputs = []
        for item_text in item_texts:
            messages = [self._message(item_text)]
            base = self._best_model_args(messages)['filter']
            kwargs = {**base, 'messages': messages}
            inputs.append((item_text, kwargs))

        # Use a thread pool to evaluate predicates concurrently
        # Threads are appropriate here since LLM calls are I/O-bound.
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(_filter_completion_wrapper, it, kw)
                       for (it, kw) in inputs]
            inputs_outputs = [f.result() for f in futures]

        # Update cost counters
        for _, kwargs, response in inputs_outputs:
            model = kwargs['model']
            self.update_cost_counters(model, response)

        # Extract evaluation results
        results = []
        for item_text, _, response in inputs_outputs:
            result = str(response.choices[0].message.content)
            results.append((item_text, result == '1'))

        return results

    def _gpt_filter_bias(self, model):
        """Add logit bias on output tokens for GPT models.

        Args:
            model (str): Name of the model to use.

        Returns:
            dict: Logit bias to encourage 0/1 outputs for GPT models.
        """
        if self._gpt4_style_model(model):
            return {15: 100, 16: 100}
        else:
            return {}
        
    def _message(self, item_text):
        """Create a message for the LLM describing the evaluation task.

        Args:
            item_text (str): Text representation of the item.

        Returns:
            dict: Message for the LLM.
        """
        item = self._encode_item(item_text)
        question = (
            'Does the following item satisfy the condition '
            f'"{self.filter_condition}"? '
            'Answer with 1 for yes, 0 for no.')
        message = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': question
                },
                item
            ]
        }
        return message

    def _retrieve_items(self, nr_rows, order):
        """Retrieve items to process next from the filtered table.

        This method is used to retrieve items from the filtered table
        based on the specified number of rows and order.

        Args:
            nr_rows (int): Number of rows to retrieve.
            order (tuple): None or tuple (column, ascending flag).
        """
        # Retrieve items from the filtered table
        if self.order_sql == None:
            order_sql = '' if order is None \
                else f'ORDER BY {order[0]} {"ASC" if order[1] else "DESC"}'
        else:
            order_sql = self.order_sql
        sql = (
            f'SELECT base_{self.filtered_column} FROM {self.tmp_table} '
            'WHERE result IS NULL '
            f'{order_sql} LIMIT {nr_rows}')
        rows = self.db.execute2list(sql)
        return [row[0] for row in rows]

    def prepare(self):
        """Prepare for execution by creating intermediate result table.

        The temporary table contains the columns of the filtered table,
        as well as columns storing the result of filter evaluations (via
        LLMs) and a result used for simulating optimizer choices.
        """
        base_columns = self.db.columns(self.filtered_table)
        temp_schema_parts = ['result BOOLEAN', 'simulated BOOLEAN']
        for col_name, col_type in base_columns:
            tmp_col_name = f'base_{col_name}'
            temp_schema_parts.append(f'{tmp_col_name} {col_type}')

        create_table_sql = \
            f'CREATE OR REPLACE TEMPORARY TABLE {self.tmp_table}(' + \
            ', '.join(temp_schema_parts) + ')'
        self.db.execute2list(create_table_sql)

        # Use pure SQL predicates for pruning, if available
        other_filters = self.query.alias2unary_sql[self.filtered_alias]
        where_sql = (
            f'WHERE {other_filters.sql()} '
            f'AND {self.filtered_column} IS NOT NULL')
        fill_table_sql = \
            f'INSERT INTO {self.tmp_table} ' + \
            'SELECT NULL, NULL, ' + \
            ', '.join(c[0] for c in base_columns) + ' ' + \
            'FROM ' + self.filtered_table + ' ' + \
            where_sql
        self.db.execute2list(fill_table_sql)

        # Initialize count of unprocessed tasks
        count_sql = f'SELECT COUNT(*) FROM {self.tmp_table}'
        count_result = self.db.execute2list(count_sql)
        self.counters.unprocessed_tasks = count_result[0][0]

    def execute(self, order):
        """Execute operator on a given number of ordered rows.

        Args:
            order (tuple): None or tuple (column, ascending flag).
        """
        # Retrieve nr_rows in sort order from temporary table
        items_to_process = self._retrieve_items(self.batch_size, order)
        # Evaluate predicates on different items concurrently (threads)
        results = self._evaluate_predicate_parallel(items_to_process)
        # Update results in the temporary table
        for item_text, result in results:
            # Escape single quotes in item text for SQL
            escaped_item_text = item_text.replace("'", "''")
            update_sql = (
                f'UPDATE {self.tmp_table} '
                f'SET result = {result}, '
                f'simulated = {result} '
                f"WHERE base_{self.filtered_column} = '{escaped_item_text}'")
            self.db.execute2list(update_sql)
        # Update task counters
        self.counters.processed_tasks += len(items_to_process)
        self.counters.unprocessed_tasks -= len(items_to_process)

    # def _open_cars_embedding(self):
    #     type_of_id = None
    #     print("self.filtered_column = ", self.filtered_column)

    #     if "image_path" in self.filtered_column:
    #         print("Loading image embeddings from DB")
    #         type_of_id = "base_car_id"
    #         rows = self.db.execute2list("SELECT car_id, embeddings FROM car_image_embeddings")
    #         vectors = {int(row[0]): np.array(row[1]) for row in rows}
    #         print(f"Image embeddings loaded: {len(vectors)}")

    #     elif "audio_path" in self.filtered_column:
    #         print("Loading audio embeddings from DB")
    #         type_of_id = "base_car_id"
    #         rows = self.db.execute2list("SELECT car_id, embeddings FROM car_audio_embeddings")
    #         vectors = {int(row[0]): np.array(row[1]) for row in rows}
    #         print(f"Audio embeddings loaded: {len(vectors)}")

    #     else:
    #         print("Nothing happened, because of image_id and audio_id in unary_filter")
    #         vectors = None

    #     return vectors, type_of_id


    # def _open_embeddings(self):
    #     print("self.filtered_column = ", self.filtered_column)

    #     # Handle both string and list
    #     columns = self.filtered_column if isinstance(self.filtered_column, list) else [self.filtered_column]

    #     # Find any column ending in _path
    #     path_column = next((col for col in columns if col.endswith("_path")), None)
    #     if path_column is None:
    #         print("Nothing happened, no _path column found in filtered_column")
    #         return None, None

    #     # Find matching table by checking which existing tables contain prefix and end with _embeddings
    #     prefix = path_column.replace("_path", "")
    #     existing_tables = {row[0] for row in self.db.execute2list("SHOW TABLES")}
    #     table = next((t for t in existing_tables if prefix in t and t.endswith("_embeddings")), None)

    #     if table is None:
    #         print(f"No embedding table found for {path_column} (looked for *{prefix}*_embeddings)")
    #         return None, None

    #     # Derive type_of_id from first column of embedding table
    #     col_info = self.db.execute2list(f"PRAGMA table_info('{table}')")
    #     type_of_id = col_info[0][1]

    #     rows = self.db.execute2list(f"SELECT {type_of_id}, embeddings FROM {table}")
    #     vectors = {int(row[0]): np.array(row[1]) for row in rows}
    #     print(f"Embeddings loaded from {table}: {len(vectors)}")

    #     print(f"type_of_id: {type_of_id}")
    #     print(self.db.execute2list(f"PRAGMA table_info('{self.tmp_table}')"))
    #     return vectors, type_of_id


    # def _open_embeddings(self):
    #     print("self.filtered_column = ", self.filtered_column)

    #     columns = self.filtered_column if isinstance(self.filtered_column, list) else [self.filtered_column]
    #     existing_tables = [row[0] for row in self.db.execute2list("SHOW TABLES")]

    #     def has_embeddings(t):
    #         return "embeddings" in [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{t}')")]

    #     # First: try to find a table whose name contains any filtered_column word
    #     table = next(
    #         (t for t in existing_tables
    #         if has_embeddings(t) and any(col in t for col in columns)),
    #         None
    #     )
    #     # Fallback: just take the first table with embeddings
    #     if table is None:
    #         table = next((t for t in existing_tables if has_embeddings(t)), None)

    #     if table is None:
    #         print("No embedding table found in database")
    #         return None, None

    #     col_names = [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{table}')")]
    #     tmp_cols = [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{self.tmp_table}')")]

    #     # Pick the _id column that also exists (as base_<col>) in tmp_table
    #     id_col = next(
    #         (c for c in col_names
    #         if c.endswith("_id") and c != "image_id" and f"base_{c}" in tmp_cols),
    #         None
    #     ) or next(
    #         (c for c in col_names if c.endswith("_id") and c != "image_id"),
    #         None
    #     ) or next(
    #         (c for c in col_names if c == "id"),
    #         None
    #     )

    #     if id_col is None:
    #         print(f"Could not find an id column in {table}")
    #         return None, None

    #     rows = self.db.execute2list(f"SELECT {id_col}, embeddings FROM {table}")
    #     vectors = {int(row[0]): np.array(row[1]) for row in rows}
    #     print(f"Embeddings loaded from {table}: {len(vectors)}, id_col: {id_col}")
    #     return vectors, id_col

    def _open_embeddings(self):
        print("self.filtered_column = ", self.filtered_column)

        columns = self.filtered_column if isinstance(self.filtered_column, list) else [self.filtered_column]
        existing_tables = [row[0] for row in self.db.execute2list("SHOW TABLES")]

        def get_columns(t):
            return [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{t}')")]

        table = next(
            (t for t in existing_tables
            if "embeddings" in get_columns(t)
            and any(col in get_columns(t) for col in columns)),
            None
        )

        if table is None:
            print(f"No embedding table matched for {columns}, skipping ordering")
            return None, None

        col_names = get_columns(table)
        tmp_cols = [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{self.tmp_table}')")]

        id_col = next(
            (c for c in col_names
            if c.endswith("_id") and c != "image_id" and f"base_{c}" in tmp_cols),
            None
        ) or next(
            (c for c in col_names if c.endswith("_id") and c != "image_id"),
            None
        ) or next(
            (c for c in col_names if c == "id"),
            None
        )

        if id_col is None:
            print(f"Could not find an id column in {table}")
            return None, None

        rows = self.db.execute2list(f"SELECT {id_col}, embeddings FROM {table}")
        vectors = {int(row[0]): np.array(row[1]) for row in rows}
        print(f"Embeddings loaded from {table}: {len(vectors)}, id_col: {id_col}")
        return vectors, id_col

    # def _open_embeddings(self):
    #     print("self.filtered_column = ", self.filtered_column)

    #     columns = self.filtered_column if isinstance(self.filtered_column, list) else [self.filtered_column]
    #     path_column = next((col for col in columns if col.endswith("_path")), None)
    #     if path_column is None:
    #         print("Nothing happened, no _path column found in filtered_column")
    #         return None, None

    #     prefix = path_column.replace("_path", "").split("_")[-1]
    #     existing_tables = {row[0] for row in self.db.execute2list("SHOW TABLES")}

    #     table = next(
    #         (t for t in existing_tables
    #         if prefix in t
    #         and "embeddings" in [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{t}')")]),
    #         None
    #     )

    #     if table is None:
    #         print(f"No embedding table found for {path_column} (looked for table with '{prefix}' and 'embeddings' column)")
    #         return None, None

    #     col_info = self.db.execute2list(f"PRAGMA table_info('{table}')")
    #     col_names = [row[1] for row in col_info]

    #     # Prefer any _id column that isn't image_id
    #     id_col = next((c for c in col_names if c.endswith("_id") and c != "image_id"), None) or \
    #             next((c for c in col_names if c == "id"), None)
    #     if id_col is None:
    #         print(f"Could not find an id column in {table}")
    #         return None, None
    #     type_of_id = id_col

    #     rows = self.db.execute2list(f"SELECT {type_of_id}, embeddings FROM {table}")
    #     vectors = {int(row[0]): np.array(row[1]) for row in rows}
    #     print(f"Embeddings loaded from {table}: {len(vectors)}")
    #     print(f"type_of_id: {type_of_id}")
    #     print(self.db.execute2list(f"table_info('{self.tmp_table}')"))
    #     print("Embedding works!")
    #     return vectors, type_of_id

    
    def create_embedding_order(self, certain_rows):
        """Order rows based on similarity of certain_rows to others"""
        vectors, type_of_id = self._open_embeddings()

        if vectors is None:
            print("No ordering")
            self.order_sql = None
            return 1

        certain_ids = list(certain_rows.iloc[:, 0])

        # If no match, try the second column of certain_rows
        if not any(i in vectors for i in certain_ids) and certain_rows.shape[1] > 1:
            certain_ids = list(certain_rows.iloc[:, 1])
            print("Retrying with second column, certain_ids sample:", certain_ids[:5])

        try:
            certain_ids = [int(i) for i in certain_ids]
        except (ValueError, TypeError):
            pass

        comparison_vectors = {k: vec for k, vec in vectors.items() if k in certain_ids}
        if len(comparison_vectors) < 1:
            print("Something went wrong in finding comparison vectors, skipping ordering")
            self.order_sql = None
            return 1

        tmp_cols = [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{self.tmp_table}')")]
        tmp_id_col = next((col for col in tmp_cols if col.endswith(type_of_id)), None)
        if tmp_id_col is None:
            print(f"Could not find {type_of_id} in {self.tmp_table}")
            self.order_sql = None
            return 1

        sql = f"SELECT DISTINCT {tmp_id_col} FROM {self.tmp_table}"
        surviving_ids = {int(row[0]) for row in self.db.execute2list(sql)}

        filtered_vectors = {k: v for k, v in vectors.items() if k in surviving_ids}

        if len(filtered_vectors) < 1:
            print("filtering vectors gone wrong")
            self.order_sql = None
            return 1

        keys_order = list(filtered_vectors.keys())

        all_scores = []
        for _, com_vec in comparison_vectors.items():
            scores = np.array([cosine(com_vec, vec) for vec in filtered_vectors.values()])
            all_scores.append(scores)
        avg_scores = np.sum(all_scores, axis=0)

        avg_scores_dict = {keys_order[i]: avg_scores[i] for i in range(len(keys_order))}
        ranked = sorted(avg_scores_dict.items(), key=lambda x: x[1], reverse=True)
        ids = [img_id for img_id, score in ranked]

        not_found_ids = [id for id in surviving_ids if id not in ids]
        ranking = ids + not_found_ids

        when_clauses = " ".join(
            f"WHEN {tmp_id_col} = '{img_id}' THEN {rank}"
            for rank, img_id in enumerate(ranking)
        )
        self.order_sql = f"ORDER BY CASE {when_clauses} ELSE 9999 END ASC"
        return 1

    # def create_embedding_order(self, certain_rows):
    #     """Order the car_id based on the similairity of certain_rows to others"""
    #     vectors = None
    #     vectors, type_of_id = self._open_embeddings()
        

    #     if vectors is None:
    #         print("No ordering")
    #         self.order_sql = None#f"ORDER BY CASE {None} ELSE 9999 END ASC"
    #         return
        
    #     certain_ids = list(certain_rows.iloc[:, 0])
    #     comparison_vectors = {k: vec for k, vec in vectors.items() if k in certain_ids}
    #     if len(comparison_vectors) < 1:
    #         print("Something went wrong in finding comparison vectors")

    #     tmp_cols = [row[1] for row in self.db.execute2list(f"PRAGMA table_info('{self.tmp_table}')")]
    #     tmp_id_col = next((col for col in tmp_cols if col.endswith(type_of_id)), None)
    #     if tmp_id_col is None:
    #         print(f"Could not find {type_of_id} in {self.tmp_table}")
    #         self.order_sql = None
    #         return
    #     sql = f"SELECT DISTINCT {tmp_id_col} FROM {self.tmp_table}"
    #     #sql = f"SELECT DISTINCT {type_of_id} FROM {self.tmp_table}"
    #     #sql = f"SELECT {type_of_id} FROM {self.tmp_table}"
    #     surviving_ids = {int(row[0]) for row in self.db.execute2list(sql)}

    #     filtered_vectors = {k: v for k, v in vectors.items() if k in surviving_ids}

    #     if len(filtered_vectors) < 1:
    #         print("filtering vectors gone wrong")
    #         self.order_sql = f"ORDER BY CASE {None} ELSE 9999 END ASC"
    #         return
        
    #     keys_order = [key for key in filtered_vectors.keys()]

    #     all_scores = []
    #     for _, com_vec in comparison_vectors.items():
    #         scores = np.array([cosine(com_vec, vec) for _, vec in filtered_vectors.items()])
    #         all_scores.append(scores)
    #     # print(all_scores)
    #     avg_scores = np.sum(all_scores, axis=0)

    #     avg_scores_dict = {keys_order[i]: avg_scores[i] for i in range(len(keys_order))}
    #     ranked = sorted(avg_scores_dict.items(), key=lambda x: x[1], reverse=True)
    #     ids = [img_id for img_id, score in ranked]

    #     # temp_scores = [score for img_id, score in ranked]

    #     not_found_ids = []
    #     if len(ids) < len(surviving_ids):
    #         not_found_ids = [id for id in surviving_ids if id not in ids]
            
    #     ranking = ids + not_found_ids
    #     # when_clauses = " ".join(
    #     #     f"WHEN {type_of_id} = {img_id} THEN {rank}"
    #     #     for rank, img_id in enumerate(ranking)
    #     # )
    #     when_clauses = " ".join(
    #         f"WHEN {tmp_id_col} = {img_id} THEN {rank}"
    #         for rank, img_id in enumerate(ranking)
    #     )
    #     self.order_sql = f"ORDER BY CASE {when_clauses} ELSE 9999 END ASC"
    #     return 1
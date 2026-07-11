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
    
    #Added func:
    def _should_pushdown_semijoin(self, db, filtered_alias, 
                               other_alias, join_col, other_table,
                               other_filters_sql, threshold=0.8):
        """Returns True if semi-join pushdown is likely to reduce row count.
        
        Args:
            threshold: Only push down if selectivity < this fraction.
        """
        # Count total rows in the filtered table
        total_sql = (
            f'SELECT COUNT(*) FROM {filtered_alias} '
            f'WHERE {join_col} IS NOT NULL')
        total = db.execute2list(total_sql)[0][0]
        if total == 0:
            return False

        # Count rows surviving the semi-join
        surviving_sql = (
            f'SELECT COUNT(*) FROM {filtered_alias} '
            f'WHERE {join_col} IN ('
            f'  SELECT {join_col} FROM {other_table} '
            f'  WHERE {other_filters_sql}'
            f')')
        surviving = db.execute2list(surviving_sql)[0][0]

        selectivity = surviving / total
        return selectivity < threshold

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

        other_filters = self.query.alias2unary_sql[self.filtered_alias]
        
        #Changed
        # ___
        extra_conditions = []

        for pred in self.query.cross_table_predicates:
            if pred.left_alias == self.filtered_alias:
                join_col = pred.left_column
                other_alias = pred.right_alias
                other_table = self.query.alias2table[other_alias]
            elif pred.right_alias == self.filtered_alias:
                join_col = pred.right_column
                other_alias = pred.left_alias
                other_table = self.query.alias2table[other_alias]
            else:
                continue

            other_sql_filter = self.query.alias2unary_sql[other_alias].sql()

            should_pushdown_bool = self._should_pushdown_semijoin(
                    self.db, self.filtered_table, other_alias,
                    join_col, other_table, other_sql_filter)
            print("Should pushdown? ", should_pushdown_bool)
            if should_pushdown_bool:
                semi_join = (
                    f'{join_col} IN ('
                    f'SELECT {join_col} FROM {other_table} '
                    f'WHERE {other_sql_filter})')
                extra_conditions.append(semi_join)

        where_parts = [other_filters.sql()] + extra_conditions + \
                    [f'{self.filtered_column} IS NOT NULL']
        where_sql = 'WHERE ' + ' AND '.join(where_parts)

        fill_table_sql = (
            f'INSERT INTO {self.tmp_table} '
            f'SELECT NULL, NULL, '
            + ', '.join(c[0] for c in base_columns)
            + f' FROM {self.filtered_table} {where_sql}')
        self.db.execute2list(fill_table_sql)
        #___

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

    #ADDED
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

    #ADDED
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

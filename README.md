## Overview
This GitHub repositoryhas the code used for the Software Engineering master thesis Reducing Execution Cost and Time in ThalamusDB’s Query Optimizer, by Mike van der Deure. This thesis tries to improve the query optimizer of ThalamusDB. The original code for ThalamusDB can be found on https://github.com/itrummer/thalamusdb. The queries used and shown in run_sim.py come from https://github.com/SemBench/SemBench. 
This GitHub page includes 4 core different approaches that try to improve the ThalamusDB query optimizer. Those are shown in the approaches folder. To run an approach in python, the source code needs to be added to the system and then the needed functions to run can be imported. This can look like:
sys.path.append(“approaches/thalamusdb_embed_certain_rows/src")
from tdb.data.relational import Database
from tdb.execution.constraints import Constraints
from tdb.execution.engine import ExecutionEngine
from tdb.queries.query import Query
There also needs to be a connection to an LLM.
## Approaches
The approaches folder has 4 different approached improvements as well as 1 combination of the 2. These approaches are explained below.
The thalamusdb_Batch approach tries to improve the engine by sending multiple items at the same time into one LLM call instead of separately. The only changes to the original code are in src/tdb/operators/semantic_filter.py, where the _evaluate_predicate_parallel function is changed and the functions _batch_message and _extract_batch_response are added.
The thalamusdb_LLM_descr approach tries to improve the original code by using a description of the items instead of the original items, when the original items are images or audio. For this, the DuckDB database that is used, needs to be changed with a added description column, where every item is described by a text. In the used experiment, the descriptions where made with LLM calls. This approach is only changed in src/tdb/operators/semantic_operator.py, where the _get_description function is added and _encode_item is changed.
The thalamusdb_all_improvements is a combination of the approaches that follow. It is then also only changed on the places where the next two approaches are changed.
The thalamusdb_earlier_join approach tries to improve the original code by using a semi-join when the temporary tables are created. This should in queries where a semi-join is possible, remove rows so they are not reviewed by an LLM. The first file that is changed is /src/tdb/operators/semantic_filter.py, where the function prepare, which creates the temporary tables, is changed and _should_pushdown_semijoin is an added function to determine if the semi-join is worth to execute. The second file that is changed is /src/tdb/queries/query.py, where _collect_unary_sql_predicates is changed and in _get_unary_alias is an else statement added.
The last approach thalamusdb_embed_certain_rows tries to improve the original code by using embeddings to order the rows. For this, the DuckDB database needs an extra column, where vectors of every row are added so these rows can be compared to each other.  The first file that is changed is /src/tdb/execution/engine.py, where the run function is changed to add the ordering to engine. The second file that is changed is src/tdb/operators/semantic_filter.py. In this file, create_embedding_order is added where the comparing of the vectors and ordering happens, and _open_embeddings is added that handles opening the embedding vectors in the right way. Furthermore, the function _retrieve_items is changed.
## preprocessing
The preprocessing folder contains two files, which were used to create the data for the new columns. Data_to_descr.py was created to automate the creation of the descriptions of images and audio for the thalamusdb_LLM_descr approach. embedding.py was created for the creation of embeddings for the thalamusdb_embed_certain_rows approach.
## run_sim
The file run_sim.py, was created and used to automate the testing and experimenting process. It contains functions that were used to run and measure runs and to then save them in excel. It also contains the queries used for the runs.
## Analysis
The file accuracy.py was used and created to analyze the quality of the runs. This includes creating tables for precision, accuracy and F1-score. The file analysis.py was created to analyse the speed and costs of the approaches. The code was used to create tables and figures for the measurements done.
## Rest
The folder /example_data shows an example of a DuckDB database that was used for the medical scenario and the json file shows what models were used for most of the testing.
More context can be found in the thesis itself.

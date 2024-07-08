from langchain.chains import create_sql_query_chain
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


stss = st.session_state

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = Ollama(model="gemma2", temperature=0.0)
chain = create_sql_query_chain(llm, db)
system = """
You are a {dialect} expert. Given an input question, create a syntactically correct
{dialect} query to run. Unless the user specifies in the question a specific number of
examples to obtain, query for at most {top_k} results using the LIMIT clause as per
{dialect}. You can order results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are
needed to answer the question. Wrap each column name in double quotes (") to denote them
as delimited identifiers. Pay attention to use only the column names you can see in the
tables below. Be careful to not query for columns that do not exist. Also, pay attention
to which column is in which table. Pay attention to use date('now') function to get the
current date, if the question involves "today".

Only use the following tables:
{table_info}

Write an initial draft of the query. Then double check the {dialect} query for common
mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}")]
).partial(dialect=db.dialect)


def parse_final_answer(output: str) -> str:
    return output.split("Final answer: ")[1]

chain = create_sql_query_chain(llm, db, prompt=prompt) | parse_final_answer


if "query" not in stss:
    stss['query'] = ""

if "result" not in stss:
    stss['result'] = ""


def generate_query():
    st.session_state["query"] = chain.invoke(
        {
            "question": stss['user_input']
        }
    )
    stss['result'] = ''


def generate_result():
    try:
        result = db.run(stss['query'])
    except:
        result = 'An error occurred :('
    st.session_state['result'] = result


def main():
    st.markdown('# Natural Language to SQL Converter')
    with st.form(key='input_form'):
        st.text_input(label='Enter question', key='user_input')
        st.form_submit_button(label='Submit', on_click=generate_query)
    st.text_area(label="Query", value=stss['query'], disabled=True, key='query_box')
    st.button(
        label="Run query",
        disabled=(stss['query'] == ''),
        key='query_submit',
        on_click=generate_result
    )
    st.text_area(label='Result', value=stss['result'], disabled=True, key='result_box')


if __name__ == "__main__":
    main()

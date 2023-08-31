import openai
import yaml

from loguru import logger
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, SQLDatabaseChain
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.sql_database.prompt import _sqlite_prompt, PROMPT_SUFFIX
from langchain import OpenAI, SQLDatabase
from langchain import FewShotPromptTemplate, PromptTemplate

from swapfiets import get_credentials_dict  # private @ https://swapfiets-data.github.io/github-hosted-pypi/
from config import get_settings


settings = get_settings()


def classify_intent(query: str, model_name: str) -> str:
    """
    Zero shot intent classification: Info about the  company (vector store) or info about the customer (SQL)

    Args:
        query: customer question as string
        model_name: name of the llm used (for OpenAI API Call)

    Returns:
        intent: classified intent as string
    """
    # compose the prompt for zero-shot intent classification
    prompt = """
    Given the following intents and corresponding examples, classify the intent of the given text into either
    
    (1) Info about the company required:
    - ...
    
    (2) Info about the customer required:
    - ...
    
    (3) Neither of the two

    ## New Text:"""

    prompt += f"\n- {query}"

    # Generate few-shot intent classification response
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # extract the intent from the response
    intent = response.choices[0].text.strip()
    return intent


def classify_task(query, model_name):
    """
    Zero shot intent classification: task or no task

    Args:
        query: customer question as string
        model_name: name of the llm used (for OpenAI API Call)

    Returns:
        task: classified task as string
    """
    # Compose the prompt for few-shot intent classification
    prompt = """
    As a customer support agent, your role is to determine the appropriate course of action to assist the customer.
    Please analyze the provided text and classify whether you can address the customer's concern by
    (1) providing an explanation so the customer can fix their problem, e.g., explain invoices, prices, functionality (intent: Explain),
    or if
    (2) the agent needs to act, e.g., update the CRM, send a payment link (intent: Task).

    Answer: (1) Explain or (2) Task?

    ## Text:"""

    prompt += f"\n- {query}"

    # Generate few-shot intent classification response
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # extract the intent from the response
    task = response.choices[0].text.strip()
    return task


def get_vectorstore_results(query, llm, embeddings) -> str:
    """Function that accesses chromadb vectorstore to find a matching result for the query.

    Args:
        query: customer question as string
        llm: language model of choice (here: 'text-davinci-003')
        embeddings: embedding model of choice (here: 'text-embedding-ada-002')

    Returns:
        result: response from RetrievalQA as string

    """
    # initialise vectorstore and set up as retriever
    vectorstore = Chroma(persist_directory='../../data/chroma_db',
                         embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity",  # “similarity” or “mmr”
                                         search_kwargs={"k": 2})  # find 2 most similar results

    # create chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_rerank",  # “stuff”, “map_reduce”, “refine”, “map_rerank”
        retriever=retriever,
        return_source_documents=True
    )

    # avoid that the process stops if vector is unsuccessful
    try:
        result = qa({"query": query})['result']

    # specify default "context"
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        result = "No context info found"

    return result


def get_sql_results(query, llm, credentials) -> str:
    """
    Function that accesses sql database to find a matching result for the query.

    Args:
        query: customer question as string
        llm: language model of choice (here: 'text-davinci-003')
        credentials: Dictionary with Snowflake credentials to construct the connection_string

    Returns:
        result: response from SQLDatabaseChain as string
    """
    # setup connection
    conn_string = f"snowflake://{credentials['user']}:{credentials['password']}@{credentials['account']}/{credentials['database']}/franca?warehouse={credentials['warehouse']}&role={credentials['role']}"
    db = SQLDatabase.from_uri(conn_string)

    example_prompt = PromptTemplate(
        input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
        template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
    )

    # load yml file as dictionary
    with open("sql_few_shot.yml", "r") as yaml_file:
        examples_dict = yaml.safe_load(yaml_file)

    # initialise embedding model for similarity search (new query to example queries)
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # initialise similarity search
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples_dict,  # list of examples available to select from
        local_embeddings,  # embedding class used to produce embeddings to measure semantic similarity
        Chroma,  # type: ignore  # vectorstore class used to store the embeddings and do a similarity search over
        k=min(2, len(examples_dict)),  # this is the number of examples to include per prompt
    )

    # define few_shot_prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_sqlite_prompt + "Here are some examples:",
        suffix=PROMPT_SUFFIX,
        input_variables=["table_info", "input", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(llm=llm, db=db,
                                      prompt=few_shot_prompt,
                                      # verbose=True,
                                      top_k=10,
                                      return_intermediate_steps=True)

    # avoid that the process stops if sql is unsuccessful
    try:
        result = chain(query)['result']

    # specify default "context"
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        result = "No context info found, ask customer for clarification"

    return result


def llm_bot(query, settings):
    """First classify intent and task, then fetch results from relevant source.

    Args:
        query: customer question with relevant info (ideally including the current date and customer number)
        settings: class with env variables

    Returns:
        email_response: generated response
    """
    # count tokens of whole process
    with get_openai_callback() as cb:

        # get necessary credentials from settings
        credentials = get_credentials_dict(settings)
        api_key = settings.openai_api_key

        # specify models for different purposes
        model_name = 'text-davinci-003'
        embedding_model = 'text-embedding-ada-002'

        # initialise llm and embeddings model
        llm = OpenAI(openai_api_key=api_key, temperature=0, model_name=model_name)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embedding_model)

        # log setup
        logger.info(f"The query: {query}")
        logger.info(f"The model: {model_name}")
        logger.info(f"The embedding model: {embedding_model}")

        # apply function for intent classification
        logger.info(">> Step 1: Classify intent")
        intent = classify_intent(query=query, model_name=model_name)
        if "Info about the customer" in intent:
            logger.info(f"Intent: {intent} > search SQL database")
        elif "Info about the company" in intent:
            logger.info(f"Intent: {intent} > search vectorstore")
        else:
            logger.info(f"No clear intent: {intent} > search vectorstore")

        # apply function for task classification
        logger.info(">> Step 2: Classify answer type")
        task = classify_task(query=query, model_name=model_name)
        logger.info(f"{task}")

        # "Info about the company" >> needs to be found in SQL database
        if "Info about the customer" in intent:
            # fetch results
            logger.info(">> Step 3: Fetching results from SQL database")
            results = get_sql_results(query=query, llm=llm, credentials=credentials)
            logger.info(f"The result: {results}")

            # draft email response
            if "Explain" in task:
                # explanation only
                logger.info(">> Step 4: Draft email response > explanation only")
                prompt_template = PromptTemplate(input_variables=["results", "query"],
                                                 template=open("bot_sql_explain.template").read())

                prompt = prompt_template.format(results=results, query=query)

                # call API
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=800
                )

                # extract text response
                email_response = response.choices[0].text.strip()

            elif "Task" in task:
                # explanation only
                logger.info(">> Step 4: Draft email response > explanation and action")
                prompt_template = PromptTemplate(input_variables=["results", "query"],
                                                 template=open("bot_sql_action.template").read())

                prompt = prompt_template.format(results=results, query=query)

                # call API
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=800
                )

                # extract text response
                email_response = response.choices[0].text.strip()

            else:
                logger.info("Task classification gone wrong.")

        # "Info about the company" >> needs to be found in vectorstore
        elif "Info about the company" in intent:
            # fetch results
            logger.info(">> Step 3: Fetching results from vector database")
            results = get_vectorstore_results(query=query, llm=llm, embeddings=embeddings)
            logger.info(f"The result: {results}")

            # draft email response for 'explanation' or 'explanation + task'
            if "Explain" in task:
                # explanation only
                logger.info(">> Step 4: Draft email response > explanation only")
                prompt_template = PromptTemplate(input_variables=["results", "query"],
                                                 template=open("bot_vector_explain.template").read())

                prompt = prompt_template.format(results=results, query=query)

                # call API
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=800
                )

                # extract text response
                email_response = response.choices[0].text.strip()

            elif "Task" in task:
                # explanation and action
                logger.info(">> Step 4: Draft email response > explanation and action")
                prompt_template = PromptTemplate(input_variables=["results", "query"],
                                                 template=open("bot_vector_action.template").read())

                prompt = prompt_template.format(results=results, query=query)

                # call API
                response = openai.Completion.create(
                    engine=model_name,
                    prompt=prompt,
                    max_tokens=800
                )

                # extract text response
                email_response = response.choices[0].text.strip()
            #
            # else:
            #     logger.info("Task classification gone wrong.")

        # "Neither of the two" >> but double-check vector store
        else:
            logger.info(">> Step 3: Checking vector database for relevant info")
            results = get_vectorstore_results(query=query, llm=llm, embeddings=embeddings)
            logger.info(f"The result: {results}")

            #
            logger.info(">> Step 4: Draft email response > no intent, generic prompt template used")
            prompt_template = PromptTemplate(input_variables=["results", "query"],
                                             template=open("bot_generic.template").read())

            prompt = prompt_template.format(query=query, results=results)

            # call API
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=800
            )

            # extract text response
            email_response = response.choices[0].text.strip()

    # DONE
    logger.info(f'Done! Spent a total of {cb.total_tokens} tokens')
    return email_response


my_query = "..."


email = llm_bot(query=my_query, settings=settings)

# print(email)

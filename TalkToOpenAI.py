"""
Cisco Sample Code License 1.1
Author: flopach 2024
"""
from openai import OpenAI
import openai
import logging
import time

log = logging.getLogger("applogger")

class LLMOpenAI:
    def __init__(self, database, chat_model="gpt-3.5-turbo", embedding_model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.database = database
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def get_embeddings(self, data):
        """
        Get embeddings for the given data using OpenAI API

        Args:
            data (list): List of strings to be sent to the API

        Returns:
            list: Embeddings received from the API
        """
        try:
            embeddings = []
            for chunk in data:
                response = openai.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"  # Use the appropriate OpenAI embedding model
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            log.error(f"Error getting embeddings from OpenAI API: {str(e)}")
            raise
    
    def extend_api_description(self,query_string,path,operation,parameters):
        """
        Extend the description for each API REST Call operation

        Args:
            query_string (str): details of the REST API call
            path (str): REST API path
            operation (str): REST API operation (GET, POST, etc.) 
            parameters (str): Query parameters
        """

        # query vector DB for local data
        context_query = self.database.query_db(query_string,10)

        # create promt message with local context data
        message = f'Query path: "{path}"\nREST operation: {operation}\nshort description: {query_string}\n{parameters}\nUse this context delimited with XML tags:\n<context>\n{context_query}\n</context>'
        log.debug(f"=== Extending the description with: ===\n {message}")

        # ask GPT
        completion = self.client.chat.completions.create(
        model=self.chat_model,
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are provided information of a specific REST API query path of the Cisco Catalyst Center. Describe what this query is for in detail. Describe how this query can be used from a user perspective."},
            {"role": "user", "content": message}
        ]
        )

        return completion.choices[0].message.content

    def ask_llm(self, query_string, chat_history, n_results_apidocs=10, n_results_apispecs=10, n_results_userguide=10):
        """
        Ask the LLM with the query string.
        Search for context in vectorDB

        Args:
            query_string (str): details of the REST API call
            chat_history (list): List of previous chat messages
            n_results_apidocs (int): Number of documents return by vectorDB query for API docs on developer.cisco.com
            n_results_apispecs (int): Number of documents return by vectorDB query for extended API specification document
            n_results_userguide (int): Number of documents return by vectorDB query for user guide
        """
        # Record the start time
        start_time = time.time()

        # context queries to vectorDB
        context_query_apidocs = self.database.query_db(query_string, n_results_apidocs, "apidocs")
        context_query_apispecs = self.database.query_db(query_string, n_results_apispecs, "apispecs")
        context_query_userguide = self.database.query_db(query_string, n_results_userguide, "userguide")
        
        context = f'''Context information delimited with XML tags:\n<context>\n{context_query_apidocs}\n</context>
                    API specification context delimited with XML tags:\n<api-context>\n{context_query_apispecs}\n</api-context>
                    User guide context delimited with XML tags:\n<userguide-context>\n{context_query_userguide}\n</userguide-context>'''

        question = f"\n\nUser question: '{query_string}'"

        message = context + question

        log.debug(message)

        # Include chat history in the messages
        messages = chat_history + [
            {"role": "system", "content": """You are the Cisco Catalyst Center REST API and Python code assistant. You provide documentation and Python code for developers.
            Always list all available query parameters from the provided context. Include the REST operation and query path.
            1. you create documentation to the specific API calls. 
            2. you create an example source code in the programming language Python using the 'requests' library.
            Tell the user if you do not know the answer. If loops or advanced code is needed, provide it.
            ###
            Every API query needs to include the header parameter 'X-Auth-Token' for authentication and authorization. This is where the access token is defined.
            If the user does not have the access token, the user needs to call the REST API query '/dna/system/api/v1/auth/token' to receive the access token. Only the API query '/dna/system/api/v1/auth/token' is using the Basic authentication scheme, as defined in RFC 7617. All other API queries need to have the header parameter 'X-Auth-Token' defined.
            ###
            """},
            {"role": "user", "content": message}
        ]

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.8,
            messages=messages
        )

        # Calculate the total duration
        duration = round(time.time() - start_time, 2)
        exec_duration = f"The query '{query_string}' took **{duration} seconds** to execute."
        log.info(exec_duration)

        return completion.choices[0].message.content + "\n\n" + exec_duration
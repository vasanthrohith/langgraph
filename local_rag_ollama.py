from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing import Annotated, Dict, TypedDict
import os
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
local_llm = "mistral:latest"
print("loaded env variables")



class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            keys: A dictionary where each key is a string.
        """
        keys: Dict[str, any]



class ChatBot():
    def __init__(self, urls):
        self.urls = urls
        self.retriever = self.get_vectorstore()  # Call get_vectorstore method without passing urls

        



    def myquery(self, query):
         print("In myquery")
         self.query = query
         response = self.run_nodes(GraphState)
         return response



    def get_vectorstore(self):
        """ Get the urls data - create vectorstores and returns retriever """
        print("Getting url data...")
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(),
        )
        retriever = vectorstore.as_retriever()
        
        # print(retriever.get_relevant_documents("who is bat"))

        return retriever
    

    # ------------------------Nodes------------------------

    def retrieve(self, state):

        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        print("state >>> ", state)
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.retriever.invoke(question)
        return {"keys": {"documents": documents, "question": question}}
    

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with relevant documents
        """

        print("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        # local = state_dict["local"]

        # LLM
        # if local == "Yes":
        llm = ChatOllama(model=local_llm, format="json", temperature=0)
        # else:
        #     # llm = ChatMistralAI(
        #     #     mistral_api_key=mistral_api_key, temperature=0, model="mistral-medium"
        #     # )
        #     pass

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
            input_variables=["question","context"],
        )

        chain = prompt | llm | JsonOutputParser()

        # Score
        filtered_docs = []
        search = "No"  # Default do not opt for web search to supplement retrieval
        for d in documents:
            score = chain.invoke(
                {
                    "question": question,
                    "context": d.page_content,
                }
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search = "Yes"  # Perform web search
                continue

            

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                # "local": local,
                "run_web_search": search,
            }
        }
    

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains generation
        """
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        # local = state_dict["local"]

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        # if local == "Yes":
        llm = ChatOllama(model="mistral:7b-instruct", temperature=0)
        # else:
        #     # llm = ChatMistralAI(
        #     #     model="mistral-medium", temperature=0, mistral_api_key=mistral_api_key
        #     # )

        #     pass

        # Post-processing
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {
            "keys": {"documents": documents, "question": question, "generation": generation}
        }
    


    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        # local = state_dict["local"]

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for retrieval. \n
            Look at the input and try to reason about the underlying sematic intent / meaning. \n
            Here is the initial question:
            \n ------- \n
            {question}
            \n ------- \n
            Formulate an improved question: """,
            input_variables=["question"],
        )

        # Grader
        # LLM
        # if local == "Yes":
        llm = ChatOllama(model="mistral:7b-instruct", temperature=0)
        # else:
        #     # llm = ChatMistralAI(
        #     #     mistral_api_key=mistral_api_key, temperature=0, model="mistral-medium"
        #     # )
        #     pass

        # Prompt
        chain = prompt | llm | StrOutputParser()
        better_question = chain.invoke({"question": question})

        return {
            "keys": {"documents": documents, "question": better_question}
        }
    

    def web_search(self, state):
        """
        Web search based on the re-phrased question using Tavily API.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Web results appended to documents.
        """

        print("---WEB SEARCH---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        # local = state_dict["local"]

        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        print(docs)
        filtered_contents = [d["content"] for d in docs if d["content"] is not None]
        web_results = "\n".join(filtered_contents)
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"keys": {"documents": documents, "question": question}}
    

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer or re-generate a question for web search.

        Args:
            state (dict): The current state of the agent, including all keys.

        Returns:
            str: Next node to call
        """

        print("---DECIDE TO GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        filtered_documents = state_dict["documents"]
        search = state_dict["run_web_search"]

        if search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"



    # -----------------------------------------------------
    


    def run_nodes(self,GraphState):
        print("--run_nodes--")

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query) 
        workflow.add_node("web_search", self.web_search)  # web search
        

        # # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)


        # Compile
        app = workflow.compile()
        # print(self.query)
        return app.invoke({"keys":{"question":self.query}})
        
        

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

chatbot = ChatBot(urls=urls)
print(chatbot.retriever)
response = chatbot.myquery(query = "what are agents?")
print(response)













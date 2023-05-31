from tools.tools import query_vector_database#, get_profile_url_w_occupation

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from key_storage import key_dict

def linkedin_url_lookup(name: str, occupation:str=None) -> str:

    """For the provided question, it will try to provide a factual answer to the question"""

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key = key_dict["OPENAI_API_KEY"],)

    if occupation==None:
        
        template = """Given the question {question} I want you to provide me a factual answer.
                      If you cannot answer the question return "not found".Also try to include references ("SOURCES"). """
        tools_for_agent1 = [
            Tool(
                name="Vector Database lookup",
                func=query_vector_database,
                description="queries a vector database for a given question and returns the most related answer along with the sources used in the answer if it is found",
            )]

        agent = initialize_agent(tools_for_agent1, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True ) #verbose helps us see the reasoning and the subtasks the llm have made      

        prompt_template = PromptTemplate(input_variables=["question"], template=template)

        qna_answer= agent.run(prompt_template.format_prompt(name_of_person=name))




    return qna_answer
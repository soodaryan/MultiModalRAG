import os 
import requests

from langchain_core.outputs import GenerationChunk
from langchain_core.language_models.llms import LLM

from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from dotenv import load_dotenv

load_dotenv()
ON_DEMAND_API_KEY= os.getenv("ON_DEMAND_API_KEY")
print(ON_DEMAND_API_KEY)

def get_llm_response(
    query, 
    api_key, 
    external_user_id = "user",
    model_name = "predefined-openai-gpt4o"):
    """
    Generates reponse from ondemand hehe 
    """
    create_session_url = 'https://api.on-demand.io/chat/v1/sessions'
    create_session_headers = {
        'apikey': api_key
    }
    create_session_body = {
        "pluginIds": [],
        "externalUserId": external_user_id
    }

    response = requests.post(create_session_url, headers=create_session_headers, json=create_session_body)
    response_data = response.json()

    session_id = response_data['data']['id']

    query_body = {
        "endpointId": model_name,
        "query": query,
        "pluginIds": [],
        "responseMode": "sync"
    }

    submit_query_url = f'https://api.on-demand.io/chat/v1/sessions/{session_id}/query'
    submit_query_headers = {
        'apikey': api_key
    }

    query_response = requests.post(submit_query_url, headers=submit_query_headers, json=query_body)
    query_response_data = query_response.json()

    try : 
        """
        Successful api call
        """
        return query_response_data["data"]["answer"]
    except : 
        """
        Unsuccessful api call - might be due to Server Error
        """
        return query_response_data
    
def chat(llm,query) :
    return get_llm_response(query,
                               api_key = ON_DEMAND_API_KEY,
                               external_user_id = "user") 


class OnDemandLLM(LLM):
    """
    This is the main wrapper function for OnDemand API call 
    """
    api_key: str
    external_user_id: str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return get_llm_response(prompt,
                               api_key = self.api_key,
                               external_user_id = "user")
        

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "OnDemandChatModel",
            "api_key": self.api_key,
            "external_user_id": self.external_user_id,
        }

    @property
    def _llm_type(self) -> str:
        return "ondemand"

def initialize_onDemand(user_id = "user") : 
    model = OnDemandLLM(
        api_key=ON_DEMAND_API_KEY,
        external_user_id="user"
    )
    return model

def main() : 
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Instantiate your custom LLM
    model = initialize_onDemand()
    
    prompt = ChatPromptTemplate.from_template("Hi, How are you {user}?")

    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"user" : "aryan"}))


if __name__ == "__main__":
    main()
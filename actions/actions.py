from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
from elasticsearch import Elasticsearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# dictionary used to store the sender id as a value
# and corresponding username
senders = dict()


THRESHOLD = 0.315

def generate_gpt_output(redis_answer, user_question):
    """
        Generate a rephrased version of the redis_answer using GPT-3.5-turbo.

        Args:
            redis_answer (str): The answer from the Redis semantic search.
            user_question (str): The question asked by the user.

        Returns:
            str: Rephrased version of the redis_answer using GPT-3.5-turbo.
    """

    # Init LLM and embeddings model
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo",
                          temperature=0.7,
                          openai_api_base='https://XXXX',
                          openai_api_type="azure",
                          openai_api_key="XXXX",
                          openai_api_version="2023-05-15")

    system_message = "You are an AI assistant that given a question from a user and a response from a database " \
                     "you are required to rephrase it in 20 words max " \
                     "you are required then to translates it to the language of the user question language " \
                     "you are required not to deviate from the answer of the database " \
                     "the answer is : "+redis_answer

    # store the prompt in a chat prompt object to be given to the model
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # pass the prompt object to the model
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    # pass the user question
    result = chain.run(user_question)

    # print(result)
    return result

class ActionPetFact(Action):
    """
    Provide a random pet fact to the user.
    """

    def name(self) -> Text:
        return "provide_pet_fact"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # check if the username is recorded
        if tracker.sender_id in senders:
            # record the username to a variable
            sender_name = senders[tracker.sender_id]
        else:
            # assign sender id as the username
            sender_name = tracker.sender_id

        # utter the username
        msg = "Sure thing " + sender_name + ", Give me a second."
        dispatcher.utter_message(text=msg)

        pet_url = {
            'dog': "https://dog-facts-api.herokuapp.com/api/v1/resources/dogs?number=1",
            'cat': "https://meowfacts.herokuapp.com/"
        }

        # get the latest entity for pet, is it dog or cat?
        current_pet = next(tracker.get_latest_entity_values("pet"), None)

        if not current_pet:
            msg = "Please specify the type of pet you want "
            dispatcher.utter_message(text=msg)
            return []

        response = requests.get(pet_url[current_pet])
        if response.status_code != 200:
            msg = "could not fetch url"
            dispatcher.utter_message(text=msg)
            return []

        response_dic = response.json()
        if current_pet == 'dog':
            fact = response_dic["fact"]
        else:
            fact = response_dic['data'][0]


        dispatcher.utter_message(text=fact)

        return []


class ActionRandomMovie(Action):
    """
    Call an API that returns a random movie image and it's title
    """
    def name(self) -> Text:
        return "provide_random_movie"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # check if the username is recorded
        if tracker.sender_id in senders:
            # record the username to a variable
            sender_name = senders[tracker.sender_id]
        else:
            # assign sender id as the username
            sender_name = tracker.sender_id

        # utter the username
        msg = "Sure thing " + sender_name + ", Give me a second."
        dispatcher.utter_message(text=msg)

        url = "https://moviesdatabase.p.rapidapi.com/titles/random?list=top_rated_250"
        headers = {
            "X-RapidAPI-Key": "XXXX",
            "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            msg = "could not fetch url"
            dispatcher.utter_message(text=msg)
            return []

        # extract the poster image and the caption (title)
        response = response.json()
        movie_img_url = response['results'][0]['primaryImage']['url']
        movie_caption = response['results'][0]['primaryImage']['caption']['plainText']

        dispatcher.utter_message(image=movie_img_url, text=movie_caption)

        return []


class ActionCast(Action):
    """
    Search the database for movies of the same title and return the cast.
    """
    def name(self) -> Text:
        return "provide_cast"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # connect to localhost database
        es = Elasticsearch("http://localhost:9200/")

        # get current movie entity
        current_movie = next(tracker.get_latest_entity_values("movie"), None)

        # if no movie entity is found
        if not current_movie:
            msg = "Please specify the movie"
            dispatcher.utter_message(text=msg)
            return []

        # search in es for a movie of matching title
        response = es.search(
            index="netflix",
            query={
                "bool": {
                    "must": [],
                    "filter": [
                        {
                            "exists": {
                                "field": "title"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "title": current_movie
                            }
                        }
                    ]
                }
            },
        )

        # extract the movie title that matches the query the most and extract the cast
        movie_cast = response["hits"]["hits"][0]["_source"]["cast"]

        dispatcher.utter_message(text=movie_cast)

        return []


class ActionDescription(Action):
    """
    Search the database for movies of the same title and return the description.
    """

    def name(self) -> Text:
        return "provide_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # connect to localhost database
        es = Elasticsearch("http://localhost:9200/")

        # get current movie entity
        current_movie = next(tracker.get_latest_entity_values("movie"), None)

        # if no movie entity is found
        if not current_movie:
            msg = "Please specify the movie"
            dispatcher.utter_message(text=msg)
            return []

        # extract the movie title that matches the query the most and extract the description
        response = es.search(
            index="netflix",
            query={
                "bool": {
                    "must": [],
                    "filter": [
                        {
                            "exists": {
                                "field": "title"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "title": current_movie
                            }
                        }
                    ]
                }
            },
        )

        # extract the movie title that matches the query the most and extract the description
        movie_description = response["hits"]["hits"][0]["_source"]["description"]

        dispatcher.utter_message(text=movie_description)

        return []


class ActionMoviesByActor(Action):
    """
    Search the database for movies of the same actor.
    """

    def name(self) -> Text:
        return "provide_movies_by_actor"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # connect to localhost database
        es = Elasticsearch("http://localhost:9200/")

        # get movie actor entity
        current_actor = next(tracker.get_latest_entity_values("actor"), None)

        # if no movie entity is found
        if not current_actor:
            msg = "Please specify the actor's name"
            dispatcher.utter_message(text=msg)
            return []

        # extract the movie contains the actor name
        response = es.search(
            index="netflix",
            size=20,
            query={
                "bool": {
                    "must": [],
                    "filter": [
                        {
                            "exists": {
                                "field": "cast"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "cast": current_actor
                            }
                        }
                    ]
                }
            },
        )

        # record all valid movie titles (i.e. with scores more than zero) in a string "movie_titles
        hits = response["hits"]["hits"]
        movie_titles = ""
        for hit in hits:
            if hit["_score"] > 0:
                movie_titles += ", " + hit["_source"]["title"]

        # if there is no movie titles found
        if movie_titles is None:
            dispatcher.utter_message(text="No movies found for " + current_actor)
            return []

        dispatcher.utter_message(text=movie_titles)

        return []


class ActionAffirmName(Action):
    """
    Save the name of the user along with sender_id in the senders dictionary
    to be later used in concurrent conversations with different users.
    """

    def name(self) -> Text:
        return "affirm_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # get the sender id
        sender_id = tracker.sender_id

        # get the name of the user
        name_from_user = next(tracker.get_latest_entity_values("user_name"), None)

        # if a name could not be found, record the sender_id as the name
        if name_from_user:
            senders[sender_id] = name_from_user
        else:
            senders[sender_id] = sender_id


        msg = senders[sender_id]+", How can i help you?"

        dispatcher.utter_message(text=msg)

        return []


class ActionFallback(Action):
    # Executed if the nlu can not determine the intent of the user from his message

    """
    Fallback action executed when intent can't be determined.
    Semantic search is used to search the user message in a redis database containing
    FAQs about the company (Amazon Sage Maker)
    """

    def name(self) -> Text:
        return "action_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # connect to the ada embeddings model
        embeddings_model = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_base="https://XXXX",
            openai_api_type="azure",
            openai_api_key="XXXX"
        )

        # loading the FAQs document files
        data = ""
        try:
            loader = TextLoader("XXXX/Amazon_sagemaker_doc.txt")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            data = text_splitter.split_documents(documents)
        except ValueError:
            print('Decoding JSON has failed')
        # pprint(data[0])

        # storing data as embeddings in Redis
        rds = Redis.from_existing_index(
            embeddings_model, redis_url="redis://localhost:6379", index_name="Test1"
        )

        # preforming similarity search
        query = tracker.latest_message['text']
        results_and_score = rds.similarity_search_with_score(query)
        reply, score = results_and_score[0][0].page_content, results_and_score[0][1]
        print(score)

        if score < THRESHOLD:
            # use gpt to generate answer
            msg = generate_gpt_output(reply, tracker.latest_message['text'])
        else:
            msg = "I didn't understand that, Can you please rephrase"

        dispatcher.utter_message(text=msg)
        return[]

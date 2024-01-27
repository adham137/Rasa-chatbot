# ChatBot with RASA and OpenAI Integration

## Overview

This project implements a versatile chatbot application using the RASA framework in Python, designed to provide information about movies on the Netflix website and answer queries related to Amazon SageMaker. Leveraging RASA's natural language understanding capabilities, the chatbot is trained with a customized NLU configuration defined in `nlu.yml`, along with rules specified in `rules.yml` and conversational flows outlined in `stories.yml`.

## Features

1. **Netflix Movie Queries**: The chatbot utilizes a custom NLU unit to answer user queries about movies on the Netflix website. The training data is sourced from the `netflix_titles.csv` dataset.

2. **Amazon SageMaker FAQ Integration**: Integration with the Langchaing framework allows the chatbot to connect to OpenAI. The `Amazon_sagemaker_doc.txt` file, containing FAQs about Amazon SageMaker, is embedded using OpenAI's embedding model. The resulting embeddings are stored in a Redis database for efficient retrieval.

3. **Multilingual Support**: The chatbot supports multiple languages. When a user interacts with the chatbot in Arabic, the response is generated in Arabic, and likewise for English. This is achieved by dynamically selecting the language that matches the user's input.

4. **Facebook Integration**: The chatbot is seamlessly connected to a Facebook page, enabling it to interact with users on the platform. This enhances accessibility and allows users to engage with the chatbot through their preferred channel.

## Project Structure

Due to the large size of the entire project, only modified files have been uploaded. Key files include:
- `nlu.yml`: Customized natural language understanding configurations.
- `rules.yml`: Rule-based responses for specific chatbot interactions.
- `stories.yml`: Sequences of user inputs and expected bot responses for training.
- `netflix_titles.csv`: Dataset for training the chatbot's movie-related queries.
- `actions.py`: customized actions depending of the user input understood by the nlu.
- `config.yml`: rasa model configuration settings.
- `Amazon_sagemaker_doc.txt`: FAQs about Amazon SageMaker for embedding.

## Usage

To run the chatbot locally, follow these steps:

1. Set up a RASA environment and install dependencies.
2. Train the chatbot using the provided training files.
3. Connect the chatbot to a Facebook page for broader user engagement.
4. Ensure the necessary Redis database is set up for storing document embeddings.

Feel free to explore the project and contribute to its development!

## Dependencies

- RASA Framework
- Langchaing Framework
- OpenAI GPT-3.5 Turbo Model
- Redis Database
- ElasticSearch Database


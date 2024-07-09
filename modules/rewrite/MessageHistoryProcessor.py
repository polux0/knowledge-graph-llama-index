from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI

#TODO: Modularize access to environment variables
from dotenv import load_dotenv
import os
load_dotenv()
#TODO: Modularize instantiation of language models

class MessageHistoryProcessor:
    """
    A class to process message history and generate contextualized questions.

    Attributes:
        es_client: The Elasticsearch client used to retrieve chat history.
        chat_id: The ID of the chat to retrieve history for.
        history_length: The number of messages to retrieve from the chat history.
        is_test: A flag indicating whether to use test data instead of retrieving actual data.
        contextualize_llm: The language model used for contextualizing questions.
        contextualize_q_system_prompt: The system prompt used for contextualizing questions.
        contextualize_q_prompt: The prompt template used for contextualizing questions.
        chain: The processing chain used to generate responses.
    """

    def __init__(self, es_client, chat_id, history_length=5, is_test=False):
        """
        Initializes the MessageHistoryProcessor with the given parameters.

        Args:
            es_client: The Elasticsearch client used to retrieve chat history.
            chat_id: The ID of the chat to retrieve history for.
            history_length: The number of messages to retrieve from the chat history.
            is_test: A flag indicating whether to use test data instead of retrieving actual data.
        """
        self.contextualize_llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), 
            model_name="gpt-3.5-turbo"
        )
        self.es_client = es_client
        self.chat_id = chat_id
        self.history_length = history_length
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it IF NEEDED and otherwise return it AS IS, please!"
        )
        self.contextualize_q_prompt = self.create_prompt_template()
        self.chain = self.create_chain()
        self.is_test = is_test

    def create_prompt_template(self):
        """
        Creates the prompt template for contextualizing questions.

        Returns:
            ChatPromptTemplate: The prompt template used for contextualizing questions.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def create_chain(self):
        """
        Creates the processing chain combining the prompt template and language model.

        Returns:
            Chain: The processing chain used to generate responses.
        """
        return self.contextualize_q_prompt | self.contextualize_llm

    def retrieve_history(self):
        """
        Retrieves the chat history.

        If `is_test` is True, returns a predefined set of messages. Otherwise, retrieves
        the chat history from Elasticsearch.

        Returns:
            list: The chat history messages.
        """
        if self.is_test:
            return [
                HumanMessage(content='hi!'),
                AIMessage(content='what\'s up?'),
                HumanMessage(content='I need some help with my project.'),
                AIMessage(content='Sure, I\'d be happy to help! What do you need assistance with?'),
                HumanMessage(content='What is Task Decomposition?'),
                AIMessage(content='Task decomposition involves breaking down a complex task into smaller and simpler steps to make it more manageable and easier to accomplish. This process can be done using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide the model in breaking down tasks effectively. Task decomposition can be facilitated by providing simple prompts to a language model, task-specific instructions, or human inputs.'),
            ]
        else:
            return self.es_client.retrieve_telegram_history(self.chat_id, self.history_length)

    def process_message(self, user_message):
        """
        Processes a user message by contextualizing it based on chat history.

        Args:
            user_message: The user message to process.

        Returns:
            Response: The processed and contextualized response.
        """
        history = self.retrieve_history()
        response = self.chain.invoke(
            {
                "chat_history": history,
                "input": [HumanMessage(content=user_message)],
            }
        )
        return response.content


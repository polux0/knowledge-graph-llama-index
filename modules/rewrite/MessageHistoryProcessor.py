from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI

#TODO: Modularize instantiation of language models
#TODO: Remove after testing 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI

from utils.environment_setup import load_environment_variables

#TODO: Remove after testing
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

    def __init__(self, es_client, chat_id, history_length=1, is_test=False):
        """
        Initializes the MessageHistoryProcessor with the given parameters.

        Args:
            es_client: The Elasticsearch client used to retrieve chat history.
            chat_id: The ID of the chat to retrieve history for.
            history_length: The number of messages to retrieve from the chat history.
            is_test: A flag indicating whether to use test data instead of retrieving actual data.
        """
        self.env_vars = load_environment_variables()
        #TODO: Modularize instantiation of language models
        self.contextualize_llm = ChatOpenAI(
            openai_api_key=self.env_vars["OPENAI_API_KEY"], 
            model_name="gpt-3.5-turbo",
            temperature = 0,
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
        # self.contextualize_q_system_prompt = (
        #     "A 'vague follow-up question' is a question in which the user requests additional information or clarification without specifying particular details or aspects." 
        #     "A 'specific follow-up question' is a question that includes key terms or references from the previous question and answer, indicating a direct continuation of the previous topic."
            
        #     "### Chain of Thoughts:"
            
        #     "If the new question is a 'vague follow-up question':"
        #     "1. SEARCH the previous question and answer for relevant context."
        #     "2. IDENTIFY the main topics or details the user was interested in."
        #     "3. ENRICH the new question with this context to create a more specific and detailed prompt."
        #     "If the new question is a 'specific follow-up question':"
        #     "1. CROSS-REFERENCE key terms in the new question with the previous question and answer."
        #     "2. If a fuzzy match is found, RETRIEVE relevant information from the previous exchange."
        #     "3. ENRICH the new question with this information to form a comprehensive and contextually accurate prompt."
        #     "If the new question does not relate to the previous question and answer:"
        #     "1. LEAVE the new question AS IS and handle it independently."

        #     "### What Not To Do:"

        #     "- DO NOT ASSUME context for a vague follow-up question without searching the previous exchange."
        #     "- DO NOT ENRICH a question with unrelated information."
        #     "- AVOID treating unrelated questions as related; handle them independently."
        #     "- NEVER IGNORE key terms that indicate a specific follow-up question."
        # )
        # self.contextualize_q_system_prompt = (
        #     "A 'vague follow-up question' is a user_input in which the user requests additional information or clarification without specifying particular details or aspects. For example, 'tell me more'." 
        #     "A 'specific follow-up question' is a user_input that includes key terms or references from the previous question and answer, indicating a direct continuation of the previous topic."
            
        #     "### Chain of Thoughts:"
            
        #     "If the user_input is a 'vague follow-up question':"
        #     "1. SEARCH the chat_history and IDENTIFY the most recent topic or theme from the question and answer"
        #     "2. ENRICH the user_input with the topic or theme that you have identified. For example 'tell me more about xyz'."
        #     "If the new question is a 'specific follow-up question':"
        #     "1. CROSS-REFERENCE key terms in the user_input with the information in the chat_history."
        #     "2. If a fuzzy match is found, RETRIEVE relevant information from the chat_history."
        #     "3. ENRICH the user_input with this information to form a comprehensive and contextually accurate prompt."
        #     "If no fuzzy match us found, the user_input does not relate to the chat_history:"
        #     "1. LEAVE the new question AS IS and handle it independently."

        #     "### What Not To Do:"

        #     "- DO NOT ASSUME context for a vague follow-up question without searching the previous exchange."
        #     "- DO NOT ENRICH a question with unrelated information."
        #     "- AVOID treating unrelated questions as related; handle them independently."
        #     "- NEVER IGNORE key terms that indicate a specific follow-up question."
        # )

        # Another prompt
        # self.contextualize_q_system_prompt = (
        #     "A 'vague follow up question' is when the user asks for more details, or asks you "
        #     "to tell them more, without being specific. If the new question is a 'vague follow "
        #     "up question', then search the previous question and answer for additional context "
        #     "about what the user is interested in, and enrich the new question with this info to "
        #     "create a new prompt. If the new question is not a 'vague follow up question', ascertain "
        #     "whether it is a 'specific follow up question' by cross-referencing key terms in the new "
        #     "question with the previous question and answer. If there is at least a fuzzy match, assume "
        #     "that the new question is a precise follow up question, retrieve relevant information from the "
        #     "previous question and answer, and use it to enrich the new question to form a new prompt. If "
        #     "there is no evidence that the new question relates to the previous question and answer in any "
        #     "meaningful way, just leave the new question AS IS."
        # )
        self.is_test = is_test
        self.contextualize_q_prompt = self.create_prompt_template()
        self.chain = self.create_chain()

    def create_prompt_template(self):
        """
        Creates the prompt template for contextualizing questions.

        Returns:
            ChatPromptTemplate: The prompt template used for contextualizing questions.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{user_input}"),
            ]
        )
        return prompt
    
    def create_prompt_alternative(self):
        prompt = PromptTemplate(
        input_variables=['question','chat_history',],
        template = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it IF NEEDED and otherwise return it AS IS, please!\n"
            "Question: {question}\n"
            "Chat history: {chat_history}\n"
            "Answer:"
            )
        )
        return prompt

    def create_chain(self):
        """
        Creates the processing chain combining the prompt template and language model.

        Returns:
            Chain: The processing chain used to generate responses.
        """
        return self.contextualize_q_prompt | self.contextualize_llm

    def create_chain_alternative(self):
        prompt = self.create_prompt_alternative()
        chat_history = self.retrieve_history()
        rag_chain = (
            {"chat_history": chat_history , "question": RunnablePassthrough()}
            | prompt
            | self.contextualize_llm
            | StrOutputParser()
        )
        return rag_chain

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
            return self.es_client.retrieve_telegram_history_different_formatting(self.chat_id, self.history_length)

    def process_message(self, user_input):
        """
        Processes a user message by contextualizing it based on chat history.
        Args:
            user_input: The user message to process.

        Returns:
            Response: The processed and contextualized response.
        """
        chat_history = self.retrieve_history()
        print("retrieved history: \n", chat_history)
        response = self.chain.invoke(
            {
                "chat_history": chat_history,
                "user_input": [HumanMessage(content=user_input)],
            }
        )
        return response.content
    
    def generate_prompt(self, user_input, chat_history):
        prompt = f"""
        "Your role is to compare the {user_input} and the {chat_history} and determine whether the {user_input} needs to be enriched with the {chat_history} or not, and then enrich the {user_input} if yes."

        "### Definitions:"
        "A 'vague follow-up' is a {user_input} in which the user requests additional information or clarification without being specific. For example, 'tell me more'."
        "A 'specific follow-up question' is a {user_input} that includes key terms or references that can be found in the {chat_history}, indicating a direct continuation of the previous topic."
        "A 'new topic' is where the {user_input} is unrelated to the {chat_history} as it has zero terms or themes that can be cross-referenced with the {chat_history}"
        
        "### Chain of Thoughts:"
        "1. DETERMINE whether the {user_input} is a 'vague follow-up question' or NOT.
        "2. IF NOT, CROSS-REFERENCE key terms or themes in the {user_input} with the information in the {chat_history}."
        "3. DETERMINE whether the {user_input} is a 'specific follow-up question', or a 'new topic'.
        
        "If the {user_input} is a 'vague follow-up question':"
        "1. SEARCH the {chat_history} and IDENTIFY the most recent topic or theme from the question and answer in the {chat_history}"
        "2. ENRICH the {user_input} with the topic or theme that you have identified from the {chat_history}. For example 'tell me more about xyz'."
        "If the {user_input} is a 'specific follow-up question':"
    
        "1. ENRICH the {user_input} with the relevant terms of themes from the {chat_history} to form a comprehensive and contextually accurate prompt."
        "If the {user_input} is a 'new topic':"

        "1. Leave the {user_input} as found. Do not change the {user_input}.
        "### What Not To Do:"

        "- DO NOT ASSUME context for a 'vague follow-up question' without searching the {chat_history}"
        "- DO NOT ENRICH a {user_input} with unrelated information."
        "- NEVER IGNORE key terms or themes that indicate a specific follow-up question." 
        "- DO NOT modify a {user_input} if it is a 'new topic'"
        "- DO NOT answer the question. Your role is to reformulate the question, ONLY if needed."
        """
        print(f"!prompt: \n", prompt)
        return prompt
    
    def generate_prompt1(self, user_input, chat_history):
            prompt = f"""
            Your role is to analyze the user_input: "{user_input}" in the context of the chat_history: "{chat_history}" and determine whether the user_input needs to be enriched with information from the chat_history.

            Follow these steps:

            1. Determine the type of user_input:
            a) Vague follow-up (e.g., "Tell me more")
            b) Specific follow-up (contains key terms or themes from chat_history)
            c) New topic (unrelated to chat_history)

            2. Process the user_input based on its type:
            a) For vague follow-ups:
                - Identify the most recent relevant topic from chat_history
                - Enrich user_input with this topic (e.g., "Tell me more about [topic]")
            b) For specific follow-ups:
                - Enrich user_input with relevant context from chat_history
            c) For new topics:
                - Do not modify the user_input

            3. Output:
            - If enriched, return the modified user_input
            - If not enriched, return "UNCHANGED: [original user_input]"

            Important:
            - Do not assume context for vague follow-ups without evidence from chat_history
            - Do not connect unrelated new topics to previous context
            - Do not answer the question; only reformulate if needed

            Provide your reasoning before giving the final output.
            """
            return prompt
    
    def test_alternative(self, user_input: str):

        api_key = self.env_vars['OPENAI_API_KEY']  # Correct way to get the API key

        client = OpenAI(api_key=api_key)  # Pass the API key if needed

        history = self.es_client.retrieve_telegram_history_different_formatting(self.chat_id, self.history_length)

        print(f"history: \n", history)

        chat_history = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in history])

        print(f"chat_history: \n", chat_history)
        # Attempt 1
        # prompt = f"""
        # Given a chat history and the latest user question which might reference context in the chat history,
        # formulate a standalone question which can be understood without the chat history.
        # Do NOT answer the question, just reformulate it IF NEEDED and otherwise return it AS IS, please!

        # Question: {question}
        # Chat history: {formatted_history}
        # Reformulated Question:

        # """
        # Attempt 2
        # prompt = f"""
        # Here is a chat history and a user's latest question. 
        # Based on the chat history and the question, reformulate the question to be standalone and clear, incorporating relevant context.

        # Chat history:
        # {formatted_history}

        # Latest Question: {question}

        # Reformulated Question:
        prompt = self.generate_prompt1(user_input=user_input, chat_history=chat_history)
        print(f"final prompt that is being fed to the llm", prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            # model="gpt-4o",	
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        reformulated_question = response.choices[0].message.content
        
        # If the response starts with "UNCHANGED:", return the original user_input
        if reformulated_question.startswith("UNCHANGED:"):
            return user_input
        else:
            return reformulated_question

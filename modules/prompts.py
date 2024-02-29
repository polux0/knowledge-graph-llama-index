TEMPLATES = {
    "default": """<|system|>Please check if the following pieces of context has any mention of the keywords provided in the Question.
                                If not then don't know the answer, just say that you don't know.
                                Stop there.
                                Please do not try to make up an answer.</s>
                  <|user|>
                                Question: {query}
                                Helpful Answer:
                                </s>""",
    "context": """<|system|>Please check if the following pieces of context has any mention of the keywords provided in the Question.
                                If yes, please provide the context.
                                If not then don't know the answer, just say that you don't know.
                                Stop there.
                                Please do not try to make up an answer.</s>
                      Question: {query}
                      Answer: </s>""",
    
}
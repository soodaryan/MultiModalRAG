def get_qa_gen_prompt(): 
    QA_generation_prompt = """
    Your task is to write a factoid question and an answer given a context.
    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
    This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

    Provide your answer as follows:

    Output:::
    Factoid question: (your factoid question)
    Answer: (your answer to the factoid question)

    Now here is the context.

    Context: {context}\n
    Output:::"""

    return QA_generation_prompt




# def get_critique_prompts() :


#     question_groundedness_critique_prompt = """
#     You will be given a context and a question.
#     Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
#     Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

#     Provide your answer as follows:

#     Answer:::
#     Evaluation: (your rationale for the rating, as a text)
#     Total rating: (your rating, as a number between 1 and 5)

#     You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

#     Now here are the question and context.

#     Question: {question}\n
#     Context: {context}\n
#     Answer::: """

#     question_relevance_critique_prompt = """
#     You will be given a question.
#     Your task is to provide a 'total rating' representing how useful this question can be to Eligible bidders for NIT Jalandhar e-tenders..
#     Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

#     Provide your answer as follows:

#     Answer:::
#     Evaluation: (your rationale for the rating, as a text)
#     Total rating: (your rating, as a number between 1 and 5)

#     You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

#     Now here is the question.

#     Question: {question}\n
#     Answer::: """

#     question_standalone_critique_prompt = """
#     You will be provided with a question.
#     Your task is to evaluate how context-independent this question is and provide a 'total rating' on a scale of 1 to 5 based on the following criteria:

#     1. A rating of 1 means the question is highly dependent on external context or additional information to be understood.
#     2. A rating of 2 means the question can be understood partially but still requires significant external context to provide a full answer.
#     3. A rating of 3 means the question is moderately clear but might need some external clarification to be fully self-contained.
#     4. A rating of 4 means the question is mostly self-contained and clear, with only minimal external clarification needed.
#     5. A rating of 5 means the question is entirely self-contained, clear, and understandable without requiring any external context.

#     ### Guidelines for evaluation:
#     - Questions with minor references to external context (e.g., "in the summary" or "in the document") can be rated as **2 or 3** if they are still somewhat understandable.
#     - Questions with domain-specific terms (e.g., technical terms or acronyms) can still achieve a **4 or 5** as long as they make sense to someone familiar with the domain.
#     - The focus is on how easily the question can be understood or interpreted, not necessarily how easy it is to answer.
#     - Use intermediate ratings (2, 3, or 4) where questions are not entirely context-dependent but are not fully self-contained.

#     Provide your answer as follows:

#     Answer:::
#     Evaluation: (your reasoning for the rating, written as a short text)
#     Total rating: (your rating, as a number between 1 and 5)

#     You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

#     Now here is the question.

#     Question: {question}\n
#     Answer:::
#     """


#     return {
#         "groundedness":question_groundedness_critique_prompt,
#         "relevance": question_relevance_critique_prompt,
#         "standalone": question_standalone_critique_prompt
#     }



def get_critique_prompts() :


    question_groundedness_critique_prompt = """
    You will be given a context and a question.
    Your task is to evaluate how well the question can be answered using only the provided context. Rate the question on a scale of 1 to 5:

    - 1: The question cannot be answered with the context.
    - 2: The question is only slightly answerable with the context.
    - 3: The question is partially answerable but lacks clarity or relevance to the context.
    - 4: The question is mostly answerable with the context but may require minor assumptions.
    - 5: The question is fully and unambiguously answerable using the context alone.

    Provide your response as follows:
    Answer:::  
    Evaluation: (your reasoning for the rating)  
    Total rating: (your rating, as a number between 1 and 5)  

    Now here is the question and context:  
    Question: {question}  
    Context: {context}  
    Answer:::  
    """

    question_relevance_critique_prompt = """
    You will be given a question.
    Your task is to rate how relevant and useful the question is for its intended purpose or audience (e.g., for retrieving information or aiding decision-making). Provide a total rating on a scale of 1 to 5:

    - 1: Not relevant or useful.
    - 2: Slightly relevant but mostly unhelpful.
    - 3: Moderately relevant and somewhat useful.
    - 4: Very relevant and useful with minor improvements needed.
    - 5: Highly relevant and extremely useful.

    Provide your response as follows:
    Answer:::  
    Evaluation: (your reasoning for the rating)  
    Total rating: (your rating, as a number between 1 and 5)  

    Now here is the question:  
    Question: {question}  
    Answer:::  
    """

    question_standalone_critique_prompt = """
    You will be given a question.
    Your task is to evaluate whether the question is self-contained and can be understood without requiring additional context. Rate the question on a scale of 1 to 5:

    - 1: The question is highly ambiguous or entirely dependent on external context.
    - 2: The question is partially clear but requires significant external context to understand.
    - 3: The question is moderately clear but may need some external clarification.
    - 4: The question is mostly clear and self-contained, with minor ambiguities.
    - 5: The question is fully self-contained, clear, and understandable without additional context.

    Provide your response as follows:
    Answer:::  
    Evaluation: (your reasoning for the rating)  
    Total rating: (your rating, as a number between 1 and 5)  

    Now here is the question:  
    Question: {question}  
    Answer:::  
    """


    return {
        "groundedness":question_groundedness_critique_prompt,
        "relevance": question_relevance_critique_prompt,
        "standalone": question_standalone_critique_prompt
    }

if __name__ == "__main__" :
    from collections import defaultdict
    output = {
        "context": "context",
        "question" : "question"
    }
    prompts = get_critique_prompts()
    for i in prompts : 
        print(prompts[i].format_map(
                        defaultdict(
                            lambda: "", {
                                "context": output.get("context", ""),
                                "question": output.get("question", "")
                                }
                            )
                        ))
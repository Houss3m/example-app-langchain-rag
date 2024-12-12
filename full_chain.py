import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain
from openai import OpenAI
from pydantic import BaseModel


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)
    system_prompt = """
    Based on the context below, answer the question of the user.
    
    Context: {context}
    
    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain

def paraphraser_model(query, response, openai_api_key=None):
    client = OpenAI(api_key=openai_api_key)
    system_prompt = f"""

    if you see that the response is not efficient to the query, you can change it but keeping the same tone. and you may use the below response as an addtional context.
    for example the user may say thanks, but the RAG model won't answer that in a conversational tone, you should engage and change the whole response.
    the users query: {query}

    the RAG model response: 
    {response}
    """
    print("before rephraser")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": response.content
            }
        ]
    )
    return completion
    
def convo_model(query, openai_api_key=None):
    client = OpenAI(api_key=openai_api_key)
    system_prompt="""
    You are a conversational agent, that replied with Cujo's tone, here are some instrucions about Cujo's tone below:
    Some observed Elements of the Tone of Robert “Cujo” Teschner:
    - Conversational Approach: Starts with greetings and informal yet respectful language.
    - Motivational Language: Focus on empowering and encouraging leaders to achieve their goals.
    - Structured Communication: Provides clear points and actionable takeaways, ensuring clarity and purpose.
    - Warm and Inclusive: Uses inclusive phrases like "friends" to build rapport with the audience. 
    you should answer the user's query: {query}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": query
            }
        ]
    )
    return completion

def checker_model(query, openai_api_key=None):
    class Question_eval(BaseModel):
        eval: int
    
    client = OpenAI(api_key=openai_api_key)
    # Simulated client API call
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given the following query, classify the query if it is generic conversational or not. "
                    "1. if the query is classified as a generic conversational query that doesn't require any additional context or a RAG model, return the number 1."
                    "0. if the query is a question that requires additional context and the use of a RAG model, return the number 0."
                    "The final answer should be an integer."
                    "for example: the query is: Thank you for your help."
                    "you should return this only: 1"
                ),
            },
            {
                "role": "user",
                "content": f"Here is the user query that you should classify: {query}",
            },
        ],
        temperature=0,
        response_format=Question_eval,
    )

    

def ask_question(chain, query, openai_api_key=None):
    
    result=checker_model(query, openai_api_key)
    
    if result != None:
        if result==0:
            response = chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "foo"}}
            )
    
            completion=paraphraser_model(query, response, openai_api_key)
    

            print(response.content)


            return completion.choices[0].message
        elif result==1:
            completion=convo_model(query,openai_api_key)
            return completion.choices[0].message



def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

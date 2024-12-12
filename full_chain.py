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


def ask_question(chain, query, openai_api_key=None):
    client = OpenAI(api_key=openai_api_key)

    print("inside ask question")
    print("before chain invoke")
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    print("after chain invoke")
    system_prompt = f"""
    if you see that the response is not efficient to the query, you can change it but keeping the same tone. and you may use the below response as an addtional context.
    for example the user may say thanks, but the RAG model won't answer that in a conversational tone, you should engage and change the whole response.
    the users query: {query}
    
    You are a paraphraser and a tone cloning assistant. below are the instructions you should follow to refine the answer and mimic Cujo's tone. 

    IMPORTANT: YOU MUST PARAPHRASE THE ANSWER BASED ON THE INSTRUCTIONS MENTIONED BELOW:

    Your tone should reflect the inspiring and disciplined approach of Robert "Cujo" Teschner, emphasizing clarity, alignment, and motivation. You use his characteristic style of motivating teams and aligning them to meaningful goals.

    "You are an inspiring and disciplined conversational partner, embodying the motivational and structured tone of Robert “Cujo” Teschner. Your role is to help users apply leadership and planning frameworks effectively while ensuring clarity, alignment, and actionable insights. Speak with energy and focus, balancing encouragement with strategic rigor."

    Some observed Elements of the Tone of Robert “Cujo” Teschner:
    - Conversational Approach: Starts with greetings and informal yet respectful language.
    - Motivational Language: Focus on empowering and encouraging leaders to achieve their goals.
    - Structured Communication: Provides clear points and actionable takeaways, ensuring clarity and purpose.
    - Warm and Inclusive: Uses inclusive phrases like "friends" to build rapport with the audience. 

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

    print(response.content)
    print("after rephraser")

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

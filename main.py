import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    RunContextWrapper,
    function_tool,
    RunConfig,
)
from input_guardrails import malicious_intent_guardrail
from output_guardrails import (
                            pii_output_guardrail, 
                            hallucination_output_guardrail,
                            self_reference_output_guardrail)
import json
from typing import cast
from my_secrets import Secrets
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent
import random
import fitz
import docx
import pandas as pd
import os

secrets = Secrets()

@dataclass
class Developer:
    name:str
    mail:str
    github:str

@function_tool("developer_info")
@cl.step(type="Developer Info")
def developer_info(developer: RunContextWrapper[Developer]) -> str:
    "Returns the name, mail and github of the developer"
    return f"Developer name: {developer.context.name}, Developer mail: {developer.context.mail}, Developer github: {developer.context.github}"


@cl.set_chat_profiles
def chat_profile():
    return [
        cl.ChatProfile(
            name="StudyBuddy",
            markdown_description="Study Buddy AI - Your AI-powered toolkit for smarter, faster studying.",
            icon="/public/reading.svg"
        )
    ]

@cl.set_starters
async def starter():
    return [
        cl.Starter(
            label="Text Summarizer",
            message="Meet your Summarizer – ready to condense content into clear summaries.",
            icon="/public/summarization.svg",
        ),
        cl.Starter(
            label="Concept Explainer",
            message="Need help understanding a topic? Your Concept Explainer is here to assist.",
            icon="/public/discussion.svg",
        ),
        cl.Starter(
            label="Quiz Generator",
            message="Ready to test your knowledge? Let the Quiz Generator create custom quizzes for you.",
            icon="/public/quiz.svg",
        ),
        cl.Starter(
            label="Text Translator",
            message="Your Translator is here – easily convert text between languages.",
            icon="/public/translating.svg"
        ),
        cl.Starter(
            label="Definition Lookup",
            message="Quick definitions at your fingertips – your Definition Assistant is here.",
            icon="/public/definition.svg"
        ),
        cl.Starter(
            label="Flashcard Generator",
            message="Boost your memory with smart flashcards – generated just for you.",
            icon="/public/flash-card.svg",
        ),
        cl.Starter(
            label="Study Scheduler",
            message="Plan smarter, not harder – your Study Scheduler is ready to help.",
            icon="/public/time-management.svg",
        ),
        cl.Starter(
            label="Code Explainer",
            message="Confused by code? The Code Explainer is here to break it down for you.",
            icon="/public/binary-code.svg",
        ),
        cl.Starter(
            label="Math Solver",
            message="Tackle math problems with confidence – your Math Solver is ready.",
            icon="/public/calculating.svg"
        ),
        cl.Starter(
            label="Research Assistant",
            message="Meet your Research Assistant – here to support your research needs.",
            icon="/public/search.svg",
        ),
    ]

@cl.on_chat_start
async def start():
    client = AsyncOpenAI(
        api_key=secrets.gemini_api_key,
        base_url=secrets.gemini_base_url,
    )
    set_default_openai_api("chat_completions")
    set_default_openai_client(client)
    set_tracing_disabled(True)
    model = OpenAIChatCompletionsModel(
        model=secrets.gemini_api_model,
        openai_client=client,
    )

    summarizer_agent = Agent(
        name="Text_Summarizer",
        instructions="""
                    Summarizes input text or the textual content of a supported file (PDF, DOCX, or TXT).

                    - summarize the text in all formats like in bullets , points, neutral or plain
                    - by default summarize the text in default format 
                    - return the output in plain text
                    - summarize the text in 3 to 5 lines if the text provided is of around 20 lines 
                    - summarize to the precise and in one-fourth length of original text
                    - if required length of summarized text can be increased or decreased
                    - do not mix up the name of tool with other available tools
                    - while summarizing the content always keep on the topic do not hallucinate anything from by yourself 
                    """,
        model=model,
        handoff_description="Generates concise summaries from raw text or documents (PDF, DOCX, TXT) with customizable tone and length.",
    )

    explainer_agent = Agent(
        name="Concept_Explainer",
        instructions="""
                    Explains a concept clearly and contextually, using either direct input or content extracted from a file.
                    -if user uploads a file and ask for any concept from the file then explain it if file format is supported
                    - if the file format is not supported then return a unsupported file error
                    - explain concept in plain text and be on topic
                    - always keep on the topic to explain concept which the user asked
                    - do not hallucinate any other concept while explaining the one provided by user
                    - do not mix the tool name with other and use the tool with the requirement of the user
                    - use tool by the requirement of prompt provided by user
                     """,
        model=model,
        handoff_description="Explains complex concepts from text or documents (PDF, DOCX, TXT) with adjustable depth and audience focus.",
    )

    quiz_generator_agent = Agent(
        name="Quiz_Generator",
        instructions="""
                    Generates a customizable quiz on a given topic with flexible question count.
                    - if user mention the difficulty then ok else take the difficulty as moderate by default
                    - the default number of quiz generated is 10 at a time its a default value 
                      if user requires the number of quiz more then this or less then user requirement takes precedence
                    - the format of quiz should be like a statement with four available options and at the end of all the questions give 
                      the right answers for every question with number of question
                    - if user uploads a supported file and ask to generate quiz from it then generate quiz using uploaded file
                    - if file format is unsupported then return a unsupported file error
                    """,
        model=model,
        handoff_description="Generates quizzes on any topic with customizable difficulty, format, and flexible question count (default: 10)",
    )

    translator_agent = Agent(
        name="Text_Translator",
        instructions="""
                    Translate the text from user input or from the supported files provided by the user.
                    - never mixed the tool name with other tools
                    - return the final translated result in plain text
                    - if user uploads a supported format file to translate then translate into the user required file 
                    - if file is unsupported then return an unsupported file format
                    - Translation should be simple and in plain text do not include any source code in it
                    """,
        model=model,
        handoff_description="Translates text or content from files (PDF, DOCX, TXT) into any target language with optional auto-detection.",
    )

    lookup_definition_agent = Agent(
        name="Lookup_Definition_Agent",
        instructions="""
                    Retrieves clear, accurate definitions for terms or concepts, either from direct input or extracted from supported files.
                    - Lookup for the definition of user specified words from supported files
                    - Also can get definition directly from llm
                    - if the definition of user specified word isn't in the file provided by user then ask the user to get it from llm 
                    - if definition is got by llm give the most appropriate definition 
                    - while getting definition from llm:
                        return definition and also its source
                    """,
        model=model,
        handoff_description="Finds accurate definitions from text or documents (PDF, DOCX, TXT) or directly get from llm, with optional context awareness and usage examples.",
    )

    flashcard_generator_agent = Agent(
        name="FlashCard_Generator_Agent",
        instructions="""
                    Generates concise, effective flashcards for active recall and spaced repetition from user input or uploaded study materials.
                    Functionality:
                    - Accept text input or supported file types (PDF, DOCX, TXT) containing study material.
                    - Automatically identify key concepts, definitions, formulas, or facts suitable for flashcards.
                    - Generate flashcards in a "question → answer" format for maximum memory retention.
                    - Ensure that each flashcard is focused, specific, and not overloaded with information.
                    - Group flashcards by topic or section if possible for better organization.
                    - Avoid duplicating content or generating cards from non-educational content.
                    - If the file includes headings or sections, reflect that structure in the flashcards.
                    - Ask the user if they prefer fill-in-the-blank, definition-based, or multiple-choice formats (optional).
                    - Politely request clarification if the content is unclear or too vague for flashcard generation.
                    Supported file types: `.pdf`, `.docx`, `.txt`
                    """,
        model=model,
        handoff_description="Generates smart flashcards from topics or files (PDF, DOCX, TXT) in Q&A, cloze, or definition formats for effective studying.",
    )

    study_schedular_agent = Agent(
        name="Study_Schedular_Agent",
        instructions="""
                Creates a personalized, goal-oriented study schedule based on user input or the content of an uploaded file.
                Functionality:
                - design the most effective and precise schedule tailored to the user's academic needs.
                - collect relevant information from the user (e.g., subjects, goals, deadlines, availability) to generate an optimal schedule.
                - By default, keep Sundays as rest/break days.
                - If the user explicitly requests to include Sunday in the schedule, prioritize their preference.
                - If the user uploads a lecture or class timetable, use that as the foundation and align the study schedule accordingly.
                - Ensure the final schedule is practical, balanced, and achievable.
                Supported file types: .pdf, .docx, .txt
                """,
        model=model,
        handoff_description="Generates personalized study plans from goals or files, based on deadlines, availability, and custom preferences.",
    )

    code_explainer_agent = Agent(
        name="Code_Explainer_Agent",
        instructions="""
                Provides clear, accurate explanations of code based on user input or uploaded files.
                Functionality:
                - Analyze and explain the purpose and logic of the code in simple, understandable terms.
                - Support multiple programming languages such as Python, C++, JavaScript, and others.
                - Accept direct code snippets or code files (e.g., `.py`, `.cpp`, `.js`, `.txt`) for analysis.
                - automatically detect the language of the code if not specified by the user.
                - break down complex logic, loops, conditions, and functions into step-by-step explanations.
                - if the user provides a specific question about a section of the code, focus the explanation accordingly.
                - Clearly differentiate between code comments, syntax, logic flow, and potential issues or optimizations.
                - avoid guessing when the code is ambiguous — instead, ask the user for clarification.
                - always debug the code or suggest the debugged code to user

                Supported file types: `.py`, `.cpp`, `.js`, `.txt`, `.java`, `.c`
                """,
        model=model,
        handoff_description="Explains and optionally debugs code from input or files (e.g., .py, .cpp, .js). Supports logic, structure, performance, or bug analysis.",
    )

    math_solver_agent = Agent(
        name="Math_Problem_Solver",
        instructions="""
                    Solves math problems step-by-step, either from user input or extracted from uploaded files.

                    Functionality:
                    - Accept direct math questions (text) or files containing math problems (PDF, DOCX, TXT).
                    - Accurately identify the type of math problem: algebra, calculus, trigonometry, arithmetic, statistics, geometry, etc.
                    - Solve the problem with detailed, logical steps — not just the final answer.
                    - Clearly explain formulas, substitutions, simplifications, and derivations used in solving.
                    - Provide graph-based insights if needed (e.g., plotting functions or visualizing equations).
                    - If the input contains multiple problems, handle them one at a time and provide clear separation.
                    - If the file contains both theory and problems, focus only on solving the problems.
                    - Politely ask for clarification if a problem is incomplete, ambiguous, or requires assumptions.

                    Supported file types: `.pdf`, `.docx`, `.txt`
                    """,
        model=model,
        handoff_description="Solves math problems with step-by-step reasoning, plotting, and support for algebra, calculus, linear algebra, and more.",
    )

    research_assistant_agent = Agent(
        name = "Research_Assistant",
        instructions="""
                Helps users gather, organize, and summarize high-quality research materials on a given topic.

                Functionality:
                - Accepts research queries as direct input or extracts topics from uploaded files (PDF, DOCX, TXT).
                - Searches for credible academic sources, journal articles, reports, and authoritative websites.
                - Summarizes key findings, arguments, or data from retrieved sources in clear, concise language.
                - Can highlight pros/cons, gaps in research, or areas for further investigation.
                - Organizes results thematically or chronologically if needed.
                - Supports generating brief literature reviews or annotated bibliographies.
                - Can cite sources in standard formats (APA, MLA, Chicago) upon request.
                - Handles interdisciplinary topics by identifying relevant cross-domain materials.
                - If file input contains specific instructions (e.g., focus questions, constraints), integrates them into the research.
                - Politely requests clarification if the topic is too broad, vague, or lacks context.
                Supported file types: `.pdf`, `.docx`, `.txt`
                """,
                model=model,
                handoff_description="Finds and summarizes credible sources to support deep research.",
    )

    agent = Agent(
        name="StudyBuddyAI",
        instructions="""You are a helpful and knowledgeable study assistant and uses available tools to give the most precise answers. 
                    Your goal is to provide clear, precise, and well-informed answers to questions related to academic learning, technology, science, programming, and general knowledge.
                    You are allowed to answer:
                    - Developer info 
                    - basic discussion 
                    - Educational questions (e.g., "What is photosynthesis?", "What is AI?")
                    - Technical or programming questions (e.g., "How do loops work in Python?")
                    - Study-related topics from any academic field (e.g., math, history, biology)
                    - Research-related or factual queries (e.g., "What is the capital of France?")
                    - if the user asks anything about developer info then use the developer info tool and get the info using it
                    You should politely decline to respond to:
                    - Personal, social, or entertainment-focused topics
                    - Inappropriate, harmful, or off-topic content
                    - Requests unrelated to learning or educational growth
                    Avoid mixing the name of tools and use the tools that user asks to use or by the requirement of the prompt provided
                    by user.
                    Stay helpful, focused, and safe.""",
        model=model,
        handoffs=[summarizer_agent,
                  explainer_agent,
                  quiz_generator_agent,
                  translator_agent,
                  lookup_definition_agent,
                  code_explainer_agent,
                  study_schedular_agent,
                  math_solver_agent,
                  flashcard_generator_agent,
                  research_assistant_agent,],
        tools=[developer_info],
        input_guardrails=[malicious_intent_guardrail],
        output_guardrails=[
                           pii_output_guardrail,
                           hallucination_output_guardrail,
                           self_reference_output_guardrail,],
    )

    dev = Developer(
        name="Muhammad Usman",
        mail="muhammadusman5965etc@gmail.com",
        github="https://github.com/MuhammadUsmanGM"
    )

    cl.user_session.set("dev", dev)
    cl.user_session.set("agent",agent)
    cl.user_session.set("history",[])

@cl.on_message
async def main(message: cl.Message):
    agent = cast(Agent, cl.user_session.get("agent"))
    history: list = cl.user_session.get("history") or []
    dev = cl.user_session.get("dev")

    def get_thinking_message():
        messages = [
            "🧠 Thinking... Just a moment while I gather your answer!",
            "📚 Processing your request... Let me put my study cap on!",
            "🤓 Crunching data and decoding knowledge... hang tight!",
            "🧠 Analyzing your input and retrieving the best response...",
            "⏳ Working on it...",
            "🔍 Gathering insights... this won't take long!",
            "💡 One sec, the neurons are firing!",
            "Give me a moment — I'm looking into it... 📚",
            "Thinking this through like a top student... ✍️",
            "Let me gather the best explanation for you... 🧠✨",
            "Crunching some knowledge for you... ⏳",
            "Analyzing this like a pro — hang tight! 🔍",
            "Almost there... just connecting the academic dots! 📖",
            "Sharpening my pencils... and my thoughts! ✏️💭",
            "Solving this puzzle one piece at a time... 🧩",
            "Flipping through mental textbooks... 📘📘📘",
            "Checking my notes on that topic... 📝",
            "Calculating the smartest answer for you... 🧮",
            "Channeling my inner tutor — just a sec! 🎓",
            "Compiling your custom study guide... ⌛",
        ]
        return random.choice(messages)

    file_content = None
    if message.elements:
        uploaded_file = message.elements[0]
        file_path = uploaded_file.path
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()

        try:
            if ext in [".txt", ".py", ".cpp", ".cc", ".csv"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()

            elif ext == ".docx":
                from docx import Document
                doc = Document(file_path)
                file_content = "\n".join([para.text for para in doc.paragraphs])

            elif ext == ".pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                file_content = "\n".join(
                    [page.extract_text() for page in reader.pages if page.extract_text()]
                )

            else:
                await cl.Message(content=f"❌ Unsupported file type: {ext}").send()
                return

        except Exception as e:
            await cl.Message(content=f"❌ Could not read file: {str(e)}").send()
            return

    # 🧠 Thinking message to display while llm generating response
    thinking_msg = cl.Message(content=get_thinking_message())
    await thinking_msg.send()

    # added user input to history
    history.append({
        "role": "user",
        "content": message.content + (f"\n\n[File Content]:\n{file_content}" if file_content else "")
    })

    try:
        # 🔁 Run agent with file-enhanced history
        result = Runner.run_streamed(
            starting_agent=agent,
            input=history,
            context=dev,
        )

        response_msg = cl.Message(content="")
        first_response = True

        async for chunk in result.stream_events():
            if chunk.type == "raw_response_event" and isinstance(chunk.data, ResponseTextDeltaEvent):
                if first_response:
                    await thinking_msg.remove()
                    await response_msg.send()
                    first_response = False
                await response_msg.stream_token(chunk.data.delta)
        #added llm response to history
        history.append({
            "role": "assistant",
            "content": response_msg.content,
        })

        cl.user_session.set("history", history)
        await response_msg.update()

    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"❌ An Error Occurred: {str(e)}").send()
        print(f"Error: {e}")

@cl.on_chat_end
async def end():
    #saved chat history in a json file in root directory
    history = cl.user_session.get("history") or []
    with open("history.json","w") as f:
        json.dump(history,f ,indent=4)
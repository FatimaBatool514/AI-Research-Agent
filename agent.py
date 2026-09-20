"""The CrewAI part: one agent, one task, one crew."""

from datetime import date

from crewai import LLM, Agent, Crew, Process, Task

from tools import duckduckgo_search

# "groq/" tells CrewAI (via LiteLLM) to use Groq. The rest is Groq's own model ID.
MODEL_NAME = "groq/openai/gpt-oss-120b"

# How much work the agent should do. Smaller = faster and uses fewer tokens.
DEPTH_SETTINGS = {
    "Quick": {"searches": 2, "words": "500-700"},
    "Standard": {"searches": 4, "words": "800-1200"},
    "Deep": {"searches": 5, "words": "1200-1600"},
}


def run_research(topic: str, depth: str, api_key: str) -> str:
    """Run the research agent and return the finished report as Markdown text."""
    settings = DEPTH_SETTINGS[depth]

    # 1) The brain: Groq-hosted gpt-oss-120b
    llm = LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,           # low = more factual, less "creative"
        max_tokens=4000,           # upper limit for the report length (in tokens)
        reasoning_effort="medium",  # gpt-oss option: "low" | "medium" | "high"
    )

    # 2) The agent: who it is + which tools it may use
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research a topic on the web and write an accurate, well-structured report with sources.",
        backstory=(
            "You are a careful analyst. You search the web, compare sources, "
            "never invent facts, and clearly say when information is uncertain."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        allow_delegation=False,  # single agent: nobody to delegate to
        max_iter=10,             # safety limit on think/search loops
        verbose=True,            # prints the agent's steps to the terminal / app logs
    )

    # 3) The task: what to do and what the result should look like
    #    {topic}, {max_searches}, {word_range}, {today} are filled in by kickoff(inputs=...)
    task = Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Steps:\n"
            "1. Use the DuckDuckGo Search tool up to {max_searches} times, "
            "with different, specific queries.\n"
            "2. Note key facts, numbers, dates and differing viewpoints.\n"
            "3. Write a report of roughly {word_range} words.\n\n"
            "Rules:\n"
            "- Only state facts supported by your search results. Do not invent "
            "facts, numbers or URLs.\n"
            "- If sources disagree or information is thin, say so.\n"
            "- Today's date is {today}; prefer recent information."
        ),
        expected_output=(
            "A Markdown report with: a title (# heading), a short Executive Summary, "
            "3-5 sections with ## headings, a 'Key Takeaways' bullet list, and a "
            "'Sources' section listing the URLs you actually used. "
            "Do not wrap the report in a code block."
        ),
        agent=researcher,
    )

    # 4) The crew: runs the task(s) with the agent(s)
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(
        inputs={
            "topic": topic,
            "max_searches": settings["searches"],
            "word_range": settings["words"],
            "today": date.today().strftime("%B %d, %Y"),
        }
    )
    return result.raw  # the final text answer

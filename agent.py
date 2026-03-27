import os
import json
import random
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

DAILY_SCHEMES = {
    "Monday":    {"theme": "Thought Leadership",   "emoji": "💡", "hashtags": ["#ThoughtLeadership", "#AIInsights", "#Innovation"]},
    "Tuesday":   {"theme": "Case Study",           "emoji": "📊", "hashtags": ["#CaseStudy", "#RealResults", "#AIinAction"]},
    "Wednesday": {"theme": "How-To / Tutorial",    "emoji": "🛠️", "hashtags": ["#HowTo", "#Tutorial", "#LangChain", "#LangGraph"]},
    "Thursday":  {"theme": "Industry News",        "emoji": "📰", "hashtags": ["#AINews", "#TechTrends", "#FutureOfWork"]},
    "Friday":    {"theme": "Community Spotlight",  "emoji": "🌟", "hashtags": ["#Community", "#BuildInPublic", "#AIStartups"]},
    "Saturday":  {"theme": "Weekend Reflection",   "emoji": "🧘", "hashtags": ["#WeekendThoughts", "#Growth", "#Mindset"]},
    "Sunday":    {"theme": "Week Preview / Goals", "emoji": "🚀", "hashtags": ["#WeekAhead", "#Goals", "#StartupLife"]},
}

MEDIA_SOURCES = [
    "MIT Technology Review", "Harvard Business Review", "TechCrunch",
    "Wired", "The Verge", "Forbes", "VentureBeat", "McKinsey Quarterly"
]


class AgentState(TypedDict):
    day_of_week: str
    scheme: dict
    media_source: str
    linkedin_post: str
    veo3_prompt: str
    veo3_video_url: str        # mocked
    inspirational_post: str
    final_output: dict



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY")
)


def detect_day_node(state: AgentState) -> AgentState:
    """Detect today's day and load the matching scheme."""
    day = datetime.now().strftime("%A")           # e.g. "Monday"
    scheme = DAILY_SCHEMES.get(day, DAILY_SCHEMES["Monday"])
    media_source = random.choice(MEDIA_SOURCES)

    print(f"\n📅 Today is {day} → Theme: {scheme['theme']}")
    return {**state, "day_of_week": day, "scheme": scheme, "media_source": media_source}


def generate_linkedin_post_node(state: AgentState) -> AgentState:
    """Generate the main LinkedIn company update post."""
    scheme = state["scheme"]
    day = state["day_of_week"]

    messages = [
        SystemMessage(content=(
            "You are a world-class LinkedIn content strategist for an AI startup. "
            "You write posts that are insightful, engaging, and professional. "
            "Posts should be 150–250 words. Use line breaks for readability. "
            "Never use hollow buzzwords. Always add concrete value."
        )),
        HumanMessage(content=(
            f"Write a LinkedIn company update for {day}.\n"
            f"Today's theme is: {scheme['theme']}.\n"
            f"Include relevant insight, a hook opening line, and end with a question to drive engagement.\n"
            f"Add these hashtags at the end: {' '.join(scheme['hashtags'])}\n"
            f"The company builds AI agents using LangGraph and LangChain."
        ))
    ]

    response = llm.invoke(messages)
    print(f"\n✅ LinkedIn post generated ({scheme['theme']})")
    return {**state, "linkedin_post": response.content}


def generate_veo3_prompt_node(state: AgentState) -> AgentState:
    """Generate a Veo 3 video prompt and mock the video URL."""
    scheme = state["scheme"]

    messages = [
        SystemMessage(content=(
            "You are a creative director specializing in short-form AI-generated videos for LinkedIn. "
            "Write a vivid, cinematic Veo 3 video generation prompt (2–4 sentences). "
            "It should visually capture the day's theme in a professional, modern style."
        )),
        HumanMessage(content=(
            f"Generate a Veo 3 video prompt for a LinkedIn post about: {scheme['theme']}.\n"
            "The brand is an AI startup. Style: sleek, futuristic, optimistic. Duration: 15 seconds."
        ))
    ]

    response = llm.invoke(messages)
    veo3_prompt = response.content

    # ── MOCK: In production, call Google Veo 3 API here ──
    mock_video_url = f"https://veo3.mock/videos/{state['day_of_week'].lower()}_{scheme['theme'].replace(' ', '_').lower()}.mp4"

    print(f"\n🎬 Veo 3 prompt generated + video URL mocked")
    return {**state, "veo3_prompt": veo3_prompt, "veo3_video_url": mock_video_url}


def generate_inspirational_post_node(state: AgentState) -> AgentState:
    """Generate a short inspirational post referencing a real media source."""
    scheme = state["scheme"]
    source = state["media_source"]

    messages = [
        SystemMessage(content=(
            "You are a thought leader who writes punchy, inspiring LinkedIn micro-posts. "
            "3–5 short sentences max. Reference a known media source naturally — don't fabricate quotes. "
            "End with one powerful call-to-action line."
        )),
        HumanMessage(content=(
            f"Write an inspirational LinkedIn post tied to today's theme: {scheme['theme']}.\n"
            f"Reference insights from {source} (general knowledge, no made-up quotes).\n"
            f"Keep it under 100 words. Add emoji sparingly."
        ))
    ]

    response = llm.invoke(messages)
    print(f"\n✨ Inspirational post generated (source: {source})")
    return {**state, "inspirational_post": response.content}


def compile_output_node(state: AgentState) -> AgentState:
    """Bundle everything into a final structured output."""
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day": state["day_of_week"],
        "theme": state["scheme"]["theme"],
        "linkedin_post": state["linkedin_post"],
        "veo3_prompt": state["veo3_prompt"],
        "veo3_video_url": state["veo3_video_url"],
        "inspirational_post": state["inspirational_post"],
        "media_source": state["media_source"],
        "hashtags": state["scheme"]["hashtags"],
    }

    # JSON file
    filename = f"output_{output['date']}_{state['day_of_week']}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Output saved to {filename}")
    return {**state, "final_output": output}


def post_to_linkedin_node(state: AgentState) -> AgentState:
    """
    POST to LinkedIn API.
    ── MOCK: In production replace with real LinkedIn API call ──
    Requires: LINKEDIN_ACCESS_TOKEN and LINKEDIN_ORG_ID env variables.
    """
    output = state["final_output"]

    
    print(f"\n📤 [MOCK] Posted to LinkedIn: '{output['theme']}' post for {output['day']}")
    return state

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    #nodes
    graph.add_node("detect_day",            detect_day_node)
    graph.add_node("generate_linkedin",     generate_linkedin_post_node)
    graph.add_node("generate_veo3",         generate_veo3_prompt_node)
    graph.add_node("generate_inspirational",generate_inspirational_post_node)
    graph.add_node("compile_output",        compile_output_node)
    graph.add_node("post_to_linkedin",      post_to_linkedin_node)

    #flow
    graph.set_entry_point("detect_day")
    graph.add_edge("detect_day",             "generate_linkedin")
    graph.add_edge("generate_linkedin",      "generate_veo3")
    graph.add_edge("generate_veo3",          "generate_inspirational")
    graph.add_edge("generate_inspirational", "compile_output")
    graph.add_edge("compile_output",         "post_to_linkedin")
    graph.add_edge("post_to_linkedin",       END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()

    initial_state: AgentState = {
        "day_of_week": "",
        "scheme": {},
        "media_source": "",
        "linkedin_post": "",
        "veo3_prompt": "",
        "veo3_video_url": "",
        "inspirational_post": "",
        "final_output": {},
    }

    print("🤖 LinkedIn Agent starting...\n" + "─" * 40)
    result = agent.invoke(initial_state)

    print("\n" + "─" * 40)
    print("🎉 FINAL OUTPUT PREVIEW:\n")
    print(json.dumps(result["final_output"], indent=2))

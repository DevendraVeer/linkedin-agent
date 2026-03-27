# linkedin-agent

LangGraph agent that generates and posts daily LinkedIn company updates. Each day of the week has a theme — the agent detects the day, writes the post, generates a Veo 3 video prompt, and adds an inspirational snippet referencing a real media source.

## daily scheme

| Day | Theme |
|-----------|------------------------|
| Monday | Thought Leadership |
| Tuesday | Case Study |
| Wednesday | How-To / Tutorial |
| Thursday | Industry News |
| Friday | Community Spotlight |
| Saturday | Weekend Reflection |
| Sunday | Week Preview |

## stack
- LangGraph + LangChain — agent orchestration
- Groq (llama-3.3-70b) — content generation
- Veo 3 — video generation (mocked, prompt is real)
- LinkedIn API — posting (mocked, integration point marked in code)
- APScheduler — runs daily at 9 AM

## setup

```bash
git clone https://github.com/DevendraVeer/linkedin-agent
cd linkedin-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install apscheduler python-dotenv langchain-groq
```

Get a free API key at console.groq.com, then:

```bash
# create a .env file with:
GROQ_API_KEY=your-key-here
```

```bash
python agent.py          # run once
python scheduler.py      # run daily at 9 AM
```

## output

Saves a JSON file each run — `output_2026-03-27_Friday.json` — with the post, video prompt, inspirational snippet, and media source.
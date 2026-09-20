# 🔎 AI Research Agent

A beginner-friendly single-agent research app built with **CrewAI**, **Groq**
(`openai/gpt-oss-120b`), **DuckDuckGo** search and **Streamlit**.

Type a topic -> the agent searches the web -> you get a Markdown report with sources.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it, add your Groq key
streamlit run app.py
```

Get a free API key at https://console.groq.com/keys

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (do **not** commit `secrets.toml`).
2. Go to https://share.streamlit.io -> **Create app** -> pick this repo, branch `main`, file `app.py`.
3. **Advanced settings** -> Python version **3.12** -> paste in *Secrets*:
   `GROQ_API_KEY = "gsk_your_key_here"`
4. Deploy.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit user interface |
| `research_agent.py` | CrewAI agent, task and crew |
| `tools.py` | DuckDuckGo search tool |
| `requirements.txt` | Python dependencies |

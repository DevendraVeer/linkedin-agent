
from apscheduler.schedulers.blocking import BlockingScheduler
from agent import build_agent, AgentState
import json

def run_daily_agent():
    print("\n⏰ Scheduled run triggered!")
    agent = build_agent()
    initial_state: AgentState = {
        "day_of_week": "", "scheme": {}, "media_source": "",
        "linkedin_post": "", "veo3_prompt": "", "veo3_video_url": "",
        "inspirational_post": "", "final_output": {},
    }
    result = agent.invoke(initial_state)
    print("\n✅ Daily agent run complete.")
    print(json.dumps(result["final_output"], indent=2))

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # Runs every day at 9:00 AM local time
    scheduler.add_job(run_daily_agent, "cron", hour=9, minute=0)
    print("📅 Scheduler started — agent will post daily at 9:00 AM")
    print("Press Ctrl+C to stop.\n")
    scheduler.start()

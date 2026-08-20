import asyncio
from dotenv import load_dotenv

from autogen_core import SingleThreadedAgentRuntime, AgentId
from model.messages import Message
from agents.player_agent import Player1Agent, Player2Agent
from agents.judge_agent import RockPaperScissorsAgent

load_dotenv(override=True)

async def main():
    runtime = SingleThreadedAgentRuntime()
    
    # 1. Dependency Injection / Registration
    await Player1Agent.register(runtime, "player1", lambda: Player1Agent("player1"))
    await Player2Agent.register(runtime, "player2", lambda: Player2Agent("player2"))
    await RockPaperScissorsAgent.register(runtime, "rock_paper_scissors", lambda: RockPaperScissorsAgent("rock_paper_scissors"))
    
    # 2. Start the Engine
    runtime.start()
    
    print("Rock Paper Scissors Engine is Starting...\n")
    agent_id = AgentId("rock_paper_scissors", "default")
    message = Message(content="go")
    
    # 3. Trigger the Flow
    response = await runtime.send_message(message, agent_id)
    
    print("--- Final Output ---")
    print(response.content)
    
    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())
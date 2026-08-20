import asyncio
import os
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. Define the Message Dataclass 
@dataclass
class Message:
    content: str



# 2. Player 1 Agent (OpenAI - Cloud)
class Player1Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
        self._player1 = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._player1.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)
    



# 3. Player 2 Agent (Ollama - Local)
class Player2Agent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Note: Agar aapke paas Ollama install nahi hai, to aap isay bhi OpenAI me change kar sakte hain
        model_client = OllamaChatCompletionClient(model="llama3.2", temperature=1.0)
        self._player2 = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        text_message = TextMessage(content=message.content, source="user")
        response = await self._player2.on_messages([text_message], ctx.cancellation_token)
        return Message(content=response.chat_message.content)



# 4. Judge Agent (The Orchestrator)
JUDGE = "You are judging a game of rock, paper, scissors. The players have made these choices:\n"

class RockPaperScissorsAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.0)
        self._judge = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        instruction = "You are playing rock, paper, scissors. Respond only with the one word, one of the following: rock, paper, or scissors."
        msg = Message(content=instruction)
        
        agentid_1 = AgentId("player1", "default")
        agentid_2 = AgentId("player2", "default")
        
        # Dono players ko message bheja aur jawab ka intezar kiya
        response1 = await self.send_message(msg, agentid_1)
        response2 = await self.send_message(msg, agentid_2)
        
        result = f"Player 1: {response1.content}\nPlayer 2: {response2.content}\n"
        judgement = f"{JUDGE}{result} Who wins?"
        
        text_message = TextMessage(content=judgement, source="user")
        response = await self._judge.on_messages([text_message], ctx.cancellation_token)
        return Message(content=result + response.chat_message.content)




# 5. Main Execution (Khel Shuru Karein)
async def main():
    runtime = SingleThreadedAgentRuntime()
    
    # Factory Register Ki
    await Player1Agent.register(runtime, "player1", lambda: Player1Agent("player1"))
    await Player2Agent.register(runtime, "player2", lambda: Player2Agent("player2"))
    await RockPaperScissorsAgent.register(runtime, "rock_paper_scissors", lambda: RockPaperScissorsAgent("rock_paper_scissors"))
    
    runtime.start()
    
    print("Game is starting... Please wait!")
    agent_id = AgentId("rock_paper_scissors", "default")
    message = Message(content="go")
    
    # Judge ko 'go' ka message bheja
    response = await runtime.send_message(message, agent_id)
    
    print("\n--- Final Output ---")
    print(response.content)
    
    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())
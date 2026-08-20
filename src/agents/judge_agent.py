from autogen_core import RoutedAgent, message_handler, MessageContext, AgentId
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from model.messages import Message

JUDGE_PROMPT = "You are judging a game of rock, paper, scissors. The players have made these choices:\n"

class RockPaperScissorsAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.0)
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: Message, ctx: MessageContext) -> Message:
        instruction = "You are playing rock, paper, scissors. Respond only with the one word, one of the following: rock, paper, or scissors."
        msg = Message(content=instruction)
        
        inner_1 = AgentId("player1", "default")
        inner_2 = AgentId("player2", "default")
        
        response1 = await self.send_message(msg, inner_1)
        response2 = await self.send_message(msg, inner_2)
        
        result = f"Player 1: {response1.content}\nPlayer 2: {response2.content}\n"
        judgement = f"{JUDGE_PROMPT}{result}Who wins?"
        
        text_message = TextMessage(content=judgement, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        return Message(content=result + response.chat_message.content)
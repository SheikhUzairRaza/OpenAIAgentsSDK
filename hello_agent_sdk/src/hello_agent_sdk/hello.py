import os
import chainlit as cl


from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step-1 : Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Step-2 : Model
model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=provider)
# Step-3 : Define configurations at run level
config = RunConfig(model=model, model_provider=provider, tracing_disabled=True)
# Step-4 : Define Agent
agent: Agent = Agent(name="Learning Supporting Agent", instructions="You have to answer the questions asked by the user. Be polite and respectful.")
# Step-5 : Run the agent
# runner = Runner.run_sync(agent, input="Hello how are you?", run_config=config)
# print(runner.final_output)


"""
    This Python function listens for messages, runs a synchronous operation using a Runner class, and
    sends the final output as a message.
    
    :param message: The `message` parameter in the `main` function represents the message object that
    triggers the function when a message is received. It contains information about the message, such as
    the content, sender, timestamp, and other relevant details. In this context, the `message` parameter
    is used to extract the
    :type message: cl.Message
"""
# Step-5 : Run the agent

@cl.on_message 
async def main(message: cl.Message):
    result = Runner.run_sync(agent,input=message.content,run_config=config)
    await cl.Message(content=result.final_output).send()
import os

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
runner = Runner.run_sync(agent, input="Hello how are you?", run_config=config)
print(runner.final_output)
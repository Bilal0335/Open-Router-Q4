from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os
from dotenv import load_dotenv

# 1. Environment load karna
load_dotenv()

# 2. Model name & API key
Model_Name = "deepseek/deepseek-r1-0528:free"
Open_Router_API_Keys = os.getenv("OPENROUTER_API_KEYS")

if not Open_Router_API_Keys:
    raise ValueError("Open_Router_API_Keys is not set in the environment variables.")

# 3. OpenRouter client
external_client = AsyncOpenAI(
    api_key=Open_Router_API_Keys,
    base_url="https://openrouter.ai/api/v1"
)

# 4. Model create karna
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model=Model_Name
)

# 5. Run config
config = RunConfig(
    model=model,
    tracing_disabled=True,
    model_provider=external_client
)

# 6. Agent banana
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

# 7. Chat loop
print("🤖 AI Assistant ready! Type your question or 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting. Goodbye!")
        break
    result = Runner.run_sync(agent, user_input, run_config=config)
    print("Assistant:", result.final_output)

from dotenv import load_dotenv
import os
import asyncio
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig

# Load environment variables
load_dotenv()

async def main():
    MODEL_NAME = "deepseek/deepseek-r1-0528:free"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEYS")

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEYS is not set in the environment variables.")

    # ✅ Async Client
    external_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # ✅ Model
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    # ✅ Config
    config = RunConfig(
        model_provider=external_client,
        model=model,
        tracing_disabled=True
    )

    # ✅ Agent
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant."
    )

    # ✅ Async run
    result = await Runner.run(
        agent,
        input="Write an essay on Quaid-e-Azam",
        run_config=config
    )

    print(result.final_output)

# ✅ Run async
if __name__ == "__main__":
    asyncio.run(main())

from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig
import asyncio

# Load environment variables
load_dotenv()

async def main():
    MODEL_NAME = "deepseek/deepseek-r1-0528:free"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEYS")

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEYS is not set in the environment variables.")

    # Initialize async client
    external_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # Wrap into model
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    config = RunConfig(
        model_provider=external_client,
        model=model,
        tracing_disabled=True
    )

    # Agent creation
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant."
    )

    # ✅ Await the run coroutine
    result = await Runner.run(
        agent,
        "Tell me about Python in programming",
        run_config=config
    )

    print(result.final_output)

# ✅ Proper asyncio call
if __name__ == "__main__":
    asyncio.run(main())

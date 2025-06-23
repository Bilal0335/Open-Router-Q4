from dotenv import load_dotenv
import os
import asyncio
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig

# Load environment variables from .env file
load_dotenv()

async def main():
    # ✅ Use a stable model
    MODEL_NAME = "openai/gpt-3.5-turbo"  # more reliable than free tier models
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEYS")

    if not OPENROUTER_API_KEY:
        raise ValueError("❌ OPENROUTER_API_KEYS is not set in the environment variables.")

    # ✅ Take user input before entering async context
    user_input = input("Enter your question or prompt: ")

    # ✅ Initialize async OpenAI client via OpenRouter
    external_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # ✅ Wrap into OpenAIChatCompletionsModel
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    # ✅ Run configuration
    config = RunConfig(
        model_provider=external_client,
        model=model,
        tracing_disabled=True
    )

    # ✅ Create assistant agent
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant."
    )

    # ✅ Try running the assistant and handle errors gracefully
    try:
        result = await Runner.run(
            agent,
            input=user_input,
            run_config=config
        )

        if result and result.final_output:
            print("\n✅ AI Response:\n")
            print(result.final_output)
        else:
            print("⚠️ No response received. Try simplifying your prompt or check model availability.")

    except Exception as e:
        print(f"❌ An error occurred:\n{e}")

# ✅ Run async function
if __name__ == "__main__":
    asyncio.run(main())

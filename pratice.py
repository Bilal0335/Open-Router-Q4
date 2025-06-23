# agents, models, aur config import kar rahe hain
# in modules ki madad se hum AI agent banate hain
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# OS aur dotenv import kar rahe hain
# .env file se API keys securely load karne ke liye
import os
from dotenv import load_dotenv

# .env file ko load karo (jisme API keys waghera hoti hain)
load_dotenv()

# AI model ka naam define kar rahe hain (ye OpenRouter ka free model hai)
Model_Name = "deepseek/deepseek-r1-0528:free"

# .env file se OpenRouter API key le rahe hain
Open_Router_API_Keys = os.getenv("OPENROUTER_API_KEYS")

# Agar API key nahi mili to error do (taake program ruk jaye)
if not Open_Router_API_Keys:
    raise ValueError("Open_Router_API_Keys is not set in the environment variables.")

# Ab ek AI client bana rahe hain jo OpenRouter ke server se connect hoga
external_client = AsyncOpenAI(
    api_key=Open_Router_API_Keys,                # API key jo .env se mili
    base_url="https://openrouter.ai/api/v1"      # OpenRouter ka API URL
)

# Ab model ko define kar rahe hain jo AI responses generate karega
model = OpenAIChatCompletionsModel(
    openai_client=external_client,               # Upar banaya hua client
    model=Model_Name                             # Model ka naam jo use hoga
)

# RunConfig ek configuration object hai jo AI run ke settings define karta hai
config = RunConfig(
    model=model,                                 # Kaunsa model use karna hai
    tracing_disabled=True,                       # Tracing (debugging logs) disable kar rahe hain
    model_provider=external_client               # Kaunsa client request bhejega
)

# Ab ek AI agent (Assistant) create kar rahe hain
# Ye agent user ka input lega aur model se jawab laayega
agent = Agent(
    name="Assistant",                            # Agent ka naam
    instructions="You are a helpful assistant.", # Agent ko bataya gaya uska role
    model=model                                  # Model jo use hoga response dene ke liye
)

# Ab hum agent ko run kar rahe hain user ke question ke sath
# run_sync ka matlab hai yeh synchronous function hai, await ki zarurat nahi
result = Runner.run_sync(
    agent,                                       # Jo agent banaya
    "Tell me about Python in programming",       # User ka question
    run_config=config                            # Configuration settings
)

# Jo response mila AI se, usko print kar do
print(result)

import asyncio
from foundry_llm import FoundryLLM

models = {
    "openai": FoundryLLM("gpt-4o-mini"),
    "claude": FoundryLLM("claude-sonnet-4-6"),
    "gemini": FoundryLLM("gemini-2.5-flash"),
}

msgs = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is idempotency?"},
]

# sync
for name, llm in models.items():
    print(name, "->", llm.invoke(msgs))
    for tok in llm.stream(msgs):
        print(tok, end="", flush=True)
    print()

# async
async def main():
    for name, llm in models.items():
        print(name, "->", await llm.ainvoke(msgs))
        async for tok in llm.astream(msgs):
            print(tok, end="", flush=True)
        print()
        await llm.aclose()

asyncio.run(main())
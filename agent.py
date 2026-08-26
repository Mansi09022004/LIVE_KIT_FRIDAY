from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, ChatContext
from livekit.plugins import (
    noise_cancellation,
    openai
)
from livekit.plugins import google
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tools import get_weather, search_web, send_email
from mem0 import AsyncMemoryClient
from mcp_client import MCPServerSse
from mcp_client.agent_tools import MCPToolsIntegration
import os
import json
import logging
import asyncio
load_dotenv()

# Without this, our own logging.info() calls (chat context, memory saving,
# etc.) are silently dropped — livekit configures its own loggers but not
# the root logger, so plain logging.info() stays invisible unless we set
# the root level ourselves.
logging.basicConfig(level=logging.INFO)


class Assistant(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=google.beta.realtime.RealtimeModel(
                # NOTE: gemini-3.1-flash-live-preview (Google's newer
                # model) does NOT support mid-session updates — it makes
                # chat_ctx/instructions immutable, which breaks our
                # generate_reply() greeting call AND would break mem0
                # memory injection. So we stay on the 2.5 native-audio
                # preview, which is still officially supported (no
                # shutdown date announced) and fully mutable. The
                # "1008 Requested entity was not found" error seen
                # earlier looks like a transient Google-side blip rather
                # than a permanent removal — if it recurs consistently,
                # that needs following up with Google directly.
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                voice="Aoede",
                temperature=0.8,
            ),
            tools=[
                get_weather,
                search_web,
                send_email
            ],
            chat_ctx=chat_ctx

        )



async def entrypoint(ctx: agents.JobContext):

    async def shutdown_hook(chat_ctx: ChatContext, mem0: AsyncMemoryClient, memory_str: str):
        logging.info("Shutting down, saving chat context to memory...")

        messages_formatted = [
        ]

        logging.info(f"Chat context messages: {chat_ctx.items}")

        for item in chat_ctx.items:
            # Some chat_ctx items (e.g. AgentConfigUpdate from the realtime
            # API) aren't actual chat messages and don't have role/content
            # attributes — skip anything that isn't a real message.
            # Only send the user's own messages to mem0 — assistant replies
            # (like Friday's "Roger Boss" catchphrase) aren't facts about
            # the user and shouldn't become memories.
            role = getattr(item, "role", None)
            if role != 'user':
                continue

            content = getattr(item, "content", None)
            content_str = ''.join(content) if isinstance(content, list) else str(content)

            if memory_str and memory_str in content_str:
                continue

            messages_formatted.append({
                "role": role,
                "content": content_str.strip()
            })

        if not messages_formatted:
            logging.info("No new user messages to save to memory.")
            return

        logging.info(f"Formatted messages to add to memory: {messages_formatted}")
        await mem0.add(
            messages_formatted,
            user_id="Mansi",
            # Guide mem0's extraction away from one-off tool requests
            # (weather checks, "send an email", etc.) — those are actions,
            # not durable facts worth remembering about the user.
            excludes=(
                "One-off requests, commands, or questions the user asked "
                "the assistant to act on, such as checking the weather or "
                "sending an email. Only store durable facts, preferences, "
                "or personal details about the user."
            ),
        )
        logging.info("Chat context saved to memory.")


    session = AgentSession(

    )



    mem0 = AsyncMemoryClient()
    user_name = 'Mansi'

    # NOTE: this mem0 SDK version requires filters={} instead of a top-level
    # user_id kwarg, and get_all() returns {"results": [...]} rather than a
    # plain list.
    response = await mem0.get_all(filters={"user_id": user_name})
    results = response.get("results", [])
    initial_ctx = ChatContext()
    memory_str = ''

    if results:
        memories = [
            {
                "memory": result["memory"],
                "updated_at": result["updated_at"]
            }
            for result in results
        ]
        memory_str = json.dumps(memories)
        logging.info(f"Memories: {memory_str}")
        initial_ctx.add_message(
            role="assistant",
            content=f"The user's name is {user_name}, and this is relvant context about him: {memory_str}."
        )

    # MCP (e.g. an n8n MCP server) is optional — only wire it in if
    # N8N_MCP_SERVER_URL is actually configured, so the agent doesn't crash
    # trying to connect to nothing when it isn't set up yet.
    n8n_mcp_url = os.environ.get("N8N_MCP_SERVER_URL")

    if n8n_mcp_url:
        mcp_server = MCPServerSse(
            params={"url": n8n_mcp_url},
            cache_tools_list=True,
            name="SSE MCP Server"
        )

        agent = await MCPToolsIntegration.create_agent_with_tools(
            agent_class=Assistant, agent_kwargs={"chat_ctx": initial_ctx},
            mcp_servers=[mcp_server]
        )
    else:
        logging.warning(
            "N8N_MCP_SERVER_URL not set — starting without MCP tools."
        )
        agent = Assistant(chat_ctx=initial_ctx)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # The realtime API's websocket connection can occasionally be flaky
    # right at startup (timeouts waiting for a generation_created event) —
    # retry a couple of times with a short backoff instead of failing
    # the very first greeting outright.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            await session.generate_reply(
                instructions=SESSION_INSTRUCTION,
            )
            break
        except Exception:
            logging.warning(
                f"generate_reply failed (attempt {attempt}/{max_attempts}), retrying...",
                exc_info=True,
            )
            if attempt == max_attempts:
                logging.error("generate_reply kept failing, giving up on the initial greeting.")
            else:
                await asyncio.sleep(1)

    ctx.add_shutdown_callback(lambda: shutdown_hook(session._agent.chat_ctx, mem0, memory_str))

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

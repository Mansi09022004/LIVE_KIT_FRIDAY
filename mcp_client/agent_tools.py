import asyncio
import logging
import json
import inspect
import typing
from typing import Any, List, Dict, Callable, Optional, Awaitable, Sequence, Tuple, Type, Union, cast
from uuid import uuid4

# Import from the MCP module
from .util import MCPUtil, FunctionTool
from .server import MCPServer, MCPServerSse
from livekit.agents import ChatContext, AgentSession, JobContext, FunctionTool as Tool
from mcp import CallToolRequest

logger = logging.getLogger("mcp-agent-tools")


# Gemini Live's websocket has a much tighter payload limit per turn than
# the regular chat API. Raw MCP tool results (e.g. a Spotify search
# returning full track objects with huge `available_markets` arrays,
# complete `images` arrays, etc.) can blow past that limit — when they
# do, Gemini doesn't degrade gracefully, it kills the whole realtime
# session with a 1007 "context exhausted" error that can't be recovered
# from. Shrink any oversized tool result before it goes back into the
# session, generically, so this protects every MCP tool (not just
# Spotify) without needing to know its schema.
_MAX_RESULT_LIST_ITEMS = 5
_MAX_RESULT_CHARS = 4000
_BULKY_KEYS = ("available_markets", "images", "external_urls", "external_ids")


def _shrink_json_value(value):
    if isinstance(value, dict):
        return {
            k: _shrink_json_value(v)
            for k, v in value.items()
            if k not in _BULKY_KEYS
        }
    if isinstance(value, list):
        return [_shrink_json_value(v) for v in value[:_MAX_RESULT_LIST_ITEMS]]
    return value


def shrink_tool_result(result_str: str) -> str:
    """Trim a raw MCP tool result so it can't blow Gemini Live's per-turn
    payload limit. Tries to parse it as JSON and drop bulky/redundant
    fields + cap list lengths; falls back to plain character truncation
    for non-JSON (or unparsable) results."""
    if not isinstance(result_str, str):
        return result_str

    try:
        parsed = json.loads(result_str)
    except (ValueError, TypeError):
        parsed = None

    if parsed is not None:
        shrunk = json.dumps(_shrink_json_value(parsed))
        if len(shrunk) <= _MAX_RESULT_CHARS:
            return shrunk
        return shrunk[:_MAX_RESULT_CHARS] + "...[truncated]"

    if len(result_str) <= _MAX_RESULT_CHARS:
        return result_str
    return result_str[:_MAX_RESULT_CHARS] + "...[truncated]"


class MCPToolsIntegration:
    """
    Helper class for integrating MCP tools with LiveKit agents.
    Provides utilities for registering dynamic tools from MCP servers.
    """

    @staticmethod
    async def prepare_dynamic_tools(mcp_servers: List[MCPServer],
                                   convert_schemas_to_strict: bool = True,
                                   auto_connect: bool = True) -> List[Callable]:
        """
        Fetches tools from multiple MCP servers and prepares them for use with LiveKit agents.

        Args:
            mcp_servers: List of MCPServer instances
            convert_schemas_to_strict: Whether to convert JSON schemas to strict format
            auto_connect: Whether to automatically connect to servers if they're not connected

        Returns:
            List of decorated tool functions ready to be added to a LiveKit agent
        """
        prepared_tools = []

        # Ensure all servers are connected if auto_connect is True
        if auto_connect:
            for server in mcp_servers:
                if not getattr(server, 'connected', False):
                    try:
                        logger.debug(f"Auto-connecting to MCP server: {server.name}")
                        await server.connect()
                    except Exception as e:
                        logger.error(f"Failed to connect to MCP server {server.name}: {e}")

        # Process each server
        for server in mcp_servers:
            logger.info(f"Fetching tools from MCP server: {server.name}")
            try:
                mcp_tools = await MCPUtil.get_function_tools(
                    server, convert_schemas_to_strict=convert_schemas_to_strict
                )
                logger.info(f"Received {len(mcp_tools)} tools from {server.name}")
            except Exception as e:
                logger.error(f"Failed to fetch tools from {server.name}: {e}")
                continue

            # Process each tool from this server
            for tool_instance in mcp_tools:
                try:
                    decorated_tool = MCPToolsIntegration._create_decorated_tool(tool_instance)
                    prepared_tools.append(decorated_tool)
                    logger.debug(f"Successfully prepared tool: {tool_instance.name}")
                except Exception as e:
                    logger.error(f"Failed to prepare tool '{tool_instance.name}': {e}")

        return prepared_tools

    @staticmethod
    def _create_decorated_tool(tool: FunctionTool) -> Callable:
        """
        Creates a decorated function for a single MCP tool that can be used with LiveKit agents.

        Args:
            tool: The FunctionTool instance to convert

        Returns:
            A decorated async function that can be added to a LiveKit agent's tools
        """
        # Get function_tool decorator from LiveKit
        # Import locally to avoid circular imports
        from livekit.agents.llm import function_tool
        import keyword as _keyword

        schema_props = tool.params_json_schema.get("properties", {})
        schema_required = set(tool.params_json_schema.get("required", []))
        type_map = {
            "string": "str", "integer": "int", "number": "float",
            "boolean": "bool", "array": "list", "object": "dict",
        }

        # NOTE: We used to build this function by hand-assigning
        # __signature__/__annotations__ onto a **kwargs function. That
        # worked with older livekit-agents versions, but newer versions
        # introspect tools via inspect.signature()/typing.get_type_hints()
        # in a way that doesn't reliably pick up manually-hacked
        # attributes (it would find the parameter in the signature but
        # then KeyError looking up its type hint). Building the function
        # from real Python source via exec() instead means the signature
        # and annotations are generated naturally by Python itself, so
        # they're guaranteed to agree.
        #
        # MCP/JSON-schema property names also aren't guaranteed to be
        # valid Python identifiers, so we sanitize them and keep a map
        # back to the original name to build the tool-call payload.
        name_map: Dict[str, str] = {}  # identifier -> original schema property name
        used_idents = set()
        for p_name in schema_props:
            ident = "".join(c if (c.isalnum() or c == "_") else "_" for c in p_name)
            if not ident or ident[0].isdigit():
                ident = f"p_{ident}"
            if _keyword.iskeyword(ident):
                ident = f"{ident}_"
            base_ident = ident
            i = 1
            while ident in used_idents:
                ident = f"{base_ident}_{i}"
                i += 1
            used_idents.add(ident)
            name_map[ident] = p_name

        param_src_parts = []
        for ident, p_name in name_map.items():
            p_details = schema_props[p_name]
            json_type = p_details.get("type", "string")
            py_type = type_map.get(json_type, "typing.Any")
            if p_name in schema_required:
                param_src_parts.append(f"{ident}: {py_type}")
            else:
                default = p_details.get("default", None)
                param_src_parts.append(f"{ident}: typing.Optional[{py_type}] = {default!r}")

        # No leading `self`/context param — livekit calls these tools with
        # plain keyword arguments matching the schema, nothing more.
        params_src = "*, " + ", ".join(param_src_parts) if param_src_parts else ""

        body_lines = ["    kwargs = {}"]
        for ident, p_name in name_map.items():
            body_lines.append(f"    kwargs[{p_name!r}] = {ident}")
        body_lines += [
            "    input_json = _json.dumps(kwargs)",
            "    _logger.info(f\"Invoking tool {_tool_name!r} with args: {kwargs}\")",
            "    result_str = await _tool.on_invoke_tool(None, input_json)",
            "    result_str = _shrink_tool_result(result_str)",
            "    _logger.info(f\"Tool {_tool_name!r} result (shrunk): {result_str}\")",
            "    return result_str",
        ]

        src = f"async def _dyn_tool_impl({params_src}) -> str:\n" + "\n".join(body_lines)

        namespace = {
            "typing": typing,
            "_json": json,
            "_logger": logger,
            "_tool": tool,
            "_tool_name": tool.name,
            "_shrink_tool_result": shrink_tool_result,
        }
        exec(src, namespace)  # noqa: S102 - building a typed wrapper from a JSON schema
        tool_impl = namespace["_dyn_tool_impl"]

        tool_impl.__name__ = tool.name
        tool_impl.__doc__ = tool.description

        # Apply the decorator and return
        return function_tool()(tool_impl)

    @staticmethod
    async def register_with_agent(agent, mcp_servers: List[MCPServer],
                                 convert_schemas_to_strict: bool = True,
                                 auto_connect: bool = True) -> List[Callable]:
        """
        Helper method to prepare and register MCP tools with a LiveKit agent.

        Args:
            agent: The LiveKit agent instance
            mcp_servers: List of MCPServer instances
            convert_schemas_to_strict: Whether to convert schemas to strict format
            auto_connect: Whether to auto-connect to servers

        Returns:
            List of tool functions that were registered
        """
        # Prepare the dynamic tools
        tools = await MCPToolsIntegration.prepare_dynamic_tools(
            mcp_servers,
            convert_schemas_to_strict=convert_schemas_to_strict,
            auto_connect=auto_connect
        )

        # Register with the agent
        if hasattr(agent, '_tools') and isinstance(agent._tools, list):
            agent._tools.extend(tools)
            logger.info(f"Registered {len(tools)} MCP tools with agent")

            # Log the names of registered tools
            if tools:
                tool_names = [getattr(t, '__name__', 'unknown') for t in tools]
                logger.info(f"Registered tool names: {tool_names}")
        else:
            logger.warning("Agent does not have a '_tools' attribute, tools were not registered")

        return tools

    @staticmethod
    async def create_agent_with_tools(agent_class, mcp_servers: List[MCPServer], agent_kwargs: Dict = None,
                                    convert_schemas_to_strict: bool = True) -> Any:
        """
        Factory method to create and initialize an agent with MCP tools already loaded.

        Args:
            agent_class: Agent class to instantiate
            mcp_servers: List of MCP servers to register with the agent
            agent_kwargs: Additional keyword arguments to pass to the agent constructor
            convert_schemas_to_strict: Whether to convert JSON schemas to strict format

        Returns:
            An initialized agent instance with MCP tools registered
        """
        # Connect to MCP servers
        for server in mcp_servers:
            if not getattr(server, 'connected', False):
                try:
                    logger.debug(f"Connecting to MCP server: {server.name}")
                    await server.connect()
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server {server.name}: {e}")

        # Create agent instance
        agent_kwargs = agent_kwargs or {}
        agent = agent_class(**agent_kwargs)

        # Prepare tools
        tools = await MCPToolsIntegration.prepare_dynamic_tools(
            mcp_servers,
            convert_schemas_to_strict=convert_schemas_to_strict,
            auto_connect=False  # Already connected above
        )

        # Register tools with agent
        if tools and hasattr(agent, '_tools') and isinstance(agent._tools, list):
            agent._tools.extend(tools)
            logger.info(f"Registered {len(tools)} MCP tools with agent")

            # Log the names of registered tools
            tool_names = [getattr(t, '__name__', 'unknown') for t in tools]
            logger.info(f"Registered tool names: {tool_names}")
        else:
            if not tools:
                logger.warning("No tools were found to register with the agent")
            else:
                logger.warning("Agent does not have a '_tools' attribute, tools were not registered")

        return agent

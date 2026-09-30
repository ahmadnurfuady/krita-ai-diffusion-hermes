from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..util import client_logger as log


class HermesRole(Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool_result = "tool_result"


@dataclass
class HermesMessage:
    role: HermesRole
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class HermesToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class HermesResponse:
    message: HermesMessage
    tool_calls: list[HermesToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class DocumentContext:
    canvas_width: int = 0
    canvas_height: int = 0
    layers: list[dict[str, Any]] = field(default_factory=list)
    active_layer: str = ""
    selection: dict[str, int] | None = None
    current_prompt: str = ""
    style_name: str = ""


# MCP tool definitions that Hermes can call to control Krita
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using the ComfyUI diffusion pipeline. Returns generated image data. Use enhanced prompts for better results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed positive prompt for image generation. Be comprehensive with style, quality, lighting etc.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt - things to avoid in the generation.",
                    },
                    "strength": {
                        "type": "number",
                        "description": "Denoising strength 0.0-1.0. Use 1.0 for full generation, lower for refinement.",
                        "default": 1.0,
                    },
                    "target_layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names for output layers. If specified, result will be split into these layers. Example: ['lineart', 'base_color', 'shading', 'highlights']",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for reproducibility. Use -1 for random.",
                        "default": -1,
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_to_layer",
            "description": "Generate an image and place the result into a specific named layer, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed positive prompt for generation.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt.",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to place the result into.",
                    },
                    "strength": {
                        "type": "number",
                        "description": "Denoising strength 0.0-1.0.",
                        "default": 1.0,
                    },
                },
                "required": ["prompt", "layer_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inpaint_region",
            "description": "Generate content within a specific rectangular region of the canvas, leaving the rest untouched.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed prompt describing what to generate in the region.",
                    },
                    "negative_prompt": {"type": "string"},
                    "x": {"type": "integer", "description": "Left coordinate of the region."},
                    "y": {"type": "integer", "description": "Top coordinate of the region."},
                    "width": {"type": "integer", "description": "Width of the region."},
                    "height": {"type": "integer", "description": "Height of the region."},
                    "strength": {
                        "type": "number",
                        "description": "Denoising strength.",
                        "default": 0.85,
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Target layer name. If empty, applies to current layer.",
                    },
                },
                "required": ["prompt", "x", "y", "width", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inpaint_selection",
            "description": "Generate content within the user's current selection on the canvas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt for the inpainting region.",
                    },
                    "negative_prompt": {"type": "string"},
                    "strength": {
                        "type": "number",
                        "default": 0.85,
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Target layer for the result.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_layer",
            "description": "Create a new empty layer in the Krita document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new layer.",
                    },
                    "layer_type": {
                        "type": "string",
                        "enum": ["paint", "group"],
                        "description": "Type of layer to create.",
                        "default": "paint",
                    },
                    "visible": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_layer",
            "description": "Set a specific layer as the active (selected) layer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the layer to select.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_layer_visibility",
            "description": "Show or hide a specific layer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "visible": {"type": "boolean"},
                },
                "required": ["name", "visible"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_layer_group",
            "description": "Create a group layer and optionally move existing layers into it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string"},
                    "child_layer_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of existing layers to move into the group.",
                    },
                },
                "required": ["group_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_selection",
            "description": "Programmatically set a rectangular selection on the canvas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_selection",
            "description": "Remove the current selection from the canvas.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_canvas_info",
            "description": "Get current canvas state: size, layers, active layer, selection.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_layers",
            "description": "Plan and create a layer structure for a painting project. Creates all layers and groups needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["paint", "group"]},
                                "children": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "description": "Layer structure from bottom to top. Groups can contain children.",
                    },
                },
                "required": ["structure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_layered",
            "description": "Generate a complete layered illustration. The agent will generate the image multiple times with different prompts focusing on different aspects and put them into separate layers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_prompt": {
                        "type": "string",
                        "description": "Overall description of the subject/scene.",
                    },
                    "layers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "prompt_suffix": {
                                    "type": "string",
                                    "description": "Additional prompt appended for this layer's generation pass.",
                                },
                                "strength": {"type": "number", "default": 1.0},
                            },
                        },
                        "description": "Layer definitions. Each gets its own generation pass.",
                    },
                    "negative_prompt": {"type": "string"},
                    "seed": {"type": "integer", "default": -1},
                },
                "required": ["subject_prompt", "layers"],
            },
        },
    },
]

HERMES_SYSTEM_PROMPT = """\
You are Hermes, an expert AI painting assistant integrated into Krita via the krita-ai-diffusion plugin. \
You control Krita's canvas through MCP (Model Context Protocol) tools.

Your capabilities:
1. **Smart Prompt Enhancement**: When a user gives a simple prompt, you enhance it with detailed \
descriptions of style, quality, lighting, composition, and artistic details for better ComfyUI generation.
2. **Layer Management**: You can create, organize, and manage layers. For complex illustrations, \
you automatically separate elements into layers (lineart, base color, shading, highlights, background, etc.).
3. **Context-Aware Editing**: You understand the canvas state - which layer is active, what's selected, \
and what content exists. You edit only what the user intends to change.
4. **Regional Generation**: You can generate or edit specific regions of the canvas while preserving \
other areas.
5. **Iterative Refinement**: You can refine existing content by adjusting strength and targeting \
specific areas.

Guidelines:
- Always enhance user prompts with quality tags, style descriptors, and technical details
- When creating characters/illustrations, default to separating into layers unless told otherwise
- Respect user selections - if they've selected a region, work within that region
- For background changes, only modify the background layer
- For character edits, only modify character layers
- Use descriptive layer names that reflect content (e.g., "Character - Lineart", "BG - Sky")
- When the user mentions a sketch or existing art, use lower strength (0.3-0.7) to preserve their work
- Provide brief explanations of what you're doing and why

Current canvas context will be provided with each message.\
"""


class HermesClient:
    def __init__(self, url: str = "", model_name: str = "hermes", api_key: str = ""):
        self._url = url
        self._model_name = model_name
        self._api_key = api_key
        self._conversation: list[HermesMessage] = []
        self._system_prompt = HERMES_SYSTEM_PROMPT

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value: str):
        self._url = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value

    @property
    def conversation(self):
        return list(self._conversation)

    def clear_conversation(self):
        self._conversation.clear()

    async def send_message(
        self,
        user_message: str,
        context: DocumentContext | None = None,
    ) -> HermesResponse:
        context_text = ""
        if context:
            context_text = self._format_context(context)

        full_message = f"{context_text}\n\n{user_message}" if context_text else user_message

        self._conversation.append(HermesMessage(HermesRole.user, full_message))

        try:
            response = await self._call_api()
            self._conversation.append(response.message)
            return response
        except Exception as e:
            log.error(f"Hermes API call failed: {e}")
            error_str = str(e)
            if "429" in error_str:
                msg = (
                    "Rate limited (429) after retries. The model's rate limit is very "
                    "strict. Try waiting a minute, or switch to a non-free model."
                )
            else:
                msg = f"Connection error: {e}. Please check the Hermes server URL in settings."
            error_msg = HermesMessage(HermesRole.assistant, msg)
            self._conversation.append(error_msg)
            return HermesResponse(error_msg)

    async def send_tool_result(self, tool_call_id: str, result: str) -> HermesResponse:
        tool_msg = HermesMessage(HermesRole.tool_result, result, tool_call_id=tool_call_id)
        self._conversation.append(tool_msg)

        try:
            response = await self._call_api()
            self._conversation.append(response.message)
            return response
        except Exception as e:
            log.error(f"Hermes tool result call failed: {e}")
            error_msg = HermesMessage(HermesRole.assistant, f"Error: {e}")
            self._conversation.append(error_msg)
            return HermesResponse(error_msg)

    async def _call_api(self) -> HermesResponse:
        import ssl
        import urllib.error
        import urllib.request

        messages = self._build_messages()
        payload = {
            "model": self._model_name or "hermes",
            "messages": messages,
            "tools": MCP_TOOLS,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        data = json.dumps(payload).encode("utf-8")
        base = self._url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        elif base.endswith("/chat/completions"):
            url = base
        else:
            url = f"{base}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if "openrouter.ai" in url:
            headers["HTTP-Referer"] = "https://github.com/Acly/krita-ai-diffusion"
            headers["X-Title"] = "Krita AI Diffusion Hermes Agent"

        loop = asyncio.get_event_loop()
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )

            def do_request():
                with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            try:
                response_data = await loop.run_in_executor(None, do_request)
                return self._parse_response(response_data)
            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if e.code == 429 and attempt < max_retries:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = base_delay * (2**attempt)
                    else:
                        delay = base_delay * (2**attempt)
                    delay = min(delay, 60.0)
                    log.warning(
                        f"Rate limited (429), retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(f"HTTP {e.code} error body: {error_body[:500]}")
                    raise

    def _build_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self._conversation:
            if msg.role == HermesRole.tool_result:
                # OpenAI API expects role "tool" with tool_call_id
                entry: dict[str, Any] = {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id or "",
                }
            elif msg.role == HermesRole.assistant and msg.tool_calls:
                # Assistant message with tool_calls: content should be null if empty
                entry = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": msg.tool_calls,
                }
            else:
                entry = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            messages.append(entry)
        return messages

    def _parse_response(self, data: dict) -> HermesResponse:
        choice = data["choices"][0]
        msg_data = choice["message"]

        tool_calls = []
        raw_tool_calls = msg_data.get("tool_calls", [])
        tc_list = []
        for tc in raw_tool_calls:
            fn = tc["function"]
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_call = HermesToolCall(
                id=tc.get("id", ""),
                name=fn["name"],
                arguments=args,
            )
            tool_calls.append(tool_call)
            tc_list.append(tc)

        message = HermesMessage(
            role=HermesRole.assistant,
            content=msg_data.get("content", ""),
            tool_calls=tc_list,
        )

        return HermesResponse(
            message=message,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _format_context(self, ctx: DocumentContext) -> str:
        parts = [f"[Canvas: {ctx.canvas_width}x{ctx.canvas_height}]"]
        if ctx.style_name:
            parts.append(f"[Style: {ctx.style_name}]")
        if ctx.active_layer:
            parts.append(f"[Active Layer: {ctx.active_layer}]")
        if ctx.selection:
            s = ctx.selection
            parts.append(f"[Selection: x={s['x']}, y={s['y']}, w={s['width']}, h={s['height']}]")
        if ctx.layers:
            layer_info = []
            for l in ctx.layers:
                vis = "👁" if l.get("visible", True) else "  "
                layer_info.append(f"  {vis} {l.get('name', '?')} ({l.get('type', '?')})")
            parts.append("[Layers (bottom→top):\n" + "\n".join(layer_info) + "\n]")
        return " ".join(parts[:4]) + ("\n" + parts[4] if len(parts) > 4 else "")

    async def check_connection(self) -> tuple[bool, str]:
        """Test if the server is reachable and credentials are valid."""
        import ssl
        import time
        import urllib.request

        if not self._url:
            return False, "URL is not set"

        base = self._url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/models"
        elif base.endswith("/chat/completions"):
            url = base.replace("/chat/completions", "/models")
        else:
            url = f"{base}/v1/models"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if "openrouter.ai" in url:
            headers["HTTP-Referer"] = "https://github.com/Acly/krita-ai-diffusion"
            headers["X-Title"] = "Krita AI Diffusion Hermes Agent"

        req = urllib.request.Request(url, headers=headers, method="GET")
        loop = asyncio.get_event_loop()
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        def do_ping():
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=10, context=ctx) as _resp:
                elapsed = int((time.time() - t0) * 1000)
                return True, f"Connected! ({elapsed}ms)"

        try:
            return await loop.run_in_executor(None, do_ping)
        except Exception:
            return await self._check_connection_chat()

    async def _check_connection_chat(self) -> tuple[bool, str]:
        import ssl
        import time
        import urllib.request

        base = self._url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        elif base.endswith("/chat/completions"):
            url = base
        else:
            url = f"{base}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if "openrouter.ai" in url:
            headers["HTTP-Referer"] = "https://github.com/Acly/krita-ai-diffusion"
            headers["X-Title"] = "Krita AI Diffusion Hermes Agent"

        payload = {
            "model": self._model_name or "hermes",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        loop = asyncio.get_event_loop()
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        def do_post():
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=15, context=ctx) as _resp:
                elapsed = int((time.time() - t0) * 1000)
                return True, f"Connected! ({elapsed}ms)"

        try:
            return await loop.run_in_executor(None, do_post)
        except Exception as e:
            return False, f"Failed: {e}"

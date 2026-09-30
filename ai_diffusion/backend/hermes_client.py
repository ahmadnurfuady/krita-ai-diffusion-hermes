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
                        "description": "Optional output layer name. Note: diffusion generates 1 complete raster image per pass.",
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
            "description": "Generate multiple separate conceptual layers sequentially (e.g. background layer, then foreground subject layer). Each pass creates an actual Krita layer. Note that full-canvas generation produces opaque layers; use inpaint_selection for localized additions.",
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
                                "region": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "integer"},
                                        "y": {"type": "integer"},
                                        "width": {"type": "integer"},
                                        "height": {"type": "integer"},
                                    },
                                    "description": "Optional bounding box for inpainting this layer. Omit for full-canvas background.",
                                },
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

Core Rules & Guidelines:
1. **Style & Model Fidelity (CRITICAL)**:
   - Check the active style in the context: `[Style: <style_name>]`.
   - You MUST adapt your enhanced prompt to the active style!
   - If the style is **Flux**, **Cinematic Photo**, **Realistic**, or **Digital Artwork**, DO NOT include anime/manga tags (no "anime style", "1girl", "manga aesthetic") unless the user explicitly requested anime!
   - For Flux: Use natural, richly descriptive photographic or artistic sentences with details on lighting, camera, lens, textures, and mood.
   - For Anime/Illustrious: Use quality tags, booru tags, and anime stylization descriptors.

2. **Multi-Pass Layered Illustration Strategy (CRITICAL)**:
   - Users expect you to create structured, multi-layer artworks in Krita rather than just 1 flat image!
   - When asked to create an illustration, character, or scene from scratch:
     - **Pass 1 [Background Layer]**: Generate the environment/scenery first using `generate_to_layer(layer_name="Background", prompt="...")`. Ensure the prompt describes ONLY the setting/background without the main subject.
     - **Pass 2 [Character / Main Subject Layer]**: Add the character on top of the background by inpainting the subject into the desired canvas area using `inpaint_region(prompt="<detailed subject prompt in setting>", x=..., y=..., width=..., height=..., strength=0.95, layer_name="Character")`.
       (Calculate coordinates based on canvas dimensions from context, e.g. on a 1024x1024 canvas, a centered character is typically x=200, y=100, width=624, height=850).
     - **Pass 3 [Atmosphere / Effects / Foreground] (optional)**: Add fog, glowing magic, particle effects, or foreground overlay with another `inpaint_region` or create an empty paint layer using `create_layer(name="Effects")`.
   - Alternatively, you can use `generate_layered` specifying layer items (Layer 1 without region for Background, Layer 2 with a targeted `region` for Character/Subject).
   - If the user explicitly asks for a single fast generation or simple edit, you may generate directly to the active layer.

3. **Context-Aware Editing**:
   - Understand the canvas state - active layer, selection bounds, and existing layers.
   - If the user has made a selection on canvas, use `inpaint_selection` to edit only within that selection.

4. **Task Completion**:
   - After executing your multi-pass workflow to fulfill the user's request, provide a concise summary of the created layers and what each contains, then STOP calling tools. Do not loop unnecessarily.

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
        return await self.send_tool_results([(tool_call_id, result)])

    async def send_tool_results(
        self, tool_results: list[tuple[str, str]]
    ) -> HermesResponse:
        for tool_call_id, result in tool_results:
            tool_msg = HermesMessage(HermesRole.tool_result, result, tool_call_id=tool_call_id)
            self._conversation.append(tool_msg)

        try:
            response = await self._call_api()
            self._conversation.append(response.message)
            return response
        except Exception as e:
            log.error(f"Hermes tool results call failed: {e}")
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

            def do_request(r=req):
                with urllib.request.urlopen(r, timeout=120, context=ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            try:
                response_data = await loop.run_in_executor(None, do_request)
                return self._parse_response(response_data)
            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                except Exception as err:
                    log.debug(f"Failed to read error body: {err}")
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

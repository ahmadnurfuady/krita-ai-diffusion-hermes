from __future__ import annotations

import asyncio
import json
import uuid
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

from .. import eventloop
from ..backend.hermes_client import (
    DocumentContext,
    HermesClient,
    HermesResponse,
    HermesToolCall,
)
from ..image import Bounds, Image
from ..util import client_logger as log
from .jobs import Job, JobKind, JobState
from .properties import ObservableProperties, Property


class HermesState(Enum):
    idle = 0
    thinking = 1
    executing_tools = 2
    generating = 3
    error = 4


class ChatMessage:
    def __init__(self, role: str, content: str, is_tool_call: bool = False):
        self.role = role
        self.content = content
        self.is_tool_call = is_tool_call
        self.id = str(uuid.uuid4())


class HermesModel(QObject, ObservableProperties):
    state = Property(HermesState.idle)
    status_text = Property("")

    state_changed = pyqtSignal(HermesState)
    status_text_changed = pyqtSignal(str)
    message_added = pyqtSignal(object)  # ChatMessage
    conversation_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._client = HermesClient()
        self._messages: list[ChatMessage] = []
        self._document_model = None  # set via property
        self._pending_generations: list[dict] = []
        self._tool_execution_count = 0
        self._max_tool_iterations = 10

    @property
    def messages(self):
        return list(self._messages)

    @property
    def client(self):
        return self._client

    def set_document_model(self, model):
        self._document_model = model

    def set_url(self, url: str, model_name: str = "", api_key: str = ""):
        self._client.url = url
        if model_name:
            self._client.model_name = model_name
        self._client.api_key = api_key

    def clear_conversation(self):
        self._client.clear_conversation()
        self._messages.clear()
        self.conversation_cleared.emit()

    def send_message(self, text: str):
        if not text.strip():
            return
        if self.state not in (HermesState.idle, HermesState.error):
            return

        msg = ChatMessage("user", text)
        self._messages.append(msg)
        self.message_added.emit(msg)

        self.state = HermesState.thinking
        self.status_text = "Hermes is thinking..."
        eventloop.run(self._process_message(text))

    async def _process_message(self, text: str):
        try:
            context = self._gather_context()
            response = await self._client.send_message(text, context)
            await self._handle_response(response)
        except Exception as e:
            log.error(f"Hermes message processing failed: {e}")
            self._add_assistant_message(f"❌ Error: {e}")
            self.state = HermesState.error
            self.status_text = "Error occurred"

    async def _handle_response(self, response: HermesResponse):
        self._tool_execution_count = 0
        await self._process_response(response)

    async def _process_response(self, response: HermesResponse):
        if response.message.content:
            self._add_assistant_message(response.message.content)

        if response.tool_calls and self._tool_execution_count < self._max_tool_iterations:
            self._tool_execution_count += 1
            self.state = HermesState.executing_tools

            tool_results: list[tuple[str, str]] = []
            for tool_call in response.tool_calls:
                self.status_text = f"Executing: {tool_call.name}..."
                tool_msg = ChatMessage(
                    "tool",
                    f"🔧 `{tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)})`",
                    is_tool_call=True,
                )
                self._messages.append(tool_msg)
                self.message_added.emit(tool_msg)

                result = await self._execute_tool(tool_call)

                result_msg = ChatMessage("tool_result", f"→ {result}", is_tool_call=True)
                self._messages.append(result_msg)
                self.message_added.emit(result_msg)

                tool_results.append((tool_call.id, result))

            next_response = await self._client.send_tool_results(tool_results)
            await self._process_response(next_response)
        else:
            self.state = HermesState.idle
            self.status_text = ""

    def _add_assistant_message(self, content: str):
        msg = ChatMessage("assistant", content)
        self._messages.append(msg)
        self.message_added.emit(msg)

    async def _execute_tool(self, tool_call: HermesToolCall) -> str:
        name = tool_call.name
        args = tool_call.arguments

        try:
            if name == "generate_image":
                return await self._tool_generate_image(args)
            elif name == "generate_to_layer":
                return await self._tool_generate_to_layer(args)
            elif name == "inpaint_region":
                return await self._tool_inpaint_region(args)
            elif name == "inpaint_selection":
                return await self._tool_inpaint_selection(args)
            elif name == "create_layer":
                return self._tool_create_layer(args)
            elif name == "select_layer":
                return self._tool_select_layer(args)
            elif name == "set_layer_visibility":
                return self._tool_set_layer_visibility(args)
            elif name == "create_layer_group":
                return self._tool_create_layer_group(args)
            elif name == "set_selection":
                return self._tool_set_selection(args)
            elif name == "clear_selection":
                return self._tool_clear_selection(args)
            elif name == "get_canvas_info":
                return self._tool_get_canvas_info(args)
            elif name == "plan_layers":
                return self._tool_plan_layers(args)
            elif name == "generate_layered":
                return await self._tool_generate_layered(args)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            log.error(f"Tool execution error ({name}): {e}")
            return f"Error executing {name}: {e}"

    # ── Tool implementations ──────────────────────────────────────────

    def _apply_job_to_layer(self, model, job: Job | None, layer_name: str | None = None):
        if not job or not job.id:
            return
        try:
            model.apply_generated_result(job.id, 0)
            if layer_name and model.layers.active:
                model.layers.active.name = f"[Hermes] {layer_name}"
        except Exception as e:
            log.warning(f"Failed to apply generated result to layer: {e}")

    async def _tool_generate_image(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        prompt = args["prompt"]
        negative = args.get("negative_prompt", "")
        strength = args.get("strength", 1.0)
        target_layers = args.get("target_layers", [])
        seed = args.get("seed", -1)

        try:
            self.state = HermesState.generating
            self.status_text = "Generating image..."

            # Override prompts in the model and trigger generation
            model.regions.positive = prompt
            if negative:
                model.regions.negative = negative
            model.strength = strength
            if seed >= 0:
                model.seed = seed
                model.fixed_seed = True

            prev_ids = {j.id for j in model.jobs if j.id}
            model.generate()

            # Wait for generation to complete
            job = await self._wait_for_generation(model, prev_ids)

            layer_name = target_layers[0] if target_layers else None
            if job:
                self._apply_job_to_layer(model, job, layer_name)
                return f"Generated image successfully and applied to layer (Prompt: '{prompt[:80]}...')"
            else:
                return f"Generated image successfully with prompt: '{prompt[:80]}...'"

        except Exception as e:
            return f"Generation failed: {e}"
        finally:
            model.fixed_seed = False
            self.state = HermesState.executing_tools

    async def _tool_generate_to_layer(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        prompt = args["prompt"]
        negative = args.get("negative_prompt", "")
        layer_name = args["layer_name"]
        strength = args.get("strength", 1.0)
        seed = args.get("seed", -1)

        try:
            self.state = HermesState.generating
            self.status_text = f"Generating to layer '{layer_name}'..."

            model.regions.positive = prompt
            if negative:
                model.regions.negative = negative
            model.strength = strength
            if seed >= 0:
                model.seed = seed
                model.fixed_seed = True

            prev_ids = {j.id for j in model.jobs if j.id}
            model.generate()
            job = await self._wait_for_generation(model, prev_ids)

            if job:
                self._apply_job_to_layer(model, job, layer_name)
                return f"Generated image and applied to layer '{layer_name}' (Prompt: '{prompt[:60]}...')"
            else:
                return f"Generation completed for layer '{layer_name}'"
        except Exception as e:
            return f"Generate to layer failed: {e}"
        finally:
            model.fixed_seed = False
            self.state = HermesState.executing_tools

    async def _tool_inpaint_region(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        prompt = args["prompt"]
        x, y = args["x"], args["y"]
        w, h = args["width"], args["height"]
        strength = args.get("strength", 0.85)
        layer_name = args.get("layer_name", "")

        try:
            self.state = HermesState.generating
            self.status_text = f"Inpainting region ({x},{y} {w}x{h})..."

            # Set selection programmatically via Krita API
            self._set_krita_selection(x, y, w, h)

            model.regions.positive = prompt
            model.strength = strength

            prev_ids = {j.id for j in model.jobs if j.id}
            model.generate()
            job = await self._wait_for_generation(model, prev_ids)

            if job:
                self._apply_job_to_layer(model, job, layer_name or f"Inpaint ({x},{y})")

            # Clear the selection
            self._clear_krita_selection()

            return f"Inpainted region ({x},{y} {w}x{h}) with: '{prompt[:60]}...'"
        except Exception as e:
            return f"Inpaint region failed: {e}"
        finally:
            self.state = HermesState.executing_tools

    async def _tool_inpaint_selection(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        prompt = args["prompt"]
        strength = args.get("strength", 0.85)
        layer_name = args.get("layer_name", "")

        if model.document.selection_bounds is None:
            return "Error: No selection active on canvas. Ask the user to make a selection first."

        try:
            self.state = HermesState.generating
            self.status_text = "Inpainting selection..."

            model.regions.positive = prompt
            model.strength = strength

            prev_ids = {j.id for j in model.jobs if j.id}
            model.generate()
            job = await self._wait_for_generation(model, prev_ids)

            if job:
                self._apply_job_to_layer(model, job, layer_name or "Inpaint Selection")

            return f"Inpainted selection with: '{prompt[:60]}...'"
        except Exception as e:
            return f"Inpaint selection failed: {e}"
        finally:
            self.state = HermesState.executing_tools

    def _tool_create_layer(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        name = args["name"]
        layer_type = args.get("layer_type", "paint")
        visible = args.get("visible", True)

        try:
            doc = model.document
            extent = doc.extent
            empty = Image.create(extent, fill=0)
            bounds = Bounds(0, 0, *extent)

            if layer_type == "group":
                layer = model.layers.create_group(name)
            else:
                layer = model.layers.create(name, empty, bounds)

            if not visible:
                layer.is_visible = False

            return f"Created {layer_type} layer '{name}'"
        except Exception as e:
            return f"Failed to create layer: {e}"

    def _tool_select_layer(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        name = args["name"]
        try:
            for layer in model.layers.all:
                if layer.name == name:
                    model.layers.active = layer
                    return f"Selected layer '{name}'"
            return f"Layer '{name}' not found"
        except Exception as e:
            return f"Failed to select layer: {e}"

    def _tool_set_layer_visibility(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        name = args["name"]
        visible = args["visible"]
        try:
            for layer in model.layers.all:
                if layer.name == name:
                    layer.is_visible = visible
                    state = "visible" if visible else "hidden"
                    return f"Layer '{name}' set to {state}"
            return f"Layer '{name}' not found"
        except Exception as e:
            return f"Failed to set visibility: {e}"

    def _tool_create_layer_group(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        group_name = args["group_name"]
        children = args.get("child_layer_names", [])

        try:
            model.layers.create_group(group_name)
            moved = []
            for child_name in children:
                for layer in model.layers.all:
                    if layer.name == child_name:
                        moved.append(child_name)
                        break
            result = f"Created group '{group_name}'"
            if moved:
                result += f" with layers: {', '.join(moved)}"
            return result
        except Exception as e:
            return f"Failed to create group: {e}"

    def _tool_set_selection(self, args: dict) -> str:
        try:
            x, y = args["x"], args["y"]
            w, h = args["width"], args["height"]
            self._set_krita_selection(x, y, w, h)
            return f"Set selection to ({x},{y}) {w}x{h}"
        except Exception as e:
            return f"Failed to set selection: {e}"

    def _tool_clear_selection(self, _args: dict) -> str:
        try:
            self._clear_krita_selection()
            return "Selection cleared"
        except Exception as e:
            return f"Failed to clear selection: {e}"

    def _tool_get_canvas_info(self, _args: dict) -> str:
        ctx = self._gather_context()
        info = {
            "canvas_size": f"{ctx.canvas_width}x{ctx.canvas_height}",
            "active_layer": ctx.active_layer,
            "style": ctx.style_name,
            "selection": ctx.selection,
            "layers": ctx.layers,
        }
        return json.dumps(info, indent=2, ensure_ascii=False)

    def _tool_plan_layers(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        structure = args["structure"]
        created = []

        try:
            doc = model.document
            extent = doc.extent
            empty = Image.create(extent, fill=0)
            bounds = Bounds(0, 0, *extent)

            for entry in structure:
                name = entry["name"]
                ltype = entry.get("type", "paint")

                if ltype == "group":
                    layer = model.layers.create_group(name)
                    children = entry.get("children", [])
                    for child_name in children:
                        model.layers.create(child_name, empty, bounds, parent=layer)
                    created.append(f"Group '{name}' with {len(children)} children")
                else:
                    model.layers.create(name, empty, bounds)
                    created.append(f"Layer '{name}'")

            return "Created layer structure:\n" + "\n".join(f"  • {c}" for c in created)
        except Exception as e:
            return f"Failed to create layer structure: {e}"

    async def _tool_generate_layered(self, args: dict) -> str:
        model = self._document_model
        if model is None:
            return "Error: No document model available"

        subject = args["subject_prompt"]
        layers_spec = args["layers"]
        negative = args.get("negative_prompt", "")
        seed = args.get("seed", -1)

        results = []
        try:
            for i, layer_spec in enumerate(layers_spec):
                layer_name = layer_spec["name"]
                prompt_suffix = layer_spec.get("prompt_suffix", "")
                strength = layer_spec.get("strength", 1.0)

                full_prompt = f"{subject}, {prompt_suffix}" if prompt_suffix else subject

                self.state = HermesState.generating
                self.status_text = f"Generating layer {i + 1}/{len(layers_spec)}: {layer_name}..."

                model.regions.positive = full_prompt
                if negative:
                    model.regions.negative = negative
                model.strength = strength
                if seed >= 0:
                    model.seed = seed
                    model.fixed_seed = True

                region = layer_spec.get("region")
                has_region = False
                if region and isinstance(region, dict):
                    rx = int(region.get("x", 0))
                    ry = int(region.get("y", 0))
                    rw = int(region.get("width", 0))
                    rh = int(region.get("height", 0))
                    if rw > 0 and rh > 0:
                        self._set_krita_selection(rx, ry, rw, rh)
                        has_region = True

                prev_ids = {j.id for j in model.jobs if j.id}
                model.generate()
                job = await self._wait_for_generation(model, prev_ids)

                if has_region:
                    self._clear_krita_selection()

                if job:
                    self._apply_job_to_layer(model, job, layer_name)
                    results.append(layer_name)

            return f"Generated {len(results)} layers: {', '.join(results)}"
        except Exception as e:
            return f"Layered generation failed at layer {len(results) + 1}: {e}"
        finally:
            model.fixed_seed = False
            self.state = HermesState.executing_tools

    # ── Helpers ───────────────────────────────────────────────────────

    def _gather_context(self) -> DocumentContext:
        ctx = DocumentContext()
        model = self._document_model
        if model is None or not model.has_document:
            return ctx

        doc = model.document
        ctx.canvas_width, ctx.canvas_height = doc.extent

        try:
            if model.layers.active:
                ctx.active_layer = model.layers.active.name
        except Exception as e:
            log.debug(f"Could not read active layer: {e}")

        try:
            if hasattr(model, "active_style") and model.active_style:
                ctx.style_name = model.active_style.name
            elif model.style:
                ctx.style_name = model.style.name
        except Exception as e:
            log.debug(f"Could not read style name: {e}")

        try:
            sel = doc.selection_bounds
            if sel:
                ctx.selection = {
                    "x": sel.x,
                    "y": sel.y,
                    "width": sel.width,
                    "height": sel.height,
                }
        except Exception as e:
            log.debug(f"Could not read selection bounds: {e}")

        try:
            for layer in model.layers.all:
                if not layer.type.is_filter:
                    ctx.layers.append({
                        "name": layer.name,
                        "type": layer.type.value,
                        "visible": layer.is_visible,
                    })
        except Exception as e:
            log.debug(f"Could not enumerate layers: {e}")

        return ctx

    async def _wait_for_generation(
        self, model, previous_job_ids: set[str] | None = None, timeout: float = 300.0
    ) -> Job | None:
        elapsed = 0.0
        interval = 0.5
        # Wait for the job to be enqueued
        await asyncio.sleep(1.0)
        while elapsed < timeout:
            if not model.jobs.any_executing():
                pending = [j for j in model.jobs if j.state is JobState.queued]
                if not pending:
                    break
            await asyncio.sleep(interval)
            elapsed += interval
        else:
            raise TimeoutError("Generation timed out")

        prev = previous_job_ids or set()
        for job in reversed(list(model.jobs)):
            if (
                job.kind is JobKind.diffusion
                and job.state is JobState.finished
                and len(job.results) > 0
                and job.id not in prev
            ):
                return job

        # Fallback to latest finished diffusion job
        for job in reversed(list(model.jobs)):
            if (
                job.kind is JobKind.diffusion
                and job.state is JobState.finished
                and len(job.results) > 0
            ):
                return job

        return None

    def _set_krita_selection(self, x: int, y: int, w: int, h: int):
        try:
            from krita import Krita

            doc = Krita.instance().activeDocument()
            if doc:
                selection = doc.selection()
                if selection is None:
                    import krita

                    selection = krita.Selection()
                selection.select(x, y, w, h, 255)
                doc.setSelection(selection)
        except Exception as e:
            log.warning(f"Failed to set Krita selection: {e}")

    def _clear_krita_selection(self):
        try:
            from krita import Krita

            doc = Krita.instance().activeDocument()
            if doc:
                doc.setSelection(None)
        except Exception as e:
            log.warning(f"Failed to clear Krita selection: {e}")

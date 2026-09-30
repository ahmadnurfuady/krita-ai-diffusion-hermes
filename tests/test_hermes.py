from ai_diffusion.backend.hermes_client import (
    MCP_TOOLS,
    DocumentContext,
    HermesClient,
)
from ai_diffusion.model.hermes import ChatMessage, HermesModel, HermesState
from ai_diffusion.model.model import Workspace
from ai_diffusion.settings import Settings


def test_hermes_tools():
    assert len(MCP_TOOLS) >= 10
    names = [t["function"]["name"] for t in MCP_TOOLS]
    expected_tools = [
        "generate_image",
        "generate_to_layer",
        "inpaint_region",
        "inpaint_selection",
        "create_layer",
        "select_layer",
        "set_layer_visibility",
        "create_layer_group",
        "set_selection",
        "clear_selection",
        "get_canvas_info",
        "plan_layers",
        "generate_layered",
    ]
    for name in expected_tools:
        assert name in names


def test_hermes_client_context():
    client = HermesClient("http://localhost:8080")
    assert client.url == "http://localhost:8080"

    ctx = DocumentContext(
        canvas_width=1024,
        canvas_height=1024,
        active_layer="Background",
        style_name="Anime",
        selection={"x": 10, "y": 20, "width": 300, "height": 400},
        layers=[{"name": "Background", "type": "paintlayer", "visible": True}],
    )
    context_str = client._format_context(ctx)
    assert "1024x1024" in context_str
    assert "Anime" in context_str
    assert "Background" in context_str
    assert "Selection" in context_str


def test_hermes_client_parse_response():
    client = HermesClient("http://localhost:8080")
    mock_data = {
        "choices": [
            {
                "message": {
                    "content": "Creating layers for anime character.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "plan_layers",
                                "arguments": '{"structure": [{"name": "Lineart", "type": "paint"}]}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    resp = client._parse_response(mock_data)
    assert resp.message.content == "Creating layers for anime character."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "plan_layers"
    assert resp.tool_calls[0].arguments == {"structure": [{"name": "Lineart", "type": "paint"}]}


def test_hermes_settings():
    s = Settings()
    assert hasattr(s, "hermes_url")
    assert hasattr(s, "hermes_enabled")
    assert hasattr(s, "hermes_model")
    assert hasattr(s, "hermes_api_key")
    assert s.hermes_url == ""
    assert s.hermes_enabled is True
    assert s.hermes_model == "hermes"
    assert s.hermes_api_key == ""

    s.hermes_url = "http://127.0.0.1:8080"
    s.hermes_model = "nousresearch/hermes-3-llama-3.1-8b"
    s.hermes_api_key = "sk-test-123"
    assert s.hermes_url == "http://127.0.0.1:8080"
    assert s.hermes_model == "nousresearch/hermes-3-llama-3.1-8b"
    assert s.hermes_api_key == "sk-test-123"
    s.hermes_enabled = False
    assert s.hermes_enabled is False


def test_hermes_workspace_enum():
    assert Workspace.hermes.value == 5
    assert Workspace.hermes.name == "hermes"


def test_hermes_model():
    model = HermesModel()
    assert model.state == HermesState.idle
    assert len(model.messages) == 0

    msg = ChatMessage("user", "Hello Hermes")
    model._messages.append(msg)
    assert len(model.messages) == 1
    assert model.messages[0].content == "Hello Hermes"

    ctx = model._gather_context()
    assert ctx.canvas_width == 0
    assert ctx.canvas_height == 0

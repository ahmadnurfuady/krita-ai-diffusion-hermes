from typing import ClassVar

from krita import Krita
from PyQt6.QtCore import QMetaObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.hermes import ChatMessage, HermesModel, HermesState
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind
from ..model.root import root
from ..settings import settings
from . import theme
from .widget import StyleSelectWidget, WorkspaceSelectWidget


class ChatBubble(QFrame):
    _role_colors: ClassVar[dict[str, tuple[str, str]]] = {
        "user": ("#2d5a88", "#e3f0ff"),
        "assistant": ("#2a6e3f", "#e8f8ee"),
        "tool": ("#6e5a2a", "#fff8e1"),
        "tool_result": ("#5a2a6e", "#f3e8ff"),
    }

    _role_colors_dark: ClassVar[dict[str, tuple[str, str]]] = {
        "user": ("#1a3a5c", "#1a3a5c"),
        "assistant": ("#1a4a2e", "#1a4a2e"),
        "tool": ("#4a3a1a", "#4a3a1a"),
        "tool_result": ("#3a1a4a", "#3a1a4a"),
    }

    def __init__(self, message: ChatMessage, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        is_dark = theme.is_dark
        colors = self._role_colors_dark if is_dark else self._role_colors
        border_color, bg_color = colors.get(message.role, ("#666", "#f0f0f0"))

        self.setStyleSheet(f"""
            ChatBubble {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 4px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        role_label = QLabel(self._role_display(message.role))
        role_font = role_label.font()
        role_font.setPointSize(max(7, role_font.pointSize() - 2))
        role_font.setBold(True)
        role_label.setFont(role_font)
        role_label.setStyleSheet(f"color: {border_color}; background: transparent; border: none;")
        layout.addWidget(role_label)

        content_label = QLabel(message.content)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        content_label.setStyleSheet("background: transparent; border: none; padding: 2px;")
        if message.is_tool_call:
            mono_font = QFont("Consolas", max(8, content_label.font().pointSize() - 1))
            content_label.setFont(mono_font)
        layout.addWidget(content_label)

        if message.role == "user":
            layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

    def _role_display(self, role: str) -> str:
        return {
            "user": "👤 You",
            "assistant": "🤖 Hermes",
            "tool": "🔧 Tool Call",
            "tool_result": "📋 Result",
        }.get(role, role.title())


class ChatInputWidget(QPlainTextEdit):
    submitted = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText(_("Ask Hermes to paint, edit, create layers, or inpaint..."))
        self.setMaximumHeight(72)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        border = "#3a3a3a" if theme.is_dark else "#ccc"
        bg = "#222" if theme.is_dark else "#fff"
        text_color = "#eee" if theme.is_dark else "#111"
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {bg};
                color: {text_color};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 6px;
                font-size: 11px;
            }}
            QPlainTextEdit:focus {{
                border: 1px solid {"#5294e2" if theme.is_dark else "#2563eb"};
            }}
        """)

    def keyPressEvent(self, e: QKeyEvent | None):
        if e is None:
            return
        if e.key() == Qt.Key.Key_Return and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            text = self.toPlainText().strip()
            if text:
                self.submitted.emit(text)
                self.clear()
            return
        super().keyPressEvent(e)


class HermesWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._model: DocumentModel | None = (
            root.active_model if hasattr(root, "_null_model") else None
        )
        self._model_bindings: list[QMetaObject.Connection | Binding] = []
        self._hermes = HermesModel()
        if self._model is not None:
            self._hermes.set_document_model(self._model)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 2, 0)
        layout.setSpacing(4)

        # 1. Top row: Workspace & Style selectors
        self.workspace_select = WorkspaceSelectWidget(self)
        self.style_select = StyleSelectWidget(self)
        top_row = QHBoxLayout()
        top_row.addWidget(self.workspace_select)
        top_row.addWidget(self.style_select)
        layout.addLayout(top_row)

        # 2. Header
        header = QHBoxLayout()
        title = QLabel("🤖 Hermes Agent")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {theme.green}; font-size: 11px;")
        header.addWidget(self._status_label)

        self._clear_button = QToolButton(self)
        self._clear_button.setText("🗑")
        self._clear_button.setToolTip(_("Clear conversation"))
        self._clear_button.clicked.connect(self._clear_conversation)
        header.addWidget(self._clear_button)
        layout.addLayout(header)

        # 3. Input prompt area (placed at top for immediate visibility)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)

        self._input = ChatInputWidget(self)
        input_layout.addWidget(self._input, 1)

        self._send_button = QPushButton("▶ " + _("Send"))
        self._send_button.setFixedHeight(44)
        self._send_button.setMinimumWidth(60)
        self._send_button.setToolTip(
            _("Send instruction to Hermes (Enter, Shift+Enter for newline)")
        )
        self._send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#2d5a88" if theme.is_dark else "#3b82f6"};
                color: white;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {"#3a6a9a" if theme.is_dark else "#2563eb"};
            }}
            QPushButton:disabled {{
                background-color: {"#333" if theme.is_dark else "#ccc"};
                color: {"#777" if theme.is_dark else "#888"};
            }}
        """)
        self._send_button.clicked.connect(self._send_message)
        input_layout.addWidget(self._send_button)
        layout.addLayout(input_layout)

        # 4. URL warning banner (visible when hermes_url is not configured)
        self._url_banner = QFrame(self)
        self._url_banner.setStyleSheet(f"""
            QFrame {{
                background-color: {"#3a2e10" if theme.is_dark else "#fff8e1"};
                border: 1px solid {"#8a6d10" if theme.is_dark else "#ffe082"};
                border-radius: 6px;
                padding: 2px 4px;
            }}
        """)
        ub_layout = QHBoxLayout(self._url_banner)
        ub_layout.setContentsMargins(6, 2, 6, 2)
        ub_layout.setSpacing(6)
        hint_label = QLabel("⚠️ " + _("Set Hermes URL in Settings → Hermes Agent"), self._url_banner)
        hint_label.setStyleSheet(
            f"color: {'#ffd54f' if theme.is_dark else '#b78103'}; font-size: 10px; border: none; background: transparent;"
        )
        ub_layout.addWidget(hint_label, 1)

        cfg_btn = QPushButton(_("Configure"), self._url_banner)
        cfg_btn.setFixedHeight(22)
        cfg_btn.setStyleSheet("font-size: 10px; padding: 2px 4px;")
        cfg_btn.clicked.connect(self._open_settings)
        ub_layout.addWidget(cfg_btn)
        layout.addWidget(self._url_banner)

        # 5. Chat history area
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFrameStyle(QFrame.Shape.NoFrame)
        self._scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._chat_container = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(4, 4, 4, 4)
        self._chat_layout.setSpacing(6)

        welcome = QLabel(
            "💡 "
            + _(
                "Hi! I'm Hermes, your AI painting assistant. Tell me what you'd like to create "
                "and I'll handle prompts, layers, and generation for you."
            )
        )
        welcome.setWordWrap(True)
        welcome.setStyleSheet(
            f"background: {'#1a2a3a' if theme.is_dark else '#eef5ff'}; "
            f"border-radius: 6px; padding: 8px; "
            f"color: {'#aaccee' if theme.is_dark else '#336699'};"
        )
        self._chat_layout.addWidget(welcome)
        self._chat_layout.addStretch()

        self._scroll_area.setWidget(self._chat_container)
        layout.addWidget(self._scroll_area, 1)

        self._update_url_hint()

    def _connect_signals(self):
        self._input.submitted.connect(self._send_message_text)
        self._hermes.message_added.connect(self._on_message_added)
        self._hermes.state_changed.connect(self._on_state_changed)
        self._hermes.status_text_changed.connect(self._on_status_changed)
        self._hermes.conversation_cleared.connect(self._on_conversation_cleared)
        settings.changed.connect(self._update_url_hint)

    def _open_settings(self):
        if action := Krita.instance().action("ai_diffusion_settings"):
            action.trigger()

    def _update_url_hint(self):
        url = getattr(settings, "hermes_url", "")
        self._url_banner.setVisible(not bool(url))
        if self._hermes.state == HermesState.idle:
            if url:
                self._status_label.setStyleSheet(f"color: {theme.green}; font-size: 11px;")
                self._status_label.setText("● " + _("Ready"))
            else:
                self._status_label.setStyleSheet(
                    f"color: {'#888' if theme.is_dark else '#999'}; font-size: 11px;"
                )
                self._status_label.setText("○ " + _("Offline"))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model != model:
            Binding.disconnect_all(self._model_bindings)
            self._model = model
            self._hermes.set_document_model(model)
            self._model_bindings = [
                bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
                bind(model, "style", self.style_select, "value"),
            ]
            self.style_select.update_styles()
            url = getattr(settings, "hermes_url", "")
            model_name = getattr(settings, "hermes_model", "")
            api_key = getattr(settings, "hermes_api_key", "")
            if url:
                self._hermes.set_url(url, model_name, api_key)
            self._update_url_hint()

    def _send_message(self):
        text = self._input.toPlainText().strip()
        if text:
            self._send_message_text(text)
            self._input.clear()

    def _send_message_text(self, text: str):
        url = getattr(settings, "hermes_url", "")
        model_name = getattr(settings, "hermes_model", "")
        api_key = getattr(settings, "hermes_api_key", "")
        if url:
            self._hermes.set_url(url, model_name, api_key)
        if not self._hermes.client.url:
            self._hermes._add_assistant_message(
                "⚠️ Hermes server URL is not configured. "
                "Go to Settings → Hermes Agent and set the URL."
            )
            self._on_message_added(self._hermes.messages[-1])
            return

        self._hermes.send_message(text)

    def _clear_conversation(self):
        self._hermes.clear_conversation()

    def _on_message_added(self, message: ChatMessage):
        bubble = ChatBubble(message, self._chat_container)
        count = self._chat_layout.count()
        self._chat_layout.insertWidget(max(0, count - 1), bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scrollbar = self._scroll_area.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def _on_state_changed(self, state: HermesState):
        busy = state not in (HermesState.idle, HermesState.error)
        self._send_button.setEnabled(not busy)
        self._input.setEnabled(not busy)

        if state == HermesState.error:
            self._status_label.setStyleSheet(f"color: {'#ff6666' if theme.is_dark else 'red'};")
            self._status_label.setText("● " + _("Error"))
        elif busy:
            self._status_label.setStyleSheet(f"color: {'#66aaff' if theme.is_dark else '#3b82f6'};")
        else:
            url = getattr(settings, "hermes_url", "")
            if url:
                self._status_label.setStyleSheet(f"color: {theme.green};")
                self._status_label.setText("● " + _("Ready"))
            else:
                self._status_label.setStyleSheet(f"color: {'#888' if theme.is_dark else '#999'};")
                self._status_label.setText("○ " + _("Offline"))

    def _on_status_changed(self, text: str):
        self._status_label.setText(text)

    def _on_conversation_cleared(self):
        while self._chat_layout.count() > 2:
            item = self._chat_layout.takeAt(1)
            if item and (w := item.widget()):
                w.deleteLater()

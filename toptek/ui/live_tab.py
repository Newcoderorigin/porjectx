"""Interactive LM Studio Live tab using Tkinter widgets."""

from __future__ import annotations

from typing import List, Mapping, MutableSequence

try:  # pragma: no cover - exercised via import behaviour
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:  # pragma: no cover - headless environments
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

from toptek import filters
from toptek.gui import DARK_PALETTE, TEXT_WIDGET_DEFAULTS
from toptek.lmstudio import LMStudioClient, build_client


_BaseFrame = ttk.Frame if ttk is not None else object


class LiveTab(_BaseFrame):
    """Simple chat interface for LM Studio."""

    def __init__(
        self,
        master: ttk.Notebook,
        config: Mapping[str, object],
        *,
        client: LMStudioClient | None = None,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available for LiveTab")
        super().__init__(master, style="DashboardBackground.TFrame")  # type: ignore[misc]
        self._config = dict(config)
        self._client = client or build_client(self._config)
        self._messages: MutableSequence[Mapping[str, str]] = []

        self.system_prompt_text = tk.Text(self, height=4, wrap="word")
        self.chat_log = tk.Text(self, height=14, wrap="word", state="disabled")
        self.input_var = tk.StringVar()
        self.error_var = tk.StringVar(value="")
        self.latency_var = tk.StringVar(value="Latency: -- ms")
        self.model_var = tk.StringVar(value=str(self._config.get("model", "")))

        self._build_ui()
        self._load_system_prompt()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="DashboardBackground.TFrame")
        header.pack(fill=tk.X, padx=16, pady=(16, 8))

        ttk.Label(
            header,
            text="LM Studio Live",
            style="Header.TLabel",
        ).pack(side=tk.LEFT)

        ttk.Label(
            header,
            textvariable=self.model_var,
            style="StatusInfo.TLabel",
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Label(
            header,
            textvariable=self.latency_var,
            style="StatusInfo.TLabel",
        ).pack(side=tk.RIGHT)

        prompt_frame = ttk.LabelFrame(self, text="System prompt")
        prompt_frame.pack(fill=tk.X, padx=16, pady=(0, 12))
        self._style_text_widget(self.system_prompt_text)
        self.system_prompt_text.pack(in_=prompt_frame, fill=tk.X, padx=8, pady=8)

        chat_frame = ttk.LabelFrame(self, text="Conversation")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))
        self._style_text_widget(self.chat_log)
        self.chat_log.pack(in_=chat_frame, fill=tk.BOTH, expand=True, padx=8, pady=8)

        input_frame = ttk.Frame(self, style="DashboardBackground.TFrame")
        input_frame.pack(fill=tk.X, padx=16, pady=(0, 16))

        entry = ttk.Entry(input_frame, textvariable=self.input_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", self._handle_enter)

        ttk.Button(input_frame, text="Send", command=self.send_message).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        ttk.Label(
            self,
            textvariable=self.error_var,
            foreground=DARK_PALETTE["danger"],
            style="StatusInfo.TLabel",
        ).pack(fill=tk.X, padx=16)

    def _load_system_prompt(self) -> None:
        prompt = str(self._config.get("system_prompt", ""))
        if prompt:
            self.system_prompt_text.insert("1.0", prompt)

    def _style_text_widget(self, widget: tk.Text) -> None:
        widget.configure(**TEXT_WIDGET_DEFAULTS)
        widget.configure(state="normal")

    def _handle_enter(self, event: tk.Event) -> None:
        self.send_message()
        return None

    def send_message(self) -> None:
        message = self.input_var.get().strip()
        if not message:
            return
        self.input_var.set("")
        self.error_var.set("")

        user_message = {"role": "user", "content": message}
        self._messages.append(user_message)
        self._append_to_log("You", message)

        try:
            response, latency_ms = self._dispatch()
        except Exception as exc:  # pragma: no cover - UI path
            self.error_var.set(str(exc))
            return

        self.latency_var.set(f"Latency: {latency_ms:.0f} ms")
        self._messages.append({"role": "assistant", "content": response})
        self._append_to_log("Assistant", response)

    def _dispatch(self) -> tuple[str, float]:
        prompt = self.system_prompt_text.get("1.0", tk.END).strip()
        messages: List[Mapping[str, str]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(self._messages)
        if self._client is None:
            raise RuntimeError("LM Studio client unavailable")
        return self._client.chat(messages)

    def _append_to_log(self, speaker: str, text: str) -> None:
        safe = filters.redact(text)
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, f"{speaker}: {safe}\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see(tk.END)

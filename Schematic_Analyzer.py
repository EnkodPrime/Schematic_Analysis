from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any


APP_TITLE = "Schematic Analyze"
CONFIG_FILE = "schematic_config.json"
DEFAULT_MODEL_DIR = "models"
PROVIDER_LOCAL = "Local GGUF"
PROVIDER_OPENAI = "OpenAI"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Follow the user's request exactly and answer briefly and directly unless asked otherwise. "
    "Do not repeat yourself. "
    "If the user asks a simple factual question, return only the answer."
)
OPENAI_MODEL_OPTIONS = (
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o4-mini",
)
GENERATION_PRESETS = {
    "Max Deterministic": {
        "temperature": "0.0",
        "top_p": "0.1",
        "top_k": "1",
        "repetition_penalty": "1.0",
        "do_sample": False,
    },
    "Normal": {
        "temperature": "0.7",
        "top_p": "0.95",
        "top_k": "50",
        "repetition_penalty": "1.1",
        "do_sample": True,
    },
    "Max Non-Deterministic": {
        "temperature": "1.3",
        "top_p": "1.0",
        "top_k": "0",
        "repetition_penalty": "1.0",
        "do_sample": True,
    },
}
MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
}

TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
}
INLINE_CONTEXT_CHAR_LIMIT = 30000
DEFAULT_N_CTX = 4096
CHARS_PER_TOKEN_ESTIMATE = 4
MIN_CONTEXT_TOKENS_FOR_FILES = 256
DEFAULT_RESERVED_OUTPUT_TOKENS = 512


class SchematicAnalyzeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("720x420")
        self.root.minsize(640, 360)

        self.base_dir = Path(__file__).resolve().parent
        self.config_path = self.base_dir / CONFIG_FILE

        self.config_data = self._load_config()
        saved_model_dir = self.config_data.get("model_dir")
        self.model_dir = Path(saved_model_dir) if saved_model_dir else self.base_dir / DEFAULT_MODEL_DIR

        self.model_dir_var = tk.StringVar(value=str(self.model_dir))
        self.provider_var = tk.StringVar(value=self.config_data.get("provider", PROVIDER_LOCAL))
        self.openai_api_key_var = tk.StringVar(value=self.config_data.get("openai_api_key", ""))
        self.openai_model_var = tk.StringVar(value=self.config_data.get("openai_model", "gpt-4.1-mini"))
        self.selected_model_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.generation_time_var = tk.StringVar(value="Generation time: N/A")
        self.loaded_model_var = tk.StringVar(value="No model loaded")
        self.system_prompt_var = tk.StringVar(
            value=self.config_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        self.max_new_tokens_var = tk.StringVar(value="256")
        self.n_ctx_var = tk.StringVar(value=str(self.config_data.get("n_ctx", DEFAULT_N_CTX)))
        self.auto_max_tokens_var = tk.BooleanVar(value=False)
        self.temperature_var = tk.StringVar(value="0.7")
        self.top_p_var = tk.StringVar(value="0.95")
        self.top_k_var = tk.StringVar(value="50")
        self.repetition_penalty_var = tk.StringVar(value="1.1")
        self.do_sample_var = tk.BooleanVar(value=True)
        self.generation_preset_var = tk.StringVar(value="Normal")
        self.pdf_path_var = tk.StringVar(value=self.config_data.get("pdf_path", ""))
        self.json_path_var = tk.StringVar(value=self.config_data.get("json_path", ""))
        self.use_inline_context_var = tk.BooleanVar(
            value=bool(self.config_data.get("use_inline_context", True))
        )
        self.include_pdf_context_var = tk.BooleanVar(
            value=bool(self.config_data.get("include_pdf_context", True))
        )
        self.include_json_context_var = tk.BooleanVar(
            value=bool(self.config_data.get("include_json_context", True))
        )

        self._models_index: dict[str, Path] = {}
        self.loaded_model_path: Path | None = None
        self.generator = None

        self._build_ui()
        self.refresh_models()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        selection_frame = ttk.Frame(container)
        selection_frame.pack(fill="x")

        ttk.Radiobutton(
            selection_frame,
            text=PROVIDER_LOCAL,
            value=PROVIDER_LOCAL,
            variable=self.provider_var,
            command=self._on_provider_changed,
        ).grid(row=0, column=0, padx=(0, 8))

        directory_entry = ttk.Entry(
            selection_frame,
            textvariable=self.model_dir_var,
            state="readonly",
        )
        directory_entry.grid(row=0, column=1, sticky="ew")

        browse_button = ttk.Button(
            selection_frame,
            text="Browse",
            command=self.choose_model_directory,
        )
        browse_button.grid(row=0, column=2, padx=(8, 0))

        refresh_button = ttk.Button(
            selection_frame,
            text="Refresh",
            command=self.refresh_models,
        )
        refresh_button.grid(row=0, column=3, padx=(8, 0))

        openai_frame = ttk.LabelFrame(container, text="OpenAI Settings", padding=12)
        openai_frame.pack(fill="x", pady=(12, 0))

        ttk.Radiobutton(
            openai_frame,
            text=PROVIDER_OPENAI,
            value=PROVIDER_OPENAI,
            variable=self.provider_var,
            command=self._on_provider_changed,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))

        ttk.Label(openai_frame, text="API key:").grid(row=0, column=1, sticky="w")
        self.openai_api_key_entry = ttk.Entry(
            openai_frame,
            textvariable=self.openai_api_key_var,
            show="*",
        )
        self.openai_api_key_entry.grid(row=0, column=2, sticky="ew", padx=(8, 12))

        ttk.Label(openai_frame, text="Model:").grid(row=0, column=3, sticky="w")
        self.openai_model_entry = ttk.Combobox(
            openai_frame,
            textvariable=self.openai_model_var,
            values=OPENAI_MODEL_OPTIONS,
        )
        self.openai_model_entry.grid(row=0, column=4, sticky="ew", padx=(8, 12))

        save_openai_button = ttk.Button(
            openai_frame,
            text="Save OpenAI Settings",
            command=self.save_openai_settings,
        )
        save_openai_button.grid(row=0, column=5)
        openai_frame.columnconfigure(2, weight=1)
        openai_frame.columnconfigure(4, weight=1)

        self.model_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.selected_model_var,
            state="readonly",
        )
        self.model_combo.grid(row=0, column=4, sticky="ew", padx=(16, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        show_button = ttk.Button(
            selection_frame,
            text="Show Selected Model",
            command=self.show_selected_model,
        )
        show_button.grid(row=0, column=5, padx=(8, 0))

        use_model_button = ttk.Button(
            selection_frame,
            text="Use Model",
            command=self.use_selected_model,
        )
        use_model_button.grid(row=0, column=6, padx=(8, 0))

        selection_frame.columnconfigure(1, weight=1)
        selection_frame.columnconfigure(4, weight=1)

        input_files_frame = ttk.LabelFrame(container, text="Input Files", padding=12)
        input_files_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(input_files_frame, text="PDF file:").grid(row=0, column=0, sticky="w")
        pdf_entry = ttk.Entry(input_files_frame, textvariable=self.pdf_path_var)
        pdf_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        pdf_browse_button = ttk.Button(
            input_files_frame,
            text="Browse",
            command=self.choose_pdf_file,
        )
        pdf_browse_button.grid(row=0, column=2, sticky="w")

        ttk.Label(input_files_frame, text="JSON file:").grid(row=0, column=3, sticky="w", padx=(16, 0))
        json_entry = ttk.Entry(input_files_frame, textvariable=self.json_path_var)
        json_entry.grid(row=0, column=4, sticky="ew", padx=(8, 8))
        json_browse_button = ttk.Button(
            input_files_frame,
            text="Browse",
            command=self.choose_json_file,
        )
        json_browse_button.grid(row=0, column=5, sticky="w")
        use_inline_context_checkbox = ttk.Checkbutton(
            input_files_frame,
            text="Use Inline Context",
            variable=self.use_inline_context_var,
            command=self._on_use_inline_context_changed,
        )
        use_inline_context_checkbox.grid(row=0, column=6, sticky="w", padx=(16, 0))
        include_pdf_checkbox = ttk.Checkbutton(
            input_files_frame,
            text="Include PDF",
            variable=self.include_pdf_context_var,
            command=self._on_include_file_context_changed,
        )
        include_pdf_checkbox.grid(row=0, column=7, sticky="w", padx=(12, 0))
        include_json_checkbox = ttk.Checkbutton(
            input_files_frame,
            text="Include JSON",
            variable=self.include_json_context_var,
            command=self._on_include_file_context_changed,
        )
        include_json_checkbox.grid(row=0, column=8, sticky="w", padx=(12, 0))
        input_files_frame.columnconfigure(1, weight=1)
        input_files_frame.columnconfigure(4, weight=1)

        action_frame = ttk.Frame(container, padding=(0, 8, 0, 0))
        action_frame.pack(fill="x")

        prompt_frame = ttk.LabelFrame(container, text="Prompt", padding=12)
        prompt_frame.pack(fill="both", expand=True, pady=(12, 0))

        prompt_toolbar = ttk.Frame(prompt_frame)
        prompt_toolbar.pack(fill="x", pady=(0, 8))

        ttk.Label(prompt_toolbar, text="Max new tokens:").pack(side="left")

        max_tokens_entry = ttk.Entry(
            prompt_toolbar,
            textvariable=self.max_new_tokens_var,
            width=8,
        )
        max_tokens_entry.pack(side="left", padx=(8, 0))

        auto_max_tokens_checkbox = ttk.Checkbutton(
            prompt_toolbar,
            text="Auto (until stop)",
            variable=self.auto_max_tokens_var,
        )
        auto_max_tokens_checkbox.pack(side="left", padx=(8, 0))

        ttk.Label(prompt_toolbar, text="Context window (n_ctx):").pack(side="left", padx=(12, 0))
        n_ctx_entry = ttk.Entry(
            prompt_toolbar,
            textvariable=self.n_ctx_var,
            width=8,
        )
        n_ctx_entry.pack(side="left", padx=(8, 0))

        generate_button = ttk.Button(
            prompt_toolbar,
            text="Generate Response",
            command=self.generate_response,
        )
        generate_button.pack(side="left", padx=(12, 0))

        system_prompt_row = ttk.Frame(prompt_frame)
        system_prompt_row.pack(fill="x", pady=(0, 8))

        ttk.Label(system_prompt_row, text="System instructions:").pack(side="left")
        system_prompt_entry = ttk.Entry(
            system_prompt_row,
            textvariable=self.system_prompt_var,
        )
        system_prompt_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        advanced_section = ttk.Frame(prompt_frame)
        advanced_section.pack(fill="x", pady=(0, 8))

        advanced_toolbar = ttk.Frame(advanced_section)
        advanced_toolbar.grid(row=0, column=0, sticky="ew")

        ttk.Label(advanced_toolbar, text="Profile:").pack(side="left")
        self.generation_preset_combo = ttk.Combobox(
            advanced_toolbar,
            textvariable=self.generation_preset_var,
            values=tuple(GENERATION_PRESETS.keys()),
            state="readonly",
            width=24,
        )
        self.generation_preset_combo.pack(side="left", padx=(6, 12))
        self.generation_preset_combo.bind("<<ComboboxSelected>>", self._on_generation_preset_changed)

        ttk.Label(advanced_toolbar, text="Temperature:").pack(side="left")
        temperature_entry = ttk.Entry(
            advanced_toolbar,
            textvariable=self.temperature_var,
            width=6,
        )
        temperature_entry.pack(side="left", padx=(6, 10))

        ttk.Label(advanced_toolbar, text="Top-p:").pack(side="left")
        top_p_entry = ttk.Entry(
            advanced_toolbar,
            textvariable=self.top_p_var,
            width=6,
        )
        top_p_entry.pack(side="left", padx=(6, 10))

        ttk.Label(advanced_toolbar, text="Top-k:").pack(side="left")
        top_k_entry = ttk.Entry(
            advanced_toolbar,
            textvariable=self.top_k_var,
            width=6,
        )
        top_k_entry.pack(side="left", padx=(6, 10))

        ttk.Label(advanced_toolbar, text="Repetition penalty:").pack(side="left")
        repetition_penalty_entry = ttk.Entry(
            advanced_toolbar,
            textvariable=self.repetition_penalty_var,
            width=6,
        )
        repetition_penalty_entry.pack(side="left", padx=(6, 10))

        do_sample_checkbox = ttk.Checkbutton(
            advanced_toolbar,
            text="Do sample",
            variable=self.do_sample_var,
        )
        do_sample_checkbox.pack(side="left")

        self._apply_generation_preset(self.generation_preset_var.get())
        self._attach_tooltips(
            max_tokens_entry=max_tokens_entry,
            n_ctx_entry=n_ctx_entry,
            auto_max_tokens_checkbox=auto_max_tokens_checkbox,
            system_prompt_entry=system_prompt_entry,
            temperature_entry=temperature_entry,
            top_p_entry=top_p_entry,
            top_k_entry=top_k_entry,
            repetition_penalty_entry=repetition_penalty_entry,
            do_sample_checkbox=do_sample_checkbox,
        )

        self.prompt_text = tk.Text(
            prompt_frame,
            height=6,
            wrap="word",
            font=("Consolas", 10),
        )
        self.prompt_text.pack(fill="both", expand=True)

        response_frame = ttk.LabelFrame(container, text="Model Response", padding=12)
        response_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.response_text = tk.Text(
            response_frame,
            height=8,
            wrap="word",
            font=("Consolas", 10),
        )
        self.response_text.pack(fill="both", expand=True)

        generation_time_label = ttk.Label(
            container,
            textvariable=self.generation_time_var,
            anchor="w",
        )
        generation_time_label.pack(fill="x", pady=(6, 0))

        status_bar = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
        )
        status_bar.pack(fill="x", pady=(12, 0))
        self._on_provider_changed()

    def choose_model_directory(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="Select Model Directory",
            initialdir=self.model_dir if self.model_dir.exists() else self.base_dir,
        )
        if not selected_dir:
            return

        self.model_dir = Path(selected_dir)
        self.model_dir_var.set(str(self.model_dir))
        self._save_model_dir()
        self.refresh_models()

    def choose_pdf_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Select PDF File",
            initialdir=self.base_dir,
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not selected_file:
            return
        self.pdf_path_var.set(selected_file)
        self._save_config()
        self.status_var.set("PDF file selected")

    def choose_json_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Select JSON File",
            initialdir=self.base_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected_file:
            return
        self.json_path_var.set(selected_file)
        self._save_config()
        self.status_var.set("JSON file selected")

    def _on_use_inline_context_changed(self) -> None:
        self._save_config()
        self.status_var.set(
            "Inline context enabled" if self.use_inline_context_var.get() else "Inline context disabled"
        )

    def _on_include_file_context_changed(self) -> None:
        self._save_config()
        self.status_var.set(
            f"Inline files: PDF={'on' if self.include_pdf_context_var.get() else 'off'}, "
            f"JSON={'on' if self.include_json_context_var.get() else 'off'}"
        )

    def refresh_models(self) -> None:
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.status_var.set(f"Created model directory: {self.model_dir}")

        model_paths = self._find_model_paths(self.model_dir)
        display_names = [path.name for path in model_paths]

        self.model_combo["values"] = display_names

        if display_names:
            self.selected_model_var.set(display_names[0])
            self._update_details(model_paths[0])
            self.status_var.set(f"Loaded {len(display_names)} model(s)")
        else:
            self.selected_model_var.set("")
            self._set_details(
                "No models were found.\n\n"
                f"Put your local models in:\n{self.model_dir}\n\n"
                "Expected standalone `.gguf` files (llama.cpp load mode)."
            )
            self.status_var.set("No models found")

        self._models_index = {path.name: path for path in model_paths}

    def save_openai_settings(self) -> None:
        self._save_config()
        self.status_var.set("OpenAI settings saved")

    def show_selected_model(self) -> None:
        selected_name = self.selected_model_var.get()
        if not selected_name:
            messagebox.showwarning("No Model Selected", "Please select a model first.")
            return

        selected_path = self._models_index.get(selected_name)
        if not selected_path:
            messagebox.showerror("Model Missing", "The selected model was not found.")
            return

        self._update_details(selected_path)
        messagebox.showinfo("Selected Model", f"Current model:\n{selected_path}")

    def use_selected_model(self) -> None:
        if self.provider_var.get() != PROVIDER_LOCAL:
            messagebox.showinfo(
                "Provider Active",
                "The active provider is OpenAI. Switch to 'Local GGUF' to load a local model.",
            )
            return

        selected_name = self.selected_model_var.get()
        if not selected_name:
            messagebox.showwarning("No Model Selected", "Please select a model first.")
            return

        selected_path = self._models_index.get(selected_name)
        if not selected_path:
            messagebox.showerror("Model Missing", "The selected model was not found.")
            return
        try:
            n_ctx = int(self.n_ctx_var.get())
        except ValueError:
            messagebox.showerror("Invalid Number", "Context window (n_ctx) must be an integer.")
            return
        if n_ctx < 512:
            messagebox.showerror("Invalid Number", "Context window (n_ctx) must be at least 512.")
            return

        self.loaded_model_var.set(f"Loading: {selected_path.name}")
        self.status_var.set("Loading model with llama.cpp...")
        self._set_response("Loading model. Please wait...")
        self._save_config()

        worker = threading.Thread(
            target=self._load_model_worker,
            args=(selected_path, n_ctx),
            daemon=True,
        )
        worker.start()

    def generate_response(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Enter a prompt before generating.")
            return

        if self.auto_max_tokens_var.get():
            max_new_tokens = -1
        else:
            try:
                max_new_tokens = int(self.max_new_tokens_var.get())
            except ValueError:
                messagebox.showerror("Invalid Number", "Max new tokens must be an integer.")
                return
        try:
            temperature = float(self.temperature_var.get())
            top_p = float(self.top_p_var.get())
            top_k = int(self.top_k_var.get())
            repetition_penalty = float(self.repetition_penalty_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Sampling Settings",
                "Temperature, Top-p, Top-k and Repetition penalty must be valid numbers.",
            )
            return

        try:
            n_ctx = int(self.n_ctx_var.get())
        except ValueError:
            messagebox.showerror("Invalid Number", "Context window (n_ctx) must be an integer.")
            return

        if not self.auto_max_tokens_var.get() and max_new_tokens <= 0:
            messagebox.showerror("Invalid Number", "Max new tokens must be greater than 0.")
            return
        if n_ctx < 512:
            messagebox.showerror("Invalid Number", "Context window (n_ctx) must be at least 512.")
            return
        if temperature < 0:
            messagebox.showerror("Invalid Number", "Temperature must be 0 or higher.")
            return
        if not 0 <= top_p <= 1:
            messagebox.showerror("Invalid Number", "Top-p must be between 0 and 1.")
            return
        if top_k < 0:
            messagebox.showerror("Invalid Number", "Top-k must be 0 or higher.")
            return
        if repetition_penalty <= 0:
            messagebox.showerror("Invalid Number", "Repetition penalty must be greater than 0.")
            return

        provider = self.provider_var.get()
        if provider == PROVIDER_LOCAL and (self.generator is None or self.loaded_model_path is None):
            messagebox.showwarning("No Model Loaded", "Load a model with 'Use Model' first.")
            return

        if provider == PROVIDER_OPENAI and not self.openai_api_key_var.get().strip():
            messagebox.showwarning("Missing API Key", "Enter an OpenAI API key first.")
            return

        if self.use_inline_context_var.get():
            try:
                prompt_with_context = self._build_prompt_with_inline_context(
                    user_prompt=prompt,
                    n_ctx=n_ctx,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as error:
                messagebox.showerror("Inline Context Error", str(error))
                return
        else:
            prompt_with_context = prompt

        self.status_var.set("Generating response...")
        self._set_response("Generating response. Please wait...")
        self.generation_time_var.set("Generation time: calculating...")

        if provider == PROVIDER_OPENAI:
            worker = threading.Thread(
                target=self._generate_openai_response_worker,
                args=(prompt_with_context, max_new_tokens, temperature, top_p),
                daemon=True,
            )
        else:
            worker = threading.Thread(
                target=self._generate_response_worker,
                args=(
                    prompt_with_context,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    self.do_sample_var.get(),
                ),
                daemon=True,
            )
        worker.start()

    def _build_prompt_with_inline_context(
        self,
        user_prompt: str,
        n_ctx: int,
        max_new_tokens: int,
    ) -> str:
        context_sections: list[str] = []

        pdf_path_raw = self.pdf_path_var.get().strip()
        if self.include_pdf_context_var.get() and pdf_path_raw:
            pdf_path = Path(pdf_path_raw)
            if not pdf_path.exists() or not pdf_path.is_file():
                raise FileNotFoundError(f"Selected PDF file was not found:\n{pdf_path}")
            pdf_text = self._extract_pdf_text(pdf_path)
            if pdf_text:
                context_sections.append(
                    f"PDF source: {pdf_path}\n"
                    f"PDF content:\n{pdf_text}"
                )

        json_path_raw = self.json_path_var.get().strip()
        if self.include_json_context_var.get() and json_path_raw:
            json_path = Path(json_path_raw)
            if not json_path.exists() or not json_path.is_file():
                raise FileNotFoundError(f"Selected JSON file was not found:\n{json_path}")
            json_text = self._extract_json_text(json_path)
            if json_text:
                context_sections.append(
                    f"JSON source: {json_path}\n"
                    f"JSON content:\n{json_text}"
                )

        if not context_sections:
            return user_prompt

        combined_context = "\n\n".join(context_sections)
        prompt_tokens_estimate = self._estimate_token_count(user_prompt)
        reserved_output_tokens = (
            DEFAULT_RESERVED_OUTPUT_TOKENS if max_new_tokens < 0 else max_new_tokens
        )
        available_context_tokens = n_ctx - prompt_tokens_estimate - reserved_output_tokens - 128

        if available_context_tokens < MIN_CONTEXT_TOKENS_FOR_FILES:
            raise ValueError(
                "Not enough free context tokens for inline files. "
                "Reduce Max new tokens, disable one file, or increase n_ctx."
            )

        max_context_chars_by_tokens = available_context_tokens * CHARS_PER_TOKEN_ESTIMATE
        max_context_chars = min(INLINE_CONTEXT_CHAR_LIMIT, max_context_chars_by_tokens)

        if len(combined_context) > max_context_chars:
            combined_context = combined_context[:max_context_chars]
            combined_context += "\n\n[Context truncated to fit model context window.]"

        return (
            "Use the following file context when answering.\n\n"
            f"{combined_context}\n\n"
            "User request:\n"
            f"{user_prompt}"
        )

    def _estimate_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "PDF inline context requires 'pypdf'. Install it with:\n"
                "pip install pypdf"
            ) from error

        reader = PdfReader(str(pdf_path))
        text_parts: list[str] = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()

    def _extract_json_text(self, json_path: Path) -> str:
        with json_path.open("r", encoding="utf-8") as file:
            parsed = json.load(file)
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _on_model_selected(self, _event: tk.Event) -> None:
        selected_name = self.selected_model_var.get()
        selected_path = self._models_index.get(selected_name)
        if selected_path:
            self._update_details(selected_path)

    def _on_generation_preset_changed(self, _event: tk.Event) -> None:
        self._apply_generation_preset(self.generation_preset_var.get())

    def _apply_generation_preset(self, preset_name: str) -> None:
        preset = GENERATION_PRESETS.get(preset_name)
        if not preset:
            return
        self.temperature_var.set(preset["temperature"])
        self.top_p_var.set(preset["top_p"])
        self.top_k_var.set(preset["top_k"])
        self.repetition_penalty_var.set(preset["repetition_penalty"])
        self.do_sample_var.set(bool(preset["do_sample"]))

    def _attach_tooltips(
        self,
        max_tokens_entry: ttk.Entry,
        n_ctx_entry: ttk.Entry,
        auto_max_tokens_checkbox: ttk.Checkbutton,
        system_prompt_entry: ttk.Entry,
        temperature_entry: ttk.Entry,
        top_p_entry: ttk.Entry,
        top_k_entry: ttk.Entry,
        repetition_penalty_entry: ttk.Entry,
        do_sample_checkbox: ttk.Checkbutton,
    ) -> None:
        profile_hint = (
            "Generation presets:\n"
            "Max Deterministic: temp=0.0, top_p=0.1, top_k=1, do_sample=off\n"
            "Normal: temp=0.7, top_p=0.95, top_k=50, rep_penalty=1.1\n"
            "Max Non-Deterministic: temp=1.3, top_p=1.0, top_k=0, do_sample=on"
        )
        self._create_tooltip(self.generation_preset_combo, profile_hint)
        self._create_tooltip(
            max_tokens_entry,
            "Maximum number of tokens to generate.\nHigher value = longer answer, slower generation.",
        )
        self._create_tooltip(
            n_ctx_entry,
            "Model context window used by llama.cpp.\n"
            "Higher value allows more input text but uses more RAM/VRAM and may be slower.",
        )
        self._create_tooltip(
            auto_max_tokens_checkbox,
            "When ON, generation uses automatic length until model stop.\n"
            "Local llama.cpp sends max_tokens=-1.",
        )
        self._create_tooltip(
            system_prompt_entry,
            "High-priority instructions for the model.\n"
            "Use this to control style, language, brevity, and behavior.\n"
            "For instruct/chat GGUF models this strongly affects output quality.",
        )
        self._create_tooltip(
            temperature_entry,
            "Controls randomness.\n0.0-0.3 very deterministic, 0.7 balanced, 1.0+ more creative/noisy.",
        )
        self._create_tooltip(
            top_p_entry,
            "Nucleus sampling threshold (0..1).\nLower = safer/focused, higher = more diverse.",
        )
        self._create_tooltip(
            top_k_entry,
            "Limits token candidates to top K probabilities.\n1 = very deterministic, 40-100 = balanced, 0 = disabled.",
        )
        self._create_tooltip(
            repetition_penalty_entry,
            "Penalizes repeated text.\n1.0 = off, 1.05-1.2 reduces loops/repetition.",
        )
        self._create_tooltip(
            do_sample_checkbox,
            "ON: probabilistic sampling (more variety).\nOFF: greedy-like behavior (more repeatable).",
        )

    def _create_tooltip(self, widget: tk.Widget, text: str) -> None:
        tooltip_window: tk.Toplevel | None = None

        def show_tooltip(_event: tk.Event) -> None:
            nonlocal tooltip_window
            if tooltip_window is not None:
                return
            tooltip_window = tk.Toplevel(self.root)
            tooltip_window.wm_overrideredirect(True)
            x = widget.winfo_rootx() + 16
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            tooltip_window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip_window,
                text=text,
                justify="left",
                padx=6,
                pady=4,
                relief="solid",
                borderwidth=1,
                background="#ffffe0",
            )
            label.pack()

        def hide_tooltip(_event: tk.Event) -> None:
            nonlocal tooltip_window
            if tooltip_window is not None:
                tooltip_window.destroy()
                tooltip_window = None

        widget.bind("<Enter>", show_tooltip, add="+")
        widget.bind("<Leave>", hide_tooltip, add="+")
        widget.bind("<FocusOut>", hide_tooltip, add="+")

    def _update_details(self, model_path: Path) -> None:
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            details = (
                f"Model file: {model_path.name}\n"
                f"Full path: {model_path}\n"
                f"Extension: {model_path.suffix or 'N/A'}\n"
                f"Size: {size_mb:.2f} MB\n"
                "Load mode: GGUF via llama.cpp"
            )
        else:
            total_size = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
            size_mb = total_size / (1024 * 1024)
            config_exists = (model_path / "config.json").exists()
            tokenizer_exists = any((model_path / file_name).exists() for file_name in TOKENIZER_FILES)
            details = (
                f"Model folder: {model_path.name}\n"
                f"Full path: {model_path}\n"
                f"Has config.json: {'Yes' if config_exists else 'No'}\n"
                f"Has tokenizer files: {'Yes' if tokenizer_exists else 'No'}\n"
                f"Total size: {size_mb:.2f} MB"
            )
        self._set_details(details)

    def _set_details(self, text: str) -> None:
        # Details textbox was removed from the UI to keep the layout compact.
        _ = text

    def _set_response(self, text: str) -> None:
        self.response_text.config(state="normal")
        self.response_text.delete("1.0", tk.END)
        self.response_text.insert("1.0", text)
        self.response_text.config(state="normal")

    def _on_provider_changed(self) -> None:
        provider = self.provider_var.get()
        self._save_config()
        if provider == PROVIDER_OPENAI:
            self.loaded_model_var.set(f"Provider: {PROVIDER_OPENAI}")
            self.status_var.set("OpenAI provider selected")
        elif self.loaded_model_path is not None:
            self.loaded_model_var.set(f"Loaded: {self.loaded_model_path.name}")
            self.status_var.set("Local provider selected")
        else:
            self.loaded_model_var.set("No model loaded")
            self.status_var.set("Local provider selected")

    def _find_model_paths(self, folder: Path) -> list[Path]:
        model_paths = [file_path for file_path in folder.rglob("*.gguf") if file_path.is_file()]
        return sorted(model_paths, key=lambda path: path.name.lower())

    def _load_model_worker(self, model_path: Path, n_ctx: int) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            self.root.after(
                0,
                lambda: self._on_model_load_failed(
                    "llama-cpp-python is not installed. Install it with:\n"
                    "pip install llama-cpp-python"
                ),
            )
            return

        try:
            generator = self._build_generator(
                model_path=model_path,
                llama_cls=Llama,
                n_ctx=n_ctx,
            )
        except Exception as error:
            self.root.after(
                0,
                lambda: self._on_model_load_failed(
                    "Failed to load the selected model with llama.cpp.\n\n"
                    f"{error}"
                ),
            )
            return

        self.root.after(0, lambda: self._on_model_loaded(model_path, generator))

    def _build_generator(
        self,
        model_path: Path,
        llama_cls,
        n_ctx: int,
    ):
        gguf_path = self._resolve_gguf_path(model_path)
        return llama_cls(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            chat_format="chatml",
            verbose=False,
        )

    def _resolve_gguf_path(self, model_path: Path) -> Path:
        if model_path.is_file() and model_path.suffix.lower() == ".gguf":
            return model_path

        if model_path.is_dir():
            gguf_files = sorted(
                [path for path in model_path.rglob("*.gguf") if path.is_file()],
                key=lambda path: path.name.lower(),
            )
            if gguf_files:
                return gguf_files[0]

        raise ValueError("No .gguf file found for the selected model.")

    def _on_model_loaded(self, model_path: Path, generator) -> None:
        self.generator = generator
        self.loaded_model_path = model_path
        self.loaded_model_var.set(f"Loaded: {model_path.name}")
        self.status_var.set("Model loaded and ready")
        self._set_response(f"Model '{model_path.name}' is loaded and ready.")

    def _on_model_load_failed(self, error_message: str) -> None:
        self.generator = None
        self.loaded_model_path = None
        self.loaded_model_var.set("No model loaded")
        self.status_var.set("Model loading failed")
        self._set_response(error_message)
        messagebox.showerror("Model Load Error", error_message)

    def _generate_response_worker(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> None:
        started_at = time.perf_counter()
        try:
            effective_temperature = temperature if do_sample else 0.0
            effective_top_p = top_p if do_sample else min(top_p, 0.1)
            effective_top_k = top_k if do_sample else 1
            request_kwargs = {
                "messages": self._build_chat_messages(prompt),
                "temperature": effective_temperature,
                "top_p": effective_top_p,
                "top_k": effective_top_k,
                "repeat_penalty": repetition_penalty,
            }
            if max_new_tokens >= 0:
                request_kwargs["max_tokens"] = max_new_tokens

            completion = self.generator.create_chat_completion(**request_kwargs)
            response_text = self._extract_chat_response_text(completion)
            elapsed_seconds = time.perf_counter() - started_at
        except Exception as error:
            self.root.after(
                0,
                lambda: self._on_generation_failed(str(error)),
            )
            return

        self.root.after(0, lambda: self._on_generation_finished(response_text, elapsed_seconds))

    def _build_chat_messages(self, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_prompt = self.system_prompt_var.get().strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_openai_response_worker(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        started_at = time.perf_counter()
        try:
            from openai import OpenAI
        except ImportError:
            self.root.after(
                0,
                lambda: self._on_generation_failed(
                    "openai is not installed. Install it with:\n"
                    "pip install openai"
                ),
            )
            return

        try:
            client = OpenAI(api_key=self.openai_api_key_var.get().strip())
            response = client.responses.create(
                model=self.openai_model_var.get().strip(),
                input=prompt,
                temperature=temperature,
                top_p=top_p,
                **({} if max_new_tokens < 0 else {"max_output_tokens": max_new_tokens}),
            )
            response_text = response.output_text or "The OpenAI model returned no output."
            elapsed_seconds = time.perf_counter() - started_at
        except Exception as error:
            self.root.after(
                0,
                lambda: self._on_generation_failed(str(error)),
            )
            return

        self.root.after(0, lambda: self._on_generation_finished(response_text, elapsed_seconds))

    def _extract_generated_text(self, results: Any, prompt: str) -> str:
        if not results:
            return "The model returned no output."

        first_item = results[0]
        if "generated_text" in first_item:
            generated_text = first_item["generated_text"]
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip() or generated_text
            return generated_text

        if "summary_text" in first_item:
            return first_item["summary_text"]

        return str(first_item)

    def _extract_chat_response_text(self, result: Any) -> str:
        if not result:
            return "The model returned no output."

        choices = result.get("choices", [])
        if not choices:
            return "The model returned no output."

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip() or "The model returned no output."

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            merged = "".join(text_parts).strip()
            return merged or "The model returned no output."

        return str(content).strip() or "The model returned no output."

    def _on_generation_finished(self, response_text: str, elapsed_seconds: float) -> None:
        self._set_response(response_text)
        self.status_var.set("Response generated")
        self.generation_time_var.set(f"Generation time: {elapsed_seconds:.2f} s")

    def _on_generation_failed(self, error_message: str) -> None:
        self._set_response(f"Generation failed:\n{error_message}")
        self.status_var.set("Generation failed")
        self.generation_time_var.set("Generation time: failed")
        messagebox.showerror("Generation Error", error_message)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}

        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

        return data if isinstance(data, dict) else {}

    def _save_model_dir(self) -> None:
        self._save_config()

    def _save_config(self) -> None:
        data = dict(self.config_data)
        data.update(
            {
                "model_dir": str(self.model_dir),
                "provider": self.provider_var.get(),
                "openai_api_key": self.openai_api_key_var.get().strip(),
                "openai_model": self.openai_model_var.get().strip(),
                "system_prompt": self.system_prompt_var.get().strip(),
                "n_ctx": self.n_ctx_var.get().strip(),
                "pdf_path": self.pdf_path_var.get().strip(),
                "json_path": self.json_path_var.get().strip(),
                "use_inline_context": self.use_inline_context_var.get(),
                "include_pdf_context": self.include_pdf_context_var.get(),
                "include_json_context": self.include_json_context_var.get(),
            }
        )
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        self.config_data = data


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = SchematicAnalyzeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

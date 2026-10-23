"""One GGUF model, real residual tensors, and upstream control vectors.

The small ctypes bridge is deliberately restricted to llama-cpp-python 0.3.36.
Its ggml_tensor prefix comes from the bundled llama.cpp 0c1e57098bba43ac29e6e3b677cdceebdd22334f.
No model weights, activation approximations, or alternative inference engine.
"""

import codecs
from contextlib import closing, contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sysconfig
import threading
from time import perf_counter

import numpy as np

from .locking import lock_file

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_VERSION = "residual-last-token-v1"
SUPPORTED_BACKEND = "0.3.36"
UPSTREAM_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"
STEERING_SEMANTICS = {
    "layer_numbering": "1-based decoder blocks; UI layer L maps to native block L-1",
    "extraction": "l_out-(L-1), post-FFN residual before final output norm; last raw-text token including GGUF BOS, no padding or chat template",
    "application": "post-FFN residual addition for all evaluated prompt and generated-token positions",
    "native_control_vectors": "llama_set_adapter_cvec for native blocks 1..N-1; buffer offset (block-1)*dimension",
    "first_block": "ggml scheduler callback l_out-0 with ggml_backend_tensor_get/set; native cvec omits block 0",
    "isolation": "clear adapter, reset full KV/recurrent memory, re-evaluate whole prompt each generation; finally clear adapter and callback injection",
}


def load_llama():
    lib_dir = Path(sysconfig.get_path("platlib")) / "nvidia/cu13/lib"
    for name in ("libcudart.so.13", "libcublas.so.13"):
        path = lib_dir / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    try:
        import llama_cpp
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f"GGUF-Backend nicht verfügbar: {error}. Installation siehe README.md.") from error
    return llama_cpp


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cached_hash(path):
    """SHA256 once per size/mtime/inode; hash-cache writes are atomic."""
    path = Path(path).resolve()
    stat = path.stat()
    signature = [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
    directory = ROOT / ".cache" / "hashes"
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")
    try:
        item = json.loads(cache.read_text())
        if item["signature"] == signature:
            return item["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if [str(path), after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino] != signature:
        raise RuntimeError("Modelldatei wurde während der Hash-Berechnung geändert.")
    item = {"signature": signature, "sha256": digest.hexdigest()}
    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(item), encoding="utf-8")
    temporary.replace(cache)
    return item["sha256"]


def gpu_layers(llama, path, layer_count):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if visible in ("", "-1") or not llama.llama_supports_gpu_offload():
        return 0
    free = _command(["nvidia-smi", "-i", visible, "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    try:
        budget = max(0, int(free) * 1024**2 - 2 * 1024**3) * .8
        return min(layer_count, int(budget / path.stat().st_size * layer_count))
    except (TypeError, ValueError):
        return 0


def regular_layer_count(metadata):
    architecture = metadata["general.architecture"]
    total = int(metadata[f"{architecture}.block_count"])
    nextn = int(metadata.get(f"{architecture}.nextn_predict_layers", 0))
    if not 0 <= nextn < total:
        raise ValueError("Ungültige Decoder-/MTP-Layeranzahl in den Modellmetadaten.")
    return total - nextn


class _TensorPrefix(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("buffer", ctypes.c_void_p),
                ("ne", ctypes.c_int64 * 4), ("nb", ctypes.c_size_t * 4)]


class Backend:
    def __init__(self, config):
        self.config = dict(config)
        self.model = None
        self._lock_file = None
        self._capture = None
        self._injection = None
        self._callback_error = None
        self._callback = None
        self._interrupted = False
        self._vectors_buffer = None
        self.metadata = {}

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def open(self):
        if self.model is not None:
            return
        try:
            path = ROOT / "runs" / ".model.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = path.open("a+b")
            try:
                lock_file(self._lock_file, blocking=False)
            except BlockingIOError as error:
                raise RuntimeError("Ein anderer CHAPPiE-Prozess verwendet bereits das Modell.") from error
            self._lock_file.seek(0)
            self._lock_file.truncate()
            self._lock_file.write(json.dumps({"pid": os.getpid(), "model": self.config["model_path"]}).encode())
            self._lock_file.flush()
            self.llama = load_llama()
            self.path = (ROOT / self.config["model_path"]).resolve()
            if not self.path.is_file():
                raise ValueError(f"Modelldatei fehlt: {self.path}")
            # Read metadata without loading a second copy of the model weights.
            from llama_cpp._internals import LlamaModel
            parameters = self.llama.llama_model_default_params()
            parameters.vocab_only = True
            with closing(LlamaModel(path_model=str(self.path), params=parameters, verbose=False)) as vocab:
                info = vocab.metadata()
                architecture = info["general.architecture"]
                count = regular_layer_count(info)
            requested = self.config["n_gpu_layers"]
            self.gpu_layers = gpu_layers(self.llama, self.path, count) if requested == "auto" else requested
            started = perf_counter()
            self.model = self.llama.Llama(
                model_path=str(self.path), n_ctx=self.config["context_size"],
                n_gpu_layers=self.gpu_layers, split_mode=self.llama.LLAMA_SPLIT_MODE_NONE,
                main_gpu=0, n_threads=self.config["n_threads"] or None, verbose=False,
            )
            self.load_time = perf_counter() - started
            self.layer_count = regular_layer_count(self.model.metadata)
            self.dimension = int(self.model.metadata[f"{architecture}.embedding_length"])
            template = self.model.metadata.get("tokenizer.chat_template")
            if not template:
                raise ValueError("GGUF enthält kein Chat-Template.")
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            self.formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=self.model.detokenize([self.model.token_eos()], special=True).decode(),
                bos_token=self.model.detokenize([self.model.token_bos()], special=True).decode(),
            )
            self._attach_bridge()
            self.metadata = self._metadata(template, architecture)
        except BaseException:
            self.close()
            raise

    def _attach_bridge(self):
        if self.llama.__version__ != SUPPORTED_BACKEND:
            raise RuntimeError(f"Steering-Brücke benötigt llama-cpp-python {SUPPORTED_BACKEND}, gefunden {self.llama.__version__}. Backend nicht automatisch ersetzt.")
        library_dir = Path(self.llama.__file__).parent / "lib"
        candidates = [library_dir / name for name in ("libggml-base.so", "libggml-base.dylib", "ggml-base.dll", "libggml-base.dll")]
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise RuntimeError("Gebündelte ggml-base-Bibliothek fehlt; Steering-Brücke kann nicht geladen werden.")
        self.ggml = ctypes.CDLL(str(path))
        for name, arguments, output in (
            ("ggml_get_name", [ctypes.c_void_p], ctypes.c_char_p),
            ("ggml_backend_tensor_get", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t], None),
            ("ggml_backend_tensor_set", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t], None),
        ):
            function = getattr(self.ggml, name)
            function.argtypes = arguments
            function.restype = output
        self._callback = self.llama.ggml_backend_sched_eval_callback(self._eval_callback)
        # The public Python constructor exposes no cb_eval argument. Replace only
        # its empty context, retaining the single already-loaded weight object.
        from llama_cpp._internals import LlamaContext
        self.model._ctx.close()
        self.model.context_params.cb_eval = self._callback
        self.model._ctx = self.model._stack.enter_context(closing(LlamaContext(
            model=self.model._model, params=self.model.context_params, verbose=False)))

    def _eval_callback(self, tensor, ask, _userdata):
        try:
            if self._interrupted:
                # Ask for the next tensor to obtain a post-evaluation callback;
                # False after evaluation tells ggml to stop the graph.
                return bool(ask)
            if self._capture is None and self._injection is None:
                return False if ask else True
            name = self.ggml.ggml_get_name(tensor).decode("utf-8")
            match = re.fullmatch(r"l_out-(\d+)", name)
            if not match:
                return False if ask else True
            block = int(match[1])
            needed = self._capture is not None or (block == 0 and self._injection is not None)
            if ask:
                return needed
            if not needed:
                return True
            shape = ctypes.cast(tensor, ctypes.POINTER(_TensorPrefix)).contents
            if shape.type != 0 or shape.ne[0] != self.dimension or shape.nb[0] != 4:
                raise RuntimeError(f"Unerwartetes Residualtensor-Layout: {name}.")
            if shape.ne[2] != 1 or shape.ne[3] != 1 or shape.ne[1] < 1:
                raise RuntimeError(f"Unerwartete Tensorform: {name}.")
            if self._capture is not None:
                value = np.empty(self.dimension, dtype=np.float32)
                offset = (shape.ne[1] - 1) * shape.nb[1]
                self.ggml.ggml_backend_tensor_get(tensor, value.ctypes.data, offset, value.nbytes)
                self._capture[block + 1] = value
            if block == 0 and self._injection is not None:
                for row in range(shape.ne[1]):
                    value = np.empty(self.dimension, dtype=np.float32)
                    offset = row * shape.nb[1]
                    self.ggml.ggml_backend_tensor_get(tensor, value.ctypes.data, offset, value.nbytes)
                    value += self._injection
                    self.ggml.ggml_backend_tensor_set(tensor, value.ctypes.data, offset, value.nbytes)
            return True
        except BaseException as error:
            # ctypes callbacks cannot propagate Python exceptions to the caller.
            self._callback_error = error
            return False

    def _check_callback(self):
        if self._interrupted:
            self._interrupted = False
            raise KeyboardInterrupt
        if self._callback_error is not None:
            error, self._callback_error = self._callback_error, None
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error
            raise RuntimeError(f"Aktivierungs-Callback fehlgeschlagen: {error}") from error

    @contextmanager
    def _interruptible(self):
        """Avoid raising SIGINT exceptions inside any upstream ctypes callback.

        In particular llama-cpp-python's logging callback can swallow exceptions
        before our eval callback runs. Record SIGINT and propagate it back in
        Python after native evaluation, restoring the caller's signal handler.
        """
        main_thread = threading.current_thread() is threading.main_thread()
        original = signal.getsignal(signal.SIGINT) if main_thread else None
        if main_thread:
            def interrupted(_signal, _frame):
                self._interrupted = True
            signal.signal(signal.SIGINT, interrupted)
        try:
            try:
                yield
                self._check_callback()
            except BaseException:
                self._check_callback()
                raise
        finally:
            if main_thread:
                signal.signal(signal.SIGINT, original)

    def _metadata(self, template, architecture):
        libs = Path(self.llama.__file__).parent / "lib"
        cpu = _command(["lscpu"])
        cpu_match = re.search(r"Model name:\s*(.+)", cpu or "")
        status = _command(["git", "-C", str(ROOT), "status", "--porcelain"])
        return {
            "model_name": self.model.metadata.get("general.name"), "model_file": str(self.path),
            "model_size_bytes": self.path.stat().st_size, "model_hash": cached_hash(self.path),
            "quantization": "IQ4_XS" if "IQ4_XS" in self.path.name else None,
            "gguf_file_type": self.model.metadata.get("general.file_type"),
            "architecture": architecture, "layer_count": self.layer_count, "dimension": self.dimension,
            "total_block_count": int(self.model.metadata[f"{architecture}.block_count"]),
            "nextn_predict_layers": int(self.model.metadata.get(f"{architecture}.nextn_predict_layers", 0)),
            "context_size": self.config["context_size"], "llama_cpp_python_version": self.llama.__version__,
            "llama_cpp_version": None, "upstream_source_reference": UPSTREAM_COMMIT,
            "backend_library_hashes": {p.name: cached_hash(p) for p in sorted(libs.iterdir()) if p.suffix in (".so", ".dylib", ".dll")},
            "system_info": self.llama.llama_print_system_info().decode(),
            "python_version": platform.python_version(), "os": platform.platform(),
            "cpu": cpu_match[1] if cpu_match else platform.processor() or None,
            "gpu": _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]),
            "n_gpu_layers": self.gpu_layers, "n_threads": self.model.n_threads,
            "git_commit": _command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "git_dirty": bool(status) if status is not None else None,
            "chat_template": template, "chat_template_hash": hashlib.sha256(template.encode()).hexdigest(),
            "steering_semantics": STEERING_SEMANTICS, "bridge_version": BRIDGE_VERSION,
        }

    def format_prompt(self, messages, enable_thinking=True):
        return self.formatter(messages=messages, enable_thinking=bool(enable_thinking)).prompt

    def tokens(self, prompt):
        return self.model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    def reset_steering(self):
        self._injection = None
        self._vectors_buffer = None
        if self.model is not None and self.model._ctx.ctx is not None:
            code = self.llama.llama_set_adapter_cvec(self.model._ctx.ctx, None, 0, self.dimension, -1, -1)
            if code != 0:
                raise RuntimeError(f"Control Vector konnte nicht zurückgesetzt werden ({code}).")

    def reset(self):
        self.reset_steering()
        self.model.reset()

    def _apply(self, vector, layer, alpha):
        if type(layer) is not int or not 1 <= layer <= self.layer_count:
            raise ValueError(f"Layer muss zwischen 1 und {self.layer_count} liegen.")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or not np.isfinite(alpha):
            raise ValueError("Steering-Vektor oder Alpha ungültig.")
        scaled = vector * np.float32(alpha)
        if not np.isfinite(scaled).all():
            raise ValueError("Alpha verursacht nicht-endliche Vektorwerte.")
        if layer == 1:
            self._injection = scaled
            return
        block = layer - 1
        buffer = np.zeros((self.layer_count - 1, self.dimension), dtype=np.float32)
        buffer[block - 1] = scaled
        self._vectors_buffer = buffer
        code = self.llama.llama_set_adapter_cvec(self.model._ctx.ctx,
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buffer.size, self.dimension, block, block)
        if code != 0:
            raise RuntimeError(f"Control Vector wurde abgelehnt ({code}).")

    def extract(self, text):
        with self._interruptible():
            return self._extract(text)

    def _extract(self, text):
        self.reset()
        tokens = self.model.tokenize(text.encode(), add_bos=True, special=False)
        if not tokens or len(tokens) >= self.config["context_size"]:
            raise ValueError("Datasettext ist leer oder überschreitet das Kontextlimit.")
        self._capture = {}
        try:
            try:
                self.model.eval(tokens)
            except BaseException:
                self._check_callback()
                raise
            self._check_callback()
            captured = self._capture
            if set(captured) != set(range(1, self.layer_count + 1)):
                raise RuntimeError("Backend hat nicht alle l_out Residualtensoren geliefert; keine Ersatzaktivierungen erzeugt.")
            if not all(np.isfinite(value).all() for value in captured.values()):
                raise RuntimeError("Nicht-endliche echte Aktivierungen.")
            return captured
        finally:
            self._capture = None
            self.reset()

    def generate(self, messages, config, vector=None, layer=None, alpha=0, on_piece=None):
        with self._interruptible():
            return self._generate(messages, config, vector, layer, alpha, on_piece)

    def _generate(self, messages, config, vector=None, layer=None, alpha=0, on_piece=None):
        started = perf_counter()
        self.reset()
        try:
            prompt = self.format_prompt(messages, config.get("enable_thinking", True))
            tokens = self.tokens(prompt)
            available = self.config["context_size"] - len(tokens) - 1
            if available < config["max_output_tokens"]:
                raise ValueError("Zu wenig Kontextplatz für die konfigurierte Antwort. Bitte /new verwenden.")
            self.model.set_seed(config["seed"])
            if vector is not None:
                self._apply(vector, layer, alpha)
            elif alpha != 0:
                raise ValueError("Steering-Alpha ohne Vektor.")
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            output = []
            first = last = None
            finish_reason = "max_output_tokens"
            stream = self.model.generate(tokens, top_k=config["top_k"], top_p=config["top_p"],
                temp=config["temperature"], min_p=config["min_p"], repeat_penalty=config["repeat_penalty"])
            try:
                for token in stream:
                    self._check_callback()
                    now = perf_counter()
                    if self.llama.llama_vocab_is_eog(self.model._model.vocab, token):
                        finish_reason = "eog"
                        break
                    if first is None:
                        first = now
                    last = now
                    output.append(token)
                    if on_piece:
                        on_piece(decoder.decode(self.model.detokenize([token], special=True)))
                    if len(output) >= config["max_output_tokens"]:
                        break
            except BaseException:
                self._check_callback()
                raise
            finally:
                stream.close()
            if on_piece:
                on_piece(decoder.decode(b"", final=True))
            finished = perf_counter()
            generation = last - first if first is not None else 0.0
            return {"response": self.model.detokenize(output, special=True).decode("utf-8", errors="replace"),
                "prompt_tokens": len(tokens), "output_tokens": len(output), "output_token_ids": output,
                "ttft": first - started if first is not None else None, "generation_time": generation,
                "total_time": finished - started, "tokens_per_second": (len(output)-1)/generation if len(output)>1 and generation>0 else None,
                "seed": config["seed"], "finish_reason": finish_reason,
                "assistant_prefix_has_think": prompt.rstrip().endswith("<think>")}
        finally:
            self.reset_steering()

    def alpha_zero_control(self, config, vector, layer):
        config = dict(config, temperature=0.0)
        messages = [{"role": "user", "content": config.get("prompt") or "Say hello."}]
        baseline = self.generate(messages, config)
        zero = self.generate(messages, config, vector, layer, 0.0)
        after = self.generate(messages, config)
        return {"passed": baseline["output_token_ids"] == zero["output_token_ids"] == after["output_token_ids"],
                "layer": layer, "baseline": baseline, "alpha_zero": zero, "baseline_after": after}

    def close(self):
        try:
            if self.model is not None:
                try:
                    self.reset_steering()
                finally:
                    self.model.close()
                    self.model = None
        finally:
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

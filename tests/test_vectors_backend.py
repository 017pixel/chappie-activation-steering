import ctypes
import json
import signal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from activation_steering.backend import Backend, _TensorPrefix, regular_layer_count
from activation_steering.vectors import VectorSet, build_vectors, load_vectors, save_vectors


class FakeBackend:
    layer_count = 2
    dimension = 2
    config = {"context_size": 8192}
    metadata = {"model_hash": "model", "llama_cpp_python_version": "test"}

    def __init__(self):
        self.calls = 0

    def extract(self, text):
        self.calls += 1
        values = [3., 4.] if text == "target" else [0., 0.]
        return {layer: np.array(values, dtype=np.float32) for layer in (1, 2)}


class VectorTests(unittest.TestCase):
    def test_exact_paired_difference_norm_and_cache_invalidation(self):
        dataset = SimpleNamespace(name="test", hash="dataset1", target_label="target", contrast_label="contrast",
            pairs=[{"target": "target", "contrast": "contrast"}], research_use=False)
        config = {"cache": True, "token_position": "last_non_padding", "normalization": "unit_l2"}
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            first = build_vectors(backend, dataset, config, directory)
            np.testing.assert_allclose(first.vectors[1], [0.6, .8])
            self.assertEqual(first.metadata["raw_norms"]["1"], 5.)
            cached = build_vectors(backend, dataset, config, directory)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(cached.metadata["vector_hash"], first.metadata["vector_hash"])
            dataset.hash = "dataset2"
            build_vectors(backend, dataset, config, directory)
            self.assertEqual(backend.calls, 4)
            config["normalization"] = "none"
            raw = build_vectors(backend, dataset, config, directory)
            np.testing.assert_allclose(raw.vectors[1], [3, 4])
            self.assertEqual(backend.calls, 6)

    def test_zero_vector_rejected(self):
        dataset = SimpleNamespace(name="zero", hash="zero", target_label="target", contrast_label="contrast",
            pairs=[{"target": "target", "contrast": "target"}], research_use=False)
        with self.assertRaisesRegex(ValueError, "Nullvektor"):
            build_vectors(FakeBackend(), dataset, {"cache": False, "token_position": "last_non_padding", "normalization": "unit_l2"})

    def test_artifact_integrity_and_pickle_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            vector_set = VectorSet({1: np.array([1., 0.], dtype=np.float32)}, {"layers": [1], "vector_dimension": 2})
            save_vectors(path, vector_set)
            np.testing.assert_equal(load_vectors(path).vectors[1], [1, 0])
            with path.open("wb") as stream:
                np.savez(stream, metadata=np.array(json.dumps(vector_set.metadata)), layer_1=np.array([0.,1.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Prüfsumme"):
                load_vectors(path)
            with path.open("wb") as stream:
                np.savez(stream, metadata=np.array({"evil": "object"}, dtype=object))
            with self.assertRaises(ValueError):
                load_vectors(path)


class ApplicationTests(unittest.TestCase):
    def test_sigint_is_deferred_and_original_handler_restored(self):
        backend = Backend({})
        original = signal.getsignal(signal.SIGINT)
        with self.assertRaises(KeyboardInterrupt):
            with backend._interruptible():
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                self.assertTrue(backend._interrupted)
                self.assertTrue(backend._eval_callback(None, True, None))
                self.assertFalse(backend._eval_callback(None, False, None))
                raise RuntimeError("native decode aborted")
        self.assertIs(signal.getsignal(signal.SIGINT), original)
        self.assertFalse(backend._interrupted)

    def test_callback_preserves_keyboard_interrupt_across_native_boundary(self):
        backend = Backend({})
        backend._callback_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            backend._check_callback()
        self.assertIsNone(backend._callback_error)

    def test_model_metadata_excludes_mtp_only_from_decoder_sweep(self):
        self.assertEqual(regular_layer_count({"general.architecture":"qwen35", "qwen35.block_count":"65", "qwen35.nextn_predict_layers":"1"}),64)
        self.assertEqual(regular_layer_count({"general.architecture":"qwen3", "qwen3.block_count":"32"}),32)

    def test_ui_layers_map_to_native_blocks_and_buffer_offsets(self):
        backend = Backend({})
        backend.layer_count, backend.dimension = 64, 2
        backend.model = SimpleNamespace(_ctx=SimpleNamespace(ctx=1))
        calls = []
        backend.llama = SimpleNamespace(llama_set_adapter_cvec=lambda *args: calls.append(args) or 0)
        backend._apply([1,2], 1, -2)
        np.testing.assert_equal(backend._injection, [-2,-4])
        for layer in (2,32,64):
            backend._apply([1,2], layer, 3)
            self.assertEqual(calls[-1][-2:], (layer-1,layer-1))
            np.testing.assert_equal(backend._vectors_buffer[layer-2], [3,6])
            self.assertEqual(np.count_nonzero(backend._vectors_buffer), 2)
        for layer in (0,65,True):
            with self.assertRaises(ValueError):
                backend._apply([1,2],layer,1)

    def test_callback_extracts_last_row_and_injects_all_rows(self):
        backend = Backend({})
        backend.dimension = 2
        memory = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        prefix = _TensorPrefix(type=0, ne=(ctypes.c_int64*4)(2,2,1,1), nb=(ctypes.c_size_t*4)(4,8,16,16))
        def read(_tensor, dest, offset, size):
            ctypes.memmove(dest, memory.ctypes.data + offset, size)
        def write(_tensor, source, offset, size):
            ctypes.memmove(memory.ctypes.data + offset, source, size)
        backend.ggml = SimpleNamespace(ggml_get_name=lambda _:b"l_out-0", ggml_backend_tensor_get=read, ggml_backend_tensor_set=write)
        backend._capture = {}
        backend._injection = np.array([10.,20.],dtype=np.float32)
        pointer = ctypes.cast(ctypes.pointer(prefix), ctypes.c_void_p)
        self.assertTrue(backend._eval_callback(pointer,True,None))
        self.assertTrue(backend._eval_callback(pointer,False,None))
        np.testing.assert_equal(backend._capture[1], [3,4])
        np.testing.assert_equal(memory, [[11,22],[13,24]])
        prefix.type = 1
        self.assertFalse(backend._eval_callback(pointer,False,None))
        with self.assertRaisesRegex(RuntimeError,"Callback"):
            backend._check_callback()


if __name__ == "__main__":
    unittest.main()

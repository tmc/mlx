import threading
import unittest

import mlx.core as mx


class TestDebug(unittest.TestCase):
    def test_label_context(self):
        x = mx.array([1.0])
        mx.debug.set_label(x, "input")
        with mx.debug.label("forward"):
            mx.eval(x + 1)
        mx.debug.remove_label(x)

    def test_group_context(self):
        stream = mx.default_stream(mx.default_device())
        mx.debug.set_stream_label(stream, "test")
        with mx.debug.group("forward", stream):
            mx.eval(mx.array([1.0]) + 1)

    def test_stream_label_lifecycle(self):
        stream = mx.default_stream(mx.default_device())
        mx.debug.set_stream_label(stream, "temporary")
        mx.debug.remove_stream_label(stream)

    def test_contexts_are_thread_local(self):
        errors = []

        def run():
            try:
                with mx.debug.label("worker"):
                    mx.eval(mx.array([1.0]) + 1)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

Backend Debugging
=================

.. currentmodule:: mlx.core

The ``mlx.core.debug`` API attaches names to MLX work without depending on the
selected GPU backend. Metal exposes the names in Xcode captures. CUDA exposes
them as NVTX ranges and CUDA stream names.

Python
------

Labels describe evaluation scopes. MLX captures the active label and group
scope when a lazy array is constructed, so a scope around graph construction
is sufficient even when evaluation is deferred or moved to another thread.
If construction and evaluation use different scopes, the captured
construction-time scope takes precedence; evaluation-time thread-local state
is used only for arrays constructed outside a scope.

.. code-block:: python

   import mlx.core as mx

   x = mx.ones((32, 32))
   y = mx.ones((32, 32))
   mx.debug.set_label(x, "input")

   with mx.debug.label("training"):
       z = x @ y
       mx.eval(z)

   mx.debug.remove_label(x)

``mx.debug.group`` creates a logical group for a stream. Groups survive GPU
command-buffer and encoder boundaries. The group stack is thread-local; a
group pushed on one thread does not affect evaluation on another thread.

.. code-block:: python

   stream = mx.default_stream(mx.default_device())
   mx.debug.set_stream_label(stream, "model")
   with mx.debug.group("forward", stream):
       mx.eval(x + y)
   mx.debug.remove_stream_label(stream)

C++
---

The C++ API is declared in ``mlx/debug.h``.

.. code-block:: cpp

   #include "mlx/debug.h"
   #include "mlx/mlx.h"

   using namespace mlx::core;

   auto x = array({1.0f, 2.0f});
   debug::set_label(x, "input");
   {
     debug::ScopedLabel scope("forward");
     auto y = exp(x);
     eval(y);
   }
   debug::remove_label(x);

Backend behavior
----------------

Metal applies array labels to ``MTLBuffer`` objects, stream labels to each
thread's command queue, and logical groups to each compute encoder. Build with
``-DMLX_METAL_DEBUG=ON`` to compile automatic array and operation names.
Explicit labels remain available without that option. Set
``MLX_DEBUG_AUTOMATIC_LABELS=1`` at runtime to enable automatic names in a
debug build. Metal capture remains available through
:func:`metal.start_capture` and :func:`metal.stop_capture`.

CUDA applies evaluation labels and groups as NVTX ranges and applies stream
labels to each thread's ``cudaStream_t``. View them with Nsight Systems or
another NVTX-aware profiler.

When no labels or groups are active, evaluation does not allocate strings or
touch an object registry. Per-operation automatic labeling is disabled by
default in all build modes.

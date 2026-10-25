"""Builders shared by several chapters. Most builders live in the chapter cell that uses them.

Importing this module registers its `@artifact` functions.

Every builder must be self-contained: it may use only its parameters, builtins and imports made inside
the function (absolute imports). That is checked when it is registered, and it is what lets the code
hash cover just the function and the `convfinqa` modules it imports, so editing another builder in
this file never makes an artifact stale. It also keeps heavy imports (torch, transformers) out of
module import time, which `status` and `refresh` pay for.
"""

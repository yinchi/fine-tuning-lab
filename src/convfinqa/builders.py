"""Registered artifact builders (``@artifact`` functions) for data, training and evaluation.

Importing this module registers them, so keep imports of heavy libraries (torch, transformers)
inside the builder functions: ``status`` and ``refresh`` import this module too.
"""

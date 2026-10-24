"""Builders shared by several chapters. Most builders live in the chapter cell that uses them.

Importing this module registers its `@artifact` functions, so keep imports of heavy libraries
(torch, transformers) inside the builder functions: `status` and `refresh` import this module too.
"""

"""The concrete artifact names the book uses.

Name templates in ``builders.py`` are open-ended (``sft-{variant}-{seed}``), so ``build --all``
needs this explicit list. Add a name here when a chapter starts calling ``artifacts.get(name)``;
a test checks that every literal name in ``docs/*.qmd`` is listed.
"""

TARGETS: list[str] = []

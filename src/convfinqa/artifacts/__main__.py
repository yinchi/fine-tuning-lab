"""`python -m convfinqa.artifacts`: run as a package, so builders and the CLI share one registry."""

import sys

from .cli import main

sys.exit(main())

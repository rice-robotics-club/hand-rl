"""Script for generating JSON schema for YAML configurations."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from src.config import Config

schema_dir = Path(__file__).parent.parent / "schemas"

schemas = {
    "schema.json": Config,
}


def main():
    """Generate JSON schema for YAML configurations."""
    schema_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in schemas.items():
        with open(schema_dir / filename, "w") as f:
            json.dump(TypeAdapter(schema).json_schema(), f, indent=2)


if __name__ == "__main__":
    main()

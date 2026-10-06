from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    """Base for all configs: field docstrings become schema descriptions."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

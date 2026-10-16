from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """A base class for configuration objects that we will define for the various types of classes in this project (simulations, systems, environments).

    If you are creating a class that you would like to configure via a YAML config file, then first create a class above with the name [your class name]Config, like so:

    ```python
    # Import this base class
    from src.base_config import Config

    class MySimConfig(Config):
        # VERY IMPORTANT! need to define this to allow the config to point to the correct class
        # based on the location of the class in the codebase, and the class name
        class_name: Literal["src.sim.my_sim:MySim"]

        # You can specify this value in a config file, rather than hard coding it!
        configurable_value: int = 0
        '''This value changes the setup of the simulation.'''

    # The class you are trying to define
    class MySim(VecSystem[MyState, MyAction]):
        # Make sure the config object is a parameter of the constructor
        def __init__(self, config: MySimConfig):
            # You can then use the configuration in your code!
            self._value = config.configurable_value
    ```

    Then, you can add this to the set of config classes that are automatically added to our **config file schema**. This is a file that defines the type of values that you can specify, which will allow your editor (something like VSCode) to give you hints about the fields and values you can specify in the configuration.

    ```python
    # In src/config/config.py:

    # Add your config class to this union of other classes, of the same type.
    SystemConfig = Annotated[
        RukaSimConfig | PianoConfig | HandPianoConfig | MySimConfig, Field(discriminator="class_name")
    ]
    ```

    If you then run the `generate_schema` after adding your class, you will update the `config/schemas/schema.json` file, which will then account for your configuration. You can then edit your config file with the expected configurable fields:

    ```yaml
    # yaml-language-server: $schema=schemas/schema.json
    ...
    system:
        # Your editor knows the configuration values you can specify based on this string:
        class_name: src.sim.my_sim:MySim
        # Your editor will tell you this field exists, and give it the description "This value changes the setup of the simulation.", which is taken from the docstring above.
        configurable_value: 1
    ```
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        use_attribute_docstrings=True, extra="forbid"
    )

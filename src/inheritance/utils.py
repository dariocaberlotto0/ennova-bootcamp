def some_tool(x):
    print(x)

class SomeMixin:
    def __init__(self, *args, some_config, **kwargs):
        print("SomeMixin init...")
        # create a dictionary
        self._some_config = {"internal": True, **some_config}
        super().__init__(*args, **kwargs)

    # Getter
    @property
    def some_config(self):
        config = self._some_config.copy()
        # Delete something I don't want to show to user
        del config["internal"]
        return config
    
    @some_config.setter
    def some_config(self, config):
        self._some_config = {"internal": True, **config}


class SomeClass:
    def __init__(self, some_data):
        print("SomeClass init...")
        self.some_data = some_data


class Enriched(SomeMixin, SomeClass):
    def __init__(self, some_data, *, some_config):
        print("Enriched init...")
        super().__init__(some_data, some_config=some_config)


s = SomeMixin(some_config={"llm": "gemini"})
# Modify directly the object
s._some_config["internal"] = False

# Calls the getter
s.some_config
# Without the setter, it modify the copy, the object remains the same
s.some_config["llm"] = "claude"

e = Enriched({"a": "b"}, some_config=s.some_config)
print(e.some_data, e.some_config)
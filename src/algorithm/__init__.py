class Algorithm:
    class_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "variants"):
            variants = cls.variants
        else:
            variants = {}
        if hasattr(cls, "shortname"):
            shortname = cls.shortname
        else:
            shortname = cls.__name__.lower()
        variants[shortname] = {}
        for name, kwargs in variants.items():
            Algorithm.class_dict[name] = (cls, kwargs)

    def init(classname, kwargs):
        if classname not in Algorithm.class_dict:
            raise ValueError(f"Algorithm not found: '{classname}'")
        class_obj, default_args = Algorithm.class_dict[classname]
        return class_obj(**default_args, **kwargs)

class FirstOrderAlgorithm(Algorithm):
    pass

class StabOracleAlgorithm(Algorithm):
    pass

from . import sgd
from . import adam
from . import stab_sgd
from . import schedule_free

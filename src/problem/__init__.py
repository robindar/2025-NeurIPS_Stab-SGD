import os
import sys

class Problem:
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
        for var, kwargs in variants.items():
            Problem.class_dict[var] = (cls, kwargs)

    def init(classname, kwargs):
        if classname not in Problem.class_dict:
            raise ValueError(f"Problem not found: '{classname}'")
        class_obj, default_args = Problem.class_dict[classname]
        return class_obj(**default_args, **kwargs)

from . import quadratic

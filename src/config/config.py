import os
import glob
import json
import time
import tqdm
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Mapping, Any

from ..algorithm import Algorithm
from ..problem import Problem


default_dict = field(default_factory=lambda: {})

@dataclass
class Headers:
    problem : str
    algorithm : str
    problem_args : Mapping[str, Any] = field(default_factory=lambda: {})
    algorithm_args : Mapping[str, Any] = field(default_factory=lambda: {})
    n_iter : float = 1

class Config:
    def __init__(self,
            data=None,
            prefix=None,
            **kwargs,
            ):
        self.data = data
        savefile = kwargs.pop("savefile", None)
        self.headers = Headers(**kwargs)

        pb, alg = self.headers.problem, self.headers.algorithm
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        rr = int((1000 * time.time()) % 1000) # just to randomize multiple writes per second
        self.savefile_slug = f"{pb}_{alg}_{timestamp}{rr:03d}"
        default_savefile = f"measurements/{prefix or '.'}/{self.savefile_slug}.json"
        self.savefile = savefile or default_savefile

    def init_problem(self):
        hd = self.headers
        return Problem.init(classname=hd.problem, kwargs=hd.problem_args)

    def init_algorithm(self):
        hd = self.headers
        return Algorithm.init(classname=hd.algorithm, kwargs=hd.algorithm_args)

    def register_recorded_data(self, data):
        self.data = data

    def get_recorded_data(self):
        return self.data

    @classmethod
    def all_recorded(cls, prefix="**", progress=True, **kwargs):
        if isinstance(prefix, list):
            for p in prefix:
                for c in cls.all_recorded(prefix=p, progress=progress, **kwargs):
                    yield c
            return None
        iterator = glob.glob("measurements/" + prefix + "/*.json")
        if progress:
            iterator = tqdm.tqdm(iterator, leave=False, desc="Scanning measurements")
        for filename in iterator:
            config = cls.load_from_file(filename)
            if config.headers_match(**kwargs):
                yield config


    def string_match(self, req, found):
        return (req is None) or (req == found)

    def dict_match(self, req, found):
        for key in req:
            if key not in found:
                return False
            val = req[key]
            if isinstance(val, (float, int)):
                if not np.isclose(val, float(found[key]), rtol=1e-8, atol=1e-12):
                    return False
            else:
                if val != req:
                    return False
        return True

    def headers_match(self, problem=None, algorithm=None, problem_args={}, algorithm_args={}):
        if not self.string_match(problem, self.headers.problem):
            return False
        if not self.string_match(algorithm, self.headers.algorithm):
            return False
        if not self.dict_match(problem_args, self.headers.problem_args):
            return False
        if not self.dict_match(algorithm_args, self.headers.algorithm_args):
            return False
        return True

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            try:
                dic = json.load(f)
            except Exception as e:
                print(f"ERROR: could not decode json from '{filename}'", flush=True)
                raise e
        data = dic.pop("data", None)
        headers = dic.pop("headers", {})
        if data is not None:
            data = pd.DataFrame.from_dict(data)
        config = cls(**headers, data=data, savefile=filename)
        return config

    def write_to_file(self, filename, quiet=False):
        serialized = { "headers": asdict(self.headers) }
        if self.data is not None:
            serialized["data"] = self.data

        directory = os.path.dirname(filename)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        with open(filename, 'w') as f:
            json.dump(serialized, f)
            f.write("\n")

        if not quiet:
            print(f"Output written to {filename}")


class Recorder:
    def __init__(self):
        self.recorded_data = {}

    def record(self, kwargs):
        for key, val in kwargs.items():
            if key not in self.recorded_data:
                # WARNING: does not match len in record if uneven calls
                self.recorded_data[key] = []
            self.recorded_data[key].append(val)

    def dump(self):
        return self.recorded_data

    def to_dataframe(self):
        return pd.DataFrame(self.recorded_data)

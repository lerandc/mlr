from collections.abc import Mapping
from dataclasses import make_dataclass
from inspect import Parameter, get_annotations, isclass, signature


def required_args(sig, filter_self=True):
    pfilter = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)

    def check_param(p):
        for t in pfilter:
            if p.kind == t:
                return False

        if p.default is p.empty:
            return True
        else:
            return False

    args = {k: v for k, v in sig.parameters.items() if check_param(v)}
    if filter_self and "self" in args:
        del args["self"]

    return args.keys()


class Config(Mapping):
    def __init__(self, mclass, *args, **kwargs):
        self.metadata_class = mclass
        self.metadata = self.metadata_class(*args, **kwargs)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, mdata):
        ## Test for interface compatible with dataclass
        try:
            kv = {k: getattr(mdata, k) for k in get_annotations(type(mdata))}
            if len(kv) == 0:
                raise TypeError("Input object has no annotations")
        except TypeError as e:
            print(str(e))

        self._metadata = mdata

    def __len__(self):
        return len(get_annotations(self.metadata_class).keys())

    def __getitem__(self, key):
        return getattr(self.metadata, key)

    def __iter__(self):
        return iter(get_annotations(self.metadata_class).keys())

    def __repr__(self):
        return self.metadata.__repr__()


class FConfig:
    def __init__(self, conf_class, **kwargs):
        # Store the class associated with the config
        if isclass(conf_class):
            self.conf_class = conf_class
        else:
            self.conf_class = type(conf_class)

        init_sig = signature(getattr(self.conf_class, "__init__"))

        self.metadata_class = make_dataclass(
            self.conf_class.__name__ + "_Config",
            fields=list(required_args(init_sig)) + [(k, v) for k, v in kwargs.items()],
        )
        match self._validate_config():
            case True:
                pass
            case False, missing_args:
                raise ValueError(
                    f"Specified configuration {kwargs}"
                    f" is missing required arguments {missing_args}"
                    f" of __init__ for {self.metadata_class.__name__}"
                )

    def _validate_config(self):
        init_sig = signature(getattr(self.conf_class, "__init__"))
        req_args = set(required_args(init_sig))
        test_args = set(get_annotations(self.metadata_class).keys())
        missing_args = req_args.difference(test_args)
        if len(missing_args) > 0:
            return False, missing_args
        else:
            return True

    def __call__(self, *args, **kwargs):
        return Config(self.metadata_class, *args, **kwargs)

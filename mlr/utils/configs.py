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

    args = {k: v.annotation for k, v in sig.parameters.items() if check_param(v)}
    if filter_self and "self" in args:
        del args["self"]

    return args


class Config(Mapping):
    def __init__(self, mclass, *args, **kwargs):
        self.metadata_class = mclass
        try:
            self.metadata = self.metadata_class(*args, **kwargs)
        except TypeError:
            raise TypeError(f"{self.metadata_class.__name__} requires the following arguments: {self.metadata_class.__annotations__}")

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, mdata):
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
    def __init__(self, conf_class, **added_args):
        # Store the class associated with the config
        if isclass(conf_class):
            self.conf_class = conf_class
        else:
            self.conf_class = type(conf_class)

        self.init_sig = signature(getattr(self.conf_class, "__init__"))
        self.def_args = required_args(self.init_sig)
        self.added_args = added_args
        self.req_args =  {**self.def_args, **self.added_args}

        self.metadata_class = make_dataclass(
            self.conf_class.__name__ + "_Config",
            fields=[(k, v) for k, v in self.req_args.items()],
        )

        match self._validate_config():
            case True:
                pass
            case False, missing_args:
                raise ValueError(
                    f"Specified configuration {added_args}"
                    f" is missing required arguments {missing_args}"
                    f" of __init__ for {self.metadata_class.__name__}"
                )

    def _validate_config(self):
        test_args = set(get_annotations(self.metadata_class).keys())
        missing_args = set(self.req_args.keys()).difference(test_args)
        if len(missing_args) > 0:
            return False, missing_args
        else:
            return True

    def __call__(self, *args, **kwargs):
        return Config(self.metadata_class, *args, **kwargs)
    
    def __repr__(self):
        return f"FConfig(conf_class={self.conf_class.__name__}, {self.added_args})"
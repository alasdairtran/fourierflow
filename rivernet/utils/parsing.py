import yaml
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params, parse_overrides, with_fallback


def yaml_to_params(params_file: str, overrides: str = "") -> Params:
    # redirect to cache, if necessary
    params_file = cached_path(params_file)

    with open(params_file) as f:
        file_dict = yaml.safe_load(f)

    overrides_dict = parse_overrides(overrides)
    param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

    return Params(param_dict)

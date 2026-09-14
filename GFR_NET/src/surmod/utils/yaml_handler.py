from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge source into a deep copy of target."""
    result = deepcopy(target)
    for key, value in source.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    experiment_name: str,
    config_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and resolve an experiment configuration from YAML templates."""
    if config_file is None:
        config_file = "configs.yaml"

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path}")

    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    templates: Dict[str, Dict[str, Any]] = raw.get("templates", {})
    experiments: Dict[str, Dict[str, Any]] = raw.get("experiments", {})

    if experiment_name not in experiments:
        raise KeyError(f"Experiment '{experiment_name}' not found.")

    experiment = experiments[experiment_name]
    inherit_list: List[str] = experiment.get("inherit", [])

    config: Dict[str, Any] = {}
    for block_name in inherit_list:
        if block_name not in templates:
            raise KeyError(f"Template block '{block_name}' not found.")
        config = deep_merge(config, templates[block_name])

    overrides = {key: value for key, value in experiment.items() if key != "inherit"}
    return deep_merge(config, overrides)
"""Generate pyproject.toml for different build variants of next-plaid-python."""

import argparse
import sys

CONFIGS = {
    "default": {
        "package_name_suffix": "",
        "features": ["pyo3/extension-module"],
    },
    "cuda": {
        "package_name_suffix": "-cuda",
        "features": ["pyo3/extension-module", "cuda-13"],
    },
    "openblas": {
        "package_name_suffix": "-openblas",
        "features": ["pyo3/extension-module", "openblas"],
    },
    "accelerate": {
        "package_name_suffix": "-accelerate",
        "features": ["pyo3/extension-module", "accelerate"],
    },
    "mkl": {
        "package_name_suffix": "-mkl",
        "features": ["pyo3/extension-module", "mkl"],
    },
}


def generate_config(variant_name: str, base_name="next-plaid"):
    """Generate a pyproject.toml file for a specific build variant."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    config = CONFIGS.get(variant_name)
    if not config:
        error = f"Unknown variant '{variant_name}'. Available: {list(CONFIGS.keys())}"
        raise ValueError(error)

    full_package_name = f"{base_name}{config['package_name_suffix']}"
    template_path = "pyproject.toml.template"

    with open(template_path) as f:
        template_content = f.read()

    features_list = config["features"]
    formatted_features = (
        "[" + ", ".join(f'"{feature}"' for feature in features_list) + "]"
    )

    new_content = template_content.replace(
        "{{ package_name }}", f'"{full_package_name}"'
    )
    new_content = new_content.replace("{{ features }}", formatted_features)

    with open("pyproject.toml", "w") as f:
        f.write(new_content)

    print(f"Successfully generated pyproject.toml for '{full_package_name}'")
    print(f"   Features: {features_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pyproject.toml for a specific build variant."
    )
    parser.add_argument(
        "variant",
        choices=CONFIGS.keys(),
        help="The build variant to generate.",
    )
    args = parser.parse_args()
    generate_config(variant_name=args.variant)

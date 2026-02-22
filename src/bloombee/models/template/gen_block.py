##
# tools/gen_model.py
"""
Generate model files (block.py, config.py, model.py) from templates and spec YAML.

Usage:
    python tools/gen_model.py --spec model_specs/llama.yaml --templates templates --out-root src/bloombee/models
    python /home/twei11/new/BloomBee/src/bloombee/models/template/gen_block.py --spec /home/twei11/new/BloomBee/src/bloombee/models/template/llama.yaml --templates /home/twei11/new/BloomBee/src/bloombee/models/template --out-root /home/twei11/new/BloomBee/src/bloombee/models/template/test
    # Generate only specific files
    python tools/gen_model.py --spec model_specs/llama.yaml --only block
    python tools/gen_model.py --spec model_specs/llama.yaml --only config
    python tools/gen_model.py --spec model_specs/llama.yaml --only model
"""
import argparse
import yaml
import pathlib
from jinja2 import Environment, FileSystemLoader

# Required fields for each template
REQUIRED_FIELDS = {
    "block": [
        "model_name",
        "class_prefix",
        "hf_model_key",
        "config_class",
        "flex_module_import",
        "attention_class",
        "mlp_class",
        "norm_class",
        "decoder_base_class",
        "weight_download_module",
        "weight_download_fn",
    ],
    "config": [
        "model_name",
        "class_prefix",
        "hf_model_key",
        "config_class",
        "hf_attention_class",
    ],
    "model": [
        "model_name",
        "class_prefix",
        "hf_model_key",
        "hf_model_class",
        "hf_pretrained_class",
        "hf_causal_lm_class",
        "hf_seq_cls_class",
        "layers_attr",
        "embed_tokens_attr",
        "final_norm_attr",
    ],
}

# Template filename -> output filename
TEMPLATES = {
    "block": ("block.py.j2", "block.py"),
    "config": ("config.py.j2", "config.py"),
    "model": ("model.py.j2", "model.py"),
}

# Default values for optional fields
DEFAULTS = {
    # block.py defaults
    "default_model_name": "llama-7b",
    "hf_tokenizer_prefix": "huggyllama",
    "uses_rotary": False,
    "prepare_mask_with_hf": False,
    "needs_bloom_to_llama_reorder": False,
    
    # config.py defaults
    "block_prefix": "model.layers",
    "has_num_key_value_groups": False,
    "dht_suffix": "-hf",
    "license_notice": None,
    "config_overrides": {},
    "config_override_comments": {},
    
    # model.py defaults
    "layers_attr": "layers",
    "embed_tokens_attr": "embed_tokens",
    "final_norm_attr": "norm",
    "has_pretraining_tp": False,
}


def validate_spec(spec: dict, template_type: str) -> None:
    """Validate that all required fields are present."""
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [f for f in required if f not in spec]
    if missing:
        raise ValueError(f"Missing required fields for {template_type}: {missing}")


def load_snippet(spec: dict, key: str = "shared_impl_snippet") -> None:
    """Load external snippet file if specified."""
    snippet_path = spec.get(key)
    if snippet_path:
        snippet_file = pathlib.Path(snippet_path)
        if not snippet_file.exists():
            raise FileNotFoundError(f"Snippet file not found: {snippet_path}")
        spec[key] = snippet_file.read_text(encoding="utf-8")
    else:
        spec[key] = None


def apply_defaults(spec: dict) -> None:
    """Apply default values for optional fields."""
    for key, default in DEFAULTS.items():
        spec.setdefault(key, default)


def generate_file(
    spec: dict,
    template_type: str,
    env: Environment,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Generate a single file from template."""
    template_name, output_name = TEMPLATES[template_type]
    
    validate_spec(spec, template_type)
    
    tpl = env.get_template(template_name)
    rendered = tpl.render(**spec)
    
    out_file = out_dir / output_name
    out_file.write_text(rendered, encoding="utf-8")
    return out_file


def main():
    ap = argparse.ArgumentParser(description="Generate model files from templates")
    ap.add_argument("--spec", required=True, help="Path to model spec YAML file")
    ap.add_argument("--templates", default="templates", help="Path to templates directory")
    ap.add_argument("--out-root", default=".", help="Output root directory")
    ap.add_argument(
        "--only",
        choices=list(TEMPLATES.keys()),
        action="append",
        help="Generate only specific file type(s). Can be specified multiple times.",
    )
    args = ap.parse_args()

    # Load spec
    spec_path = pathlib.Path(args.spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    
    # Load snippets
    load_snippet(spec, "shared_impl_snippet")
    
    # Apply defaults
    apply_defaults(spec)
    
    # Setup Jinja environment
    env = Environment(
        loader=FileSystemLoader(args.templates),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    # Determine output directory
    out_dir = pathlib.Path(args.out_root) / spec["model_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate files
    templates_to_generate = args.only if args.only else list(TEMPLATES.keys())
    
    for template_type in templates_to_generate:
        try:
            out_file = generate_file(spec, template_type, env, out_dir)
            print(f"Generated {out_file}")
        except Exception as e:
            print(f"Error generating {template_type}: {e}")
            raise


if __name__ == "__main__":
    main()
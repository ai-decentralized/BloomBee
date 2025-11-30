# Model Code Generation Templates

This directory contains Jinja2 templates and tooling for automatically generating model-specific code files (`block.py`, `config.py`, `model.py`) from a single YAML specification.

## Overview

Instead of manually writing boilerplate code for each new model architecture, you can define a YAML(e.g. llama.yaml) spec file with model-specific parameters, and the generator produces part of necessary files with consistent structure. However, the real computation function(src/bloombee/models/llama/flex_llama.py) and other stuffs still need hand writting.

## Directory Structure

```
templates/
├── block_py.j2              # Template for FlexGen decoder block
├── config_py.j2             # Template for distributed config class
├── model_py.j2              # Template for distributed model classes
└── decoder_shared_impl.pyfrag   # Shared forward() implementation
└── llama.yaml               # Model specification file
└── gen_model.py             # Generation script
```

## Usage

### Generate All Files
Use llama model as example:
```bash
python gen_model.py 
--spec $PATH_TO_SPEC$/llama.yaml \
--templates $PATH_TO_TEMPLATE_FOLDER$ \
--out-root src/bloombee/models
```

This generates:
- `src/bloombee/models/llama/block.py`
- `src/bloombee/models/llama/config.py`
- `src/bloombee/models/llama/model.py`

### Generate Specific Files

```bash
# Generate only block.py
python tools/gen_model.py --spec $PATH_TO_SPEC$/llama.yaml --only block

# Generate only config.py and model.py
python tools/gen_model.py --spec $PATH_TO_SPEC$/llama.yaml --only config --only model
```

## YAML Specification Reference

A complete specification file contains parameters for all three templates:

```yaml
# ============================================================
# Basic Identifiers (used by all templates)
# ============================================================
model_name: llama                  # Output directory name
class_prefix: Llama                # Class name prefix (e.g., DistributedLlamaModel)
hf_model_key: llama                # HuggingFace model key
config_class: LlamaConfig          # HuggingFace config class

# ============================================================
# block.py Parameters
# ============================================================
flex_module_import: bloombee.models.llama.flex_llama
attention_class: FLEX_LlamaAttention
mlp_class: FLEX_LlamaMLP
norm_class: FLEX_LlamaRMSNorm
decoder_base_class: LlamaDecoderLayer

weight_download_module: bloombee.flexgen_utils.llama_config
weight_download_fn: download_llama_weights

# Boolean flags
needs_bloom_to_llama_reorder: true   # Enable KV cache format conversion
uses_rotary: true                     # Enable rotary position embeddings
prepare_mask_with_hf: true            # Use HF's 4D causal mask

# Tokenizer settings
default_model_name: llama-7b
hf_tokenizer_prefix: huggyllama

# Optional: custom forward() implementation
shared_impl_snippet: path/to/decoder_shared_impl.pyfrag

# ============================================================
# config.py Parameters
# ============================================================
hf_attention_class: LlamaAttention
block_prefix: model.layers
has_num_key_value_groups: true       # Enable GQA support
license_notice: "Make sure you follow the Llama terms of use..."
dht_suffix: "-hf"

# Config overrides applied in from_pretrained()
config_overrides:
  pretraining_tp: 1
  use_cache: true

# ============================================================
# model.py Parameters
# ============================================================
hf_model_class: LlamaModel
hf_pretrained_class: LlamaPreTrainedModel
hf_causal_lm_class: LlamaForCausalLM
hf_seq_cls_class: LlamaForSequenceClassification

# Model architecture attributes (vary between architectures)
layers_attr: layers              # self.layers (LLaMA) vs self.h (GPT-2)
embed_tokens_attr: embed_tokens  # self.embed_tokens (LLaMA) vs self.wte (GPT-2)
final_norm_attr: norm            # self.norm (LLaMA) vs self.ln_f (GPT-2)

has_pretraining_tp: true         # Whether model has pretraining_tp config
```

## Architecture Attribute Reference

Different model architectures use different attribute names:

| Model | `layers_attr` | `embed_tokens_attr` | `final_norm_attr` |
|-------|---------------|---------------------|-------------------|
| LLaMA | `layers` | `embed_tokens` | `norm` |
| GPT-2 | `h` | `wte` | `ln_f` |
| BLOOM | `h` | `word_embeddings` | `ln_f` |
| Falcon | `h` | `word_embeddings` | `ln_f` |
| Mistral | `layers` | `embed_tokens` | `norm` |

## Template Customization

### Boolean Flags

| Flag | Effect when `true` |
|------|-------------------|
| `uses_rotary` | Imports and enables `apply_rotary_pos_emb` |
| `prepare_mask_with_hf` | Uses HuggingFace's `_prepare_4d_causal_attention_mask` |
| `needs_bloom_to_llama_reorder` | Adds KV cache format conversion methods |
| `has_num_key_value_groups` | Adds `num_key_value_groups` property for GQA |
| `has_pretraining_tp` | Includes `pretraining_tp` in model init |

### Custom Implementation Snippets

The `shared_impl_snippet` parameter allows injecting custom `forward()` implementations into the decoder layer. The snippet should define a `forward` method with the appropriate signature. This is useful when the default template logic doesn't match the model's requirements.

## Troubleshooting

### Missing Required Fields

```
ValueError: Missing required fields for block: ['attention_class', 'mlp_class']
```

Ensure all required fields are defined in your YAML spec.

### Template Not Found

```
jinja2.exceptions.TemplateNotFound: block_py.j2
```

Check that `--templates` points to the correct directory containing `.j2` files.

### Inference Results Differ

If generated code produces different results than the original:

1. Ensure `shared_impl_snippet` points to the correct implementation
2. Verify all boolean flags match the original code's behavior
3. Check that KV cache format conversion is correctly configured
from bloombee.models.bloom import *
from bloombee.models.falcon import *
from bloombee.models.llama import *
from bloombee.models.mixtral import *
from bloombee.models.deepseekv3 import *

try:
    from bloombee.models.gpt_oss import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.gpt_oss"):
        raise

try:
    from bloombee.models.gemma4 import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.gemma4"):
        raise

try:
    from bloombee.models.qwen3 import *
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("transformers.models.qwen3"):
        raise

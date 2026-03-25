from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./checkpoints/lmhead/lm_head_llama13B-20.pt",
    path_in_repo="lm_head_llama13B-20.pt",
    repo_id="xxiong59/lm-head-for-speculative-pruning",
    repo_type="model"
)
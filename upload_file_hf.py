from huggingface_hub import login, upload_file

login(token="")

upload_file(
    path_or_fileobj="./checkpoints/lmhead/lm_head_llama30B-15.pt",
    path_in_repo="lm_head_llama30B-15.pt",
    repo_id="xxiong59/lm-head-for-speculative-pruning",
    repo_type="model"
)
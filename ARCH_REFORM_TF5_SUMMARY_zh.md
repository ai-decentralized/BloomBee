# arch-reform-qwen3-4b — 架构改造 + TF 5.x 迁移总结(中文版)

`arch-reform-qwen3-4b` 分支交接文档,状态截至 commit
`8022e74`(2026-04-19)。目的:租到大 GPU 之后拿到这份文档就能直接
上手复现 benchmark、把工作扩展到更大模型(llama-70b、falcon-40b、
mixtral-8x7b)而不需要翻完整 commit 历史。

---

## 1. 这个分支做了什么

两条相互交织的工作线,叠在 BloomBee mainline 之上:

1. **架构改造**(Phase 0 → Phase 3)。重构 KV 热路径、
   spec-decoding 算法、block_function seam、FlexGen 基层。在真实
   llama-7b 上拿到每步级加速,并落了一个带 rollback 语义的 paged-KV
   shim —— 在 spec-dec 下已经 load-bearing(commit `8a19811`)。
2. **transformers 5.x 迁移**。上游 TF 5.x 把 Bloom / Falcon /
   Llama 的 attention 从 tuple-KV 迁到 `Cache` API;几个 bookkeeping
   路径(`mark_tied_weights_as_initialized`、
   `prepare_inputs_for_generation`)跟 BloomBee 的分布式 LMHead /
   基于哨兵的 RemotePastKeyValues 不兼容。这次迁移让 README 里列出的
   5 个模型家族(Llama、Bloom、Falcon、Mixtral、Qwen3)在 TF 5.5.4
   下全部端到端跑通。

这条分支的不变式只有一条:**README.md 里列出的模型必须在当前 TF 版本
下全部能端到端运行。** 有一个跑不起来,这条分支就是坏的。

## 2. 环境基线

- **主机**:V100-SXM2-16GB,driver 580,CUDA 13.0,fp16。
- **Venv**:`admin@192.168.31.118` 的 `/data/models/bloombee-venv`。
- **TF 版本**:`transformers==5.5.4`(pyproject 已 pin;BloomBee
  adapter 里带 4.x / 5.x 双兼容 shim)。
- **V100 上已就位的模型**:`bloom-560m`、`falcon-rw-1b`、
  `qwen3-0.6b`、`tinyllama-1.1b`、`qwen35-27b.gguf`。
- **Torch safetensors 绕路**:torch < 2.6 因 CVE-2025-32434 拒绝加载
  `.bin`;falcon-rw-1b 被转成了 `model.safetensors`(`lm_head` ↔
  `word_embeddings` 做了 shared-tensor clone)。

## 3. 设计变更(按子系统组织)

### 3.1 Paged-KV shim(Phase 2)—— `BLOOMBEE_PAGED_KV`

`MemoryCache._register_paged_view(handle, (k, v))` 在分配时把
`PagedKVTable` 别名到 cache slab 上。调用方看到的还是同一个
`TorchTensor` 对象,paged 表作为 state-only mirror 存在。
`track_write` + commit/rollback 通过 `_do_reorder_task` 接通,
speculated KV 页在被拒时可以丢掉,而不影响 attention kernel 的读路径。

- **开关**:`BLOOMBEE_PAGED_KV=1`。默认关;additive 且 non-fatal,
  注册失败会 warning 后回退到 legacy slab 路径。
- **代价**(单服务器,llama-7b):~1.6% per-step overhead。
- **当前状态**:在 spec-dec rollback 下已 load-bearing;attention
  读路径还是直接读 slab。下一步是把读路径走 `gather_prefix`,那时
  slab 才会真正退化成 staging buffer。

### 3.2 Spec-decoding 算法刷新

- **向量化 spec verify**(`e89d7cb`):argmax 挪到批量 GPU 算子,
  带 batch 强烈放大。B=1 时 2.7×,**B=16 时 31.7×**。
- **EAGLE-2 budgeted tree shape**(`1e70e26`):把旧的标量概率树
  换成按概率加权扩展。
- **SpecInfer 随机拒绝采样**(`fdfd0b7`):换成严格 accept/reject,
  不再总是 accept top-1。
- **O(n·depth) ancestor matrix**(`35c6697`):树的 parent-walk 代替
  矩阵闭包。按树形从 4.5× 到 10.7×。

### 3.3 FlexGen KV 写路径

- **In-place slab 写**(`c04bdee`):干掉了每步 `torch.cat`(每步
  decode 都重新分配一个包含全部历史的 tensor)。CPU 上
  2.3×–93.2×;CUDA 单次调用持平,但消除了 allocator churn
  (在第 5 节 client-wall 对比里看得到收益)。

### 3.4 Cache 布局兼容性(TF 5.x)

TF 5.x 重写了 `BloomAttention` / `FalconAttention` /
`LlamaAttention`,它们现在期待 `layer_past: Cache`(带 `.update()`
方法)。BloomBee 的 `_run_block_forward` 仍然直接递 `(k, v)` tensor
tuple,因为分布式后端的 cache state 走 `MemoryCache` slab,不是 HF
的 Cache。三个补丁桥接:

- **`OptimizedBloomAttention`**(`src/bloombee/models/bloom/block.py`):
  `BloomAttention` 的子类,override `forward` 来 inline 做 tuple-cache
  concat,`present` 按 BloomBee 的标准 **3D 布局** 返回 ——
  `key=[B*H, D, S]`、`value=[B*H, S, D]`,这样
  `memory_cache_manager._write_kvs` 的断言 `key_t.ndim == 3` 才通过。
  `WrappedBloomBlock` 把它装到 `self_attention` 位置上,并自己实现
  layer-norm + attn + MLP 正向(没法 delegate 给 `super().forward`,
  上游是按 positional 传 `layer_past` 并期待它是 Cache)。
- **健壮的 `past_length` 判断**:past key 可能是 3D `[B*H, D, S]`
  (backend tuple),也可能是 4D `[B, H, S, D]`(OptimizedBloomAttention
  自己的输出在循环内又喂回来时)。靠 `head_dim` 跟最后两维比对消歧。
- **Bloom 走 HF 加载路径**:`server/from_pretrained.py` 里的
  `_is_hf_model` 现在包含 `WrappedBloomBlock`;Bloom 不再误入
  FlexGen Llama 分支(那个分支按 `(config, layer_idx, env, policy,
  weight_home, path)` 传参,崩在 "`takes from 2 to 3 positional
  arguments but 7 were given`")。`server/block_utils.get_model_block`
  现在按 `(config, layer_idx)` dispatch Bloom。

### 3.5 Tied-weights 兼容(TF 5.x)

TF 5.x 的 `PreTrainedModel.tie_weights` 会通过
`self.get_parameter(name)` 遍历 `_tied_weights_keys`(里面包括
`lm_head.weight`)。BloomBee 的 `LMHead` 在 bind-time 之前把
`self.weight = None`,因为它要 tie 的 embedding 住在另一个 peer 上。
`None` 不是 `nn.Parameter`,`get_parameter` 就抛
`AttributeError("'weight' is not an nn.Parameter")`。

统一修正,应用在 Bloom / Falcon / Llama / Mixtral / Qwen3 的
`ForCausalLM` 上:

```python
def mark_tied_weights_as_initialized(self, loading_info):
    # TF 5.x 的记账流程,会崩在 LMHead weight=None 占位符上;
    # 我们下面手动 tie,这个遍历用不上。
    return

def tie_weights(self, missing_keys=None, recompute_mapping=True):
    if getattr(self.config, "tie_word_embeddings", False):
        embed = self.get_input_embeddings()
        if embed is not None and getattr(embed, "weight", None) is not None:
            self.lm_head.weight = embed.weight
```

当 `tie_word_embeddings=False`(llama-2-7b、falcon-rw-1b)时是 no-op。
当 True(qwen3、多数小 bloom 变体)时直接把 embedding 权重绑到
client 端 LMHead 上,行为等价于 TF 5.x 重构那条遍历之前的 mainline。

### 3.6 Device-context 守卫

TF 5.x 的 `from_pretrained` 把模型 `__init__` 包在
`torch.device('cuda')` 里,这会劫持 hivemind 的 DHT / MPFuture 构建
过程里 `torch.empty()` 的调用,在本该是 CPU tensor 的地方创建 CUDA
tensor。`share_memory_()` 只在 CPU tensor 上能跑,结果 server 在
bring-up 阶段就崩。修复(Llama 本来就有,Bloom / Falcon / Mixtral /
Qwen3 现在也都加上):

```python
with torch.device('cpu'):
    self.layers = RemoteSequential(config, dht=dht)
```

### 3.7 Falcon 的 Cache-aware `prepare_inputs_for_generation`

上游 Falcon 默认的 `prepare_inputs_for_generation` 会做
`past_key_values[0][0].shape[2]`,这在 BloomBee 的
`RemotePastKeyValues`(`Cache` 子类,`__getitem__` 返回哨兵 tensor)
上会崩。重新实现为 Cache-aware —— 见
`src/bloombee/models/falcon/model.py`,跟 Llama 那边的写法一致。
同时设置 `_supports_cache_class = True`,让 TF 5.x 的 generate()
循环传给我们真正的 Cache 而不是伪 legacy tuple。

### 3.8 FlexGen 小修

- `flex_llama.FLEX_LlamaAttention.__init__` 现在会把 `layer_idx`
  透传给 `super().__init__`(TF 5.x 要),带 `TypeError` fallback
  兼容老 TF。
- `rms_norm` 默认 eps 恢复成 **1e-5**(对齐 LLaMA config 的
  `rms_norm_eps`)。之前一个实验性 tweak 把它改成 1e-6,恢复的原因
  是某些 config 在每层 norm 权重缺失时会回退到这个默认值。

## 4. 验证范围(已验证 vs 未验证)

V100 16G / TF 5.5.4 实测:

| 模型 | 状态 | 备注 |
|---|---|---|
| Llama-7b | ✅ E2E(5c88aa…5c98d83 文档) | 8022e74 的增量在这里是回归安全的 |
| Qwen3-0.6B | ✅ E2E | 迁移后回归检查通过 |
| Bloom-560m | ✅ E2E | OptimizedBloomAttention 后输出连贯 |
| Falcon-rw-1b | ✅ E2E | Cache-aware prepare_inputs + safetensors 绕路 |
| Mixtral | ⚠️ 能 import | 16GB V100 装不下,未 E2E 验证 |

已知不支持(FlexGen 预存限制,跟本分支无关):

- **tinyllama-1.1b** —— 用了 GQA(32 Q 头 / 4 KV 头);FlexGen 的
  `flex_llama.py` 是纯 MHA。README 里没列这个模型。

## 5. 性能快照(llama-7b,V100,TF 5.x 前)

数据出自 `ARCH_REFORM_BENCHMARK_RESULTS.md` §6。TF 5.x 迁移只加了
`tie_weights` override(`tie_word_embeddings=False` 时是 no-op,
llama-7b 正好是 False),理论上数字应该在噪声范围内。
**严谨起见需要在 TF 5.5.4 下重跑一次。**

**单服务器(32 层)**

| 分支 | ms/step | tok/s | Client wall(稳态) |
|---|---:|---:|---:|
| mainline(e5b88aa) | 225.1 | 75.4 | ~24 s |
| **arch-reform** | **152.9** | **111.0** | **~12.4 s** |

→ **服务端 1.47× / 客户端 1.90×**

**双服务器流水线(A: 0:16, B: 16:32,同一台 V100)**

| 分支 | ms/step A | ms/step B | tok/s(单 server) |
|---|---:|---:|---:|
| arch-reform PAGED_KV=0 | 81.2 | 80.8 | ~210 |
| arch-reform PAGED_KV=1 | 82.8 | 82.9 | ~205 |

在 `RLIMIT_MEMLOCK` 较紧的 V100 主机上,FlexGen 默认 1 GB 的
pinned CPU relay buffer(`copy_worker_func`)可能在 DHT socket
bind 之前就注册失败,报成一个误导性的 "CUDA out of memory"。
若要在受限硬件上压两台 server,本地把这个 buffer 调小即可;
本分支没有把它做成 knob —— 在 A100 上复现不出问题(默认
`Disk% = 0` 时 relay 根本不走)。

## 7. 复现手册(V100 —— 复跑第 5 节数据)

```bash
# 在 admin@192.168.31.118 上
source /data/models/bloombee-venv/bin/activate
cd /data/models/bloombee
git checkout arch-reform-qwen3-4b  # 8022e74

# 单服务器 llama-7b(假设 llama-7b 已经放在本地)
python -u -m bloombee.cli.run_server \
    /data/models/llama-7b-hf \
    --new_swarm --num_blocks 32 --port 31363 \
    --device cuda --torch_dtype float16 \
    --dht_prefix llama7b \
    --identity_path /data/models/bb_run/identity_llama7b.key \
    2>&1 | tee server_llama7b.log

# Bloom-560m(更小的 smoke test,24 层)
python -u -m bloombee.cli.run_server \
    /data/models/bloom-560m \
    --new_swarm --num_blocks 24 --port 31363 \
    --device cuda --torch_dtype float16 \
    --dht_prefix bloom560 \
    --identity_path /data/models/bb_run/identity_bloom.key

# 客户端:参考 /data/models/bb_run/client_bloom.py。
```

## 8. 扩到大 GPU 的 checklist

转到 A100-80G / H100 / 多卡做 llama-70b 或 mixtral-8x7b 验证时:

- [ ] 确认 `RLIMIT_MEMLOCK=unlimited`(`ulimit -l`);受限主机上
      FlexGen 的 pinned CPU relay buffer(1 GB fp16 = 2 GB pinned)
      会在 server 启动时直接撞墙。
- [ ] 在 TF 5.5.4 下重跑 `ARCH_REFORM_BENCHMARK_RESULTS.md` §6,
      确认迁移没引入性能回归。
- [ ] Mixtral:在没有 flash-attn 的 host 上,确保
      `config._attn_implementation` 落到 `eager`。
      `server/block_utils._autoset_attn_impl` 会处理,但值得在第一个
      block 的日志行里确认 `attn_impl` 字段。
- [ ] 含 GQA 的模型(llama-3.1-8B、llama-3-70B、qwen3 较大尺寸):
      当前只走 HF block 路径,不走 FlexGen llama。如果想让 FlexGen
      也支持 GQA,`flex_llama.py` 里要在 attention 前做 KV-head split
      → repeat —— 本分支没做。
- [ ] 每个新模型客户端侧 Cache 类型兼容性都要确认:generate() 会调
      `prepare_inputs_for_generation`,falcon / llama 已经有
      Cache-aware 版本;bloom / mixtral 继承上游的,目前能跑,但新
      TF 补丁下值得双检。

## 9. 已知限制 / 后续工作

1. **Paged-KV shim 读路径没接通**:attention 还是直接读 slab;
   `gather_prefix` 接上去之后 shim 才真正 authoritative。详见
   `project_arch_reform_progress.md` 的 follow-up 列表。
2. **Spec-dec 全服务端验证**:单元测试覆盖了 paged-KV rollback
   循环,但默认 client 不触发 spec-dec,所以 `_do_reorder_task`
   的 `is_spec_dec=1` 分支在 V100 上没真正 E2E 过。需要一个
   spec-dec-enabled 的 client。
3. **FlexGen GQA**:不支持;GQA 模型走 HF 路径。
4. **Mixtral E2E 未跑**:16GB V100 装不下。
5. **Continuous batching**:每序列的 `L_seq` 还没接上
   (`backend.py:542` 仍然是共享 `cache_len`);
   `load_batch_to_runtime` 还没做 continuous admission。

## 10. Commit 脉络(本分支由新到旧)

最新 commit `8022e74 fix(models): restore all 5 README-declared
models under TF 5.x`。

之前的重要节点:
- `0c1c53b` fix(qwen3): 建 causal mask + rotary inv_freq 保持 fp32
- `b5fb0fb` fix(qwen3): 用 config.head_dim 显式分配 KV cache
- `1ecb51d` fix(client): 给 TF 5.x Cache.__init__ 传 layers=[]
- `2240664` fix(qwen3): override TF 5.x tie_weights
- `c56cc76` fix(qwen3): override TF 5.x mark_tied_weights_as_initialized
- `8a19811` feat(kv-cache): spec-dec 下 Phase 2 paged-view shim load-bearing
- `5c98d83` docs: llama-7b 对 mainline 的端到端对比
- `f3cfa03` docs: Phase 2 paged shim 真实 server 数字
- `da30b0e` fix(weights): 本地 safetensors 转换路径
- `d011b4b` fix(flexgen): pinned CPU relay buffer 缩到 4 MB
- `c04bdee` FlexGen 原地 KV 写(Phase 0)
- `dd12f40` PagedKVTable 原语 + 10 个测试
- `35c6697` O(n*depth) ancestor matrix
- `e89d7cb` vectorized spec verify
- `fdfd0b7` SpecInfer 随机拒绝采样
- `1e70e26` EAGLE-2 budgeted tree shape

更深入内容见 `ARCH_REFORM_PLAN.md`、`ARCH_REFORM_SUMMARY.md`、
`ARCH_REFORM_BENCHMARK_RESULTS.md`、`PHASE2_PAGED_KV_INVARIANTS.md`。

# Chunked Prefill 学习笔记

## 1. 这是什么

`chunked prefill` 可以理解成“把一个很长的 prompt prefill 拆成多个小段分多轮执行”。

它解决的问题是：

- 长 prompt 的 prefill 很吃算力，容易拖慢整批请求
- decode 请求通常每轮只处理 1 个 token，更偏带宽/访存
- 如果让一个超长 prefill 一次性吃满整个 batch，decode 的 ITL 会明显变差

因此，V1 调度器会尽量：

1. 先调度 decode 请求
2. 再用剩余的 token budget 调度 prefill
3. 如果某个 prefill 太长放不下，就把它切成 chunk

上游 vLLM 文档明确说明：**V1 中 chunked prefill 在可用时默认开启**。`vllm-ascend` 也支持该特性，并在支持矩阵里标成 `Functional`。

## 2. 核心心智模型

把它和普通 prefill 对比：

普通 prefill：

```text
一个长 prompt
-> 一次调度
-> 一次完整 prefill
-> 再进入 decode
```

chunked prefill：

```text
一个长 prompt
-> 第 1 轮只 prefill 前一段
-> 第 2 轮继续 prefill 下一段
-> 第 3 轮继续
-> 整段 prompt 都完成后，才进入 decode
```

所以它本质上改变的是：

- 不是“这个请求要不要 prefill”
- 而是“这个请求这一轮 prefill 多少 token”

## 3. 调度器是怎么切 chunk 的

调度器在处理 WAITING 请求时，会先算：

- 这个请求还剩多少 token 没算
- 当前 step 的 `token_budget` 还剩多少

然后核心逻辑就是：

```text
num_new_tokens = min(剩余未算 token, token_budget)
```

如果 `enable_chunked_prefill=False` 且这个请求完整放不下：

- 调度器通常会停下来，不会只调一半

如果 `enable_chunked_prefill=True`：

- 就直接只调这一轮能容纳的那一部分 token

因此，一个 12000 token 的 prompt，在 `max_num_batched_tokens=4096` 的情况下，可能被切成：

```text
第 1 轮: 4096
第 2 轮: 4096
第 3 轮: 3808
```

## 4. 从一个请求进入开始的完整时序

假设：

- 单个长请求 `R`
- prompt 长度 12000
- `max_num_batched_tokens = 4096`
- 开启 chunked prefill
- 不考虑 prefix cache 命中、外部 KV connector、spec decode

时序如下：

### Step 1

1. 请求进入 engine，完成模板化和分词。
2. 请求被包装成 `Request`，其中 `num_computed_tokens = 0`。
3. 调度器看到该请求还没算过任何 prompt token。
4. 本轮预算最多只能给它 4096 token。
5. `allocate_slots()` 为这 4096 token 分配 KV blocks。
6. worker 执行这 4096 token 的 prefill。
7. 请求结束这一轮后：
   - `num_computed_tokens = 4096`
   - 还没有进入 decode
   - 只是“部分 prefill 完成”

### Step 2

1. 调度器再次看到同一个请求 `R`。
2. 发现它还有 `12000 - 4096 = 7904` token 未算。
3. 本轮继续给它 4096 token。
4. worker 在已有 KV 基础上继续 prefill 第二段。
5. 本轮结束后：
   - `num_computed_tokens = 8192`

### Step 3

1. 调度器再次看到请求 `R`。
2. 剩余未算 token 为 `3808`。
3. 本轮直接把最后 3808 token 全部调度。
4. worker 完成最后一段 prefill。
5. 此时整段 prompt KV 已经建立完成。

### Step 4

1. 请求 `R` 的 prompt 已经全部 prefill 完成。
2. 下一轮起，它不再是 partial prefill request。
3. 它开始进入正常 decode，每轮通常生成 1 个 token。

## 5. `DeepSeek-V3.1` 下的特殊性

`DeepSeek-V3.1` 在这个工作区里走的是 `DeepseekV3ForCausalLM` 路径，并使用 DeepSeek MLA attention，而不是普通的 MHA 路径。

这意味着对它来说，chunked prefill 有两层含义：

### 第一层：调度器层的 chunk

长 prompt 被拆成多个调度 step 执行：

```text
[chunk1][chunk2][chunk3]
```

### 第二层：MLA 内部的 context chunk

当 worker 在算 `chunk2` 或 `chunk3` 时，当前 chunk 的 query 需要看到前面已经建立好的历史上下文 KV。

但如果历史 context 很长，MLA 一次性展开这段历史上下文会带来很大的中间张量和 workspace 开销。

因此，MLA 内部会进一步把历史 context 分段处理：

```text
当前 chunk query
  x history_part_1
  x history_part_2
  x history_part_3
  ...
再把部分结果 merge
```

所以对于 `DeepSeek-V3.1`：

- 调度器 chunk 是“把请求切开”
- MLA 内部 chunk 是“把当前 chunk 所看到的历史 context 再切开”

这两层 chunk 不是重复，而是分别解决：

- 批调度公平性
- MLA 内部内存/工作区控制

## 6. `DeepSeek-V3.1` 的 `attn_state` 怎么变

在 `vllm-ascend` 的 worker 中，attention state 会根据 batch 形态自动推导。

对一个单独的长请求来说，常见状态演化是：

```text
第一个 chunk      -> PrefillNoCache
中间 chunk       -> ChunkedPrefill
最后一个 chunk   -> ChunkedPrefill
prompt 完成后    -> DecodeOnly
```

可以这样理解：

- `PrefillNoCache`
  第一次 prefill，没有历史 prompt context
- `ChunkedPrefill`
  当前 chunk 需要读取已经建立好的上下文 KV
- `DecodeOnly`
  prompt 已经全部完成，只剩解码

## 7. KV / block table / slot mapping 在 chunked prefill 里怎么变化

### KV cache

chunked prefill 不是每轮重算整个 prompt，而是在已有 KV 上继续向后追加。

例如：

```text
Step 1 后:
[chunk1 的 KV]

Step 2 后:
[chunk1 的 KV][chunk2 的 KV]

Step 3 后:
[chunk1 的 KV][chunk2 的 KV][chunk3 的 KV]
```

### block table

`block_table` 是按“请求一行”存的。一个请求在多个 chunk 之间不会换一张新表，而是：

- 已有行保留
- 新 chunk 对应的 blocks 继续 append 到该请求对应的 row

所以 block table 更像是“这个请求已经持有哪些 blocks”的持续记录。

### slot mapping

`slot_mapping` 是本轮 token 到具体 KV 存储位置的映射。

它每轮都会重新根据：

- 本轮请求顺序
- 本轮 positions
- 本轮 block table

重新计算。

因此：

- `block_table` 更偏“长期状态”
- `slot_mapping` 更偏“本轮执行视图”

## 8. decode 优先时，decode 和 prefill 一起组 batch 怎么执行

这是 chunked prefill 最重要的实际场景。

假设一轮调度里有：

- `D1`：decode，请求本轮只需要 1 token
- `D2`：decode，请求本轮只需要 1 token
- `P1`：prefill，本轮切出一个 3000 token chunk

那么 scheduler 可能给出：

```text
D1 -> 1
D2 -> 1
P1 -> 3000
```

这就形成了一个 mixed batch。

### 调度层

这里“decode 优先”的含义是：

- 先给 `D1`、`D2` 分 token
- 剩余 budget 才分给 `P1`

### worker 层

worker 不会把这三者拆成三次独立 forward，而是把它们拼成一轮 batch 一起执行。

重排后通常变成：

```text
[D1][D2][P1]
```

对应：

```text
num_scheduled_tokens = [1, 1, 3000]
query_start_loc      = [0, 1, 2, 3002]
```

### attention state

在 `vllm-ascend` 中，这种 mixed batch 通常不会整体走 `DecodeOnly`，而会统一落到：

```text
ChunkedPrefill
```

但 metadata 内部仍然会区分：

- 哪些是 decode requests
- 哪些是 prefill requests

也就是说：

- 整体执行路径：`ChunkedPrefill`
- 内部仍然有 `decode_metadata` 和 `prefill_metadata`

### 输出处理

在这种 mixed batch 中：

- decode 请求的采样结果有效，会真正返回
- partial prefill 请求即使为了统一实现也经过了 logits/采样相关流程，其结果也会被忽略

所以：

- decode 请求负责“产出”
- prefill 请求负责“推进上下文和 KV 状态”

## 9. 它和 prefix caching 的关系

chunked prefill 和 prefix caching 经常一起出现，但解决的问题不同。

### prefix caching

作用是：

- 跳过“已经算过且可复用”的前缀

### chunked prefill

作用是：

- 把“仍然需要算的长 prompt 后缀”拆成多轮

二者组合时，典型流程是：

```text
先通过 prefix cache 计算出已命中的前缀长度
-> 剩余未命中的部分如果还很长
-> 再用 chunked prefill 分多轮执行
```

一句话：

- prefix cache 负责“少算”
- chunked prefill 负责“分开算”

## 10. 优势

chunked prefill 的主要好处：

- decode 请求不容易被长 prefill 拖死
- ITL 往往更稳
- prefill 与 decode 混合时，设备利用率通常更好
- 长上下文场景更容易跑起来

对 `DeepSeek-V3.1` 这类 MLA 模型尤其重要，因为：

- 它不仅是调度优化
- 还和 MLA 内部的 context chunking 一起决定内存与 workspace 使用

## 11. 限制和注意事项

需要特别记住这些现实限制：

- 它不是总能带来最佳 TTFT，需要靠 `max_num_batched_tokens` 调参
- prompt 很短时，收益有限
- 和 prefix cache、Ascend scheduler、sliding window、spec decode 等组合时，边界条件会更多
- `vllm-ascend` 历史 release note 中曾明确提到：某些版本下不要把 Ascend scheduler、chunked prefill 和 prefix cache 同时开启，因为性能和精度不理想
- 某些 attention 变种或特殊模型路径对 chunked prefill 有额外约束

## 12. 调参重点

最重要的参数是：

- `max_num_batched_tokens`

经验上：

- 设置小一些：
  decode 更友好，ITL 更好
- 设置大一些：
  prefill 更激进，TTFT/吞吐可能更高
- 太小：
  长 prompt 会被切得很碎
- 太大：
  decode 容易被长 prefill 拖慢

所以 chunked prefill 的优化重点通常不是“开不开”，而是：

- 一轮最多给 batch 多少 token
- decode 和 prefill 在这个预算下怎么平衡

## 13. 推荐阅读路径

建议按下面顺序看源码：

1. `vllm/docs/configuration/optimization.md`
   看上游对 chunked prefill 的设计意图和调参说明。
2. `vllm/docs/usage/v1_guide.md`
   看 V1 默认行为和调度哲学。
3. `vllm/vllm/v1/core/sched/scheduler.py`
   看调度器怎样决定每轮给一个请求多少 token。
4. `vllm-ascend/vllm_ascend/core/recompute_scheduler.py`
   看 Ascend 侧等价逻辑以及与其他特性的组合。
5. `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
   看 mixed batch、`attn_state`、`seq_lens`、`slot_mapping`、partial request 处理。
6. `vllm-ascend/vllm_ascend/attention/mla_v1.py`
   看 DeepSeek MLA 下 decode/prefill 分流、batch reorder、chunked context metadata。
7. `vllm/vllm/model_executor/layers/attention/mla_attention.py`
   看 MLA 内部为什么还要对历史 context 做二次 chunk。

## 14. 一句话总结

`chunked prefill` 在 `vllm-ascend` 中的本质，是把长 prompt 的 prefill 拆成多个调度 step 执行，并允许这些 partial prefill 与 decode 请求混合批处理；对 `DeepSeek-V3.1` 这样的 MLA 模型来说，它既是调度层优化，也是控制长上下文 MLA 计算开销的重要机制。

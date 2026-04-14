# KV Cache 学习笔记

## 1. 先给结论

在这个项目里，KV cache 的**核心管理粒度是 block**。

不是按 token 管，也不是按整条 request 管，而是按 block 管。这样做的原因是：

- 按 token 管：元数据过碎，分配、回收、复用开销太大
- 按 request 管：粒度过粗，无法支持 prefix cache、chunked prefill、局部释放、局部驱逐
- 按 block 管：刚好兼顾复用粒度、调度效率、内存局部性和实现复杂度

一句话：

**KV cache 的“管理单位”是 block。**

## 2. 三层视角

理解 KV cache 时，最好把它拆成三层：

### 2.1 物理存储层

真正放在设备内存里的 KV tensor。

它回答的问题是：

- 数据放在哪里
- 张量 shape 是什么
- K/V 是否分开
- MLA 模型是否有特殊布局

### 2.2 block 管理层

这一层负责：

- 全局 block 池
- block 分配
- block 回收
- 引用计数
- prefix cache 命中
- 缓存驱逐

核心对象是：

- `KVCacheBlock`
- `BlockPool`
- `KVCacheManager`

### 2.3 请求映射层

这一层负责：

- 某个 request 当前持有哪些 block
- 本轮 token 对应哪个 block、哪个 offset

核心对象是：

- `block_table`
- `slot_mapping`
- `req_id_to_index`

## 3. 为什么按 block 管

### 3.1 不是按 token 管

如果按 token 管，会遇到：

- 每个 token 都要独立维护元数据
- allocator 碎片化严重
- prefix cache 很难高效做“连续前缀命中”
- 长请求会变成大量小操作

### 3.2 不是按 request 管

如果按 request 管，会失去：

- 前缀部分复用
- chunked prefill 的增量构建
- sliding window 的局部丢弃
- LRU 风格的块级驱逐

所以 request 粒度太粗，不适合 V1 统一调度架构。

### 3.3 block 是工程上最合适的粒度

block 粒度下可以同时支持：

- prefix caching
- chunked prefill
- mixed batch
- 局部回收和驱逐
- 各种 attention / MLA / sparse / quantized 路径

## 4. `KVCacheBlock` 是什么

`KVCacheBlock` 是一个**逻辑块的元数据对象**，不是数据本体。

它通常有这些关键字段：

- `block_id`
- `ref_cnt`
- `block_hash`
- free queue 指针

你可以把它理解成：

```text
block_id = 17
ref_cnt = 1
block_hash = 某个前缀链 hash
```

这里：

- `block_id` 是逻辑编号
- `ref_cnt` 表示当前有多少活跃请求在用它
- `block_hash` 决定它能否作为 prefix cache 命中目标

## 5. `BlockPool`：全局块库存

`BlockPool` 是整个本地 KV cache 的全局库存。

它做的事情包括：

- 启动时预创建所有 block
- 维护 free queue
- 维护 cached block hash map
- 分配新 block
- free block
- touch block
- evict block
- reset prefix cache

最重要的设计点是：

**block 一开始就全部创建好，形成 block pool。**

这样可以避免运行时频繁创建 Python 对象，也方便统一做生命周期管理。

## 6. `ref_cnt`、free queue、cached block、eviction 的关系

这四个概念最容易混。

一句话记忆：

- `ref_cnt`：这个 block 现在有没有被请求持有
- `free queue`：这个 block 现在能不能被重分配
- `cached block`：这个 block 现在还能不能被 prefix cache 命中
- `eviction`：把一个 cached block 的缓存身份取消，然后拿去重用

### 6.1 `ref_cnt`

- `ref_cnt > 0`：这个 block 正在被某个或多个请求使用
- `ref_cnt == 0`：当前没有活跃请求在用它

### 6.2 free queue

free queue 里放的是：

- 当前 `ref_cnt == 0`
- 可以被未来分配/重用的 block

注意：

**在 free queue 里的 block 仍然可能是 cached block。**

### 6.3 cached block

只要一个 block：

- 已经是 full block
- 有 `block_hash`
- 且已插入 cached map

它就是 cached block。

cached 的含义是：

**后续请求可以通过 block hash 命中它。**

### 6.4 eviction

当 allocator 想重用一个空闲 block，而这个 block 恰好还是 cached block 时，会先：

- 从 cached map 里移除
- 清掉 `block_hash`

这就叫 eviction。

所以 eviction 的本质不是 free，而是：

**取消它的“可被 prefix cache 命中”的身份。**

### 6.5 一个 block 的四种典型状态

#### A. 正在用 + 已缓存

```text
ref_cnt > 0
block_hash != None
不在 free queue
```

#### B. 空闲 + 已缓存

```text
ref_cnt == 0
block_hash != None
在 free queue
```

这是最有价值的 prefix cache 块。

#### C. 正在用 + 未缓存

```text
ref_cnt > 0
block_hash == None
不在 free queue
```

常见于：

- 尚未填满的最后一个块

#### D. 空闲 + 未缓存

```text
ref_cnt == 0
block_hash == None
在 free queue
```

这是普通空闲块。

## 7. `KVCacheManager` 管什么

`KVCacheManager` 是请求和 block 之间的逻辑中枢。

它不直接关心物理 tensor 怎么排，而是关心：

- 某个请求当前命中了哪些 block
- 还要分配多少新 block
- 哪些 full block 可以进入 prefix cache
- 请求结束时该释放哪些 block

最关键的动作：

- `get_computed_blocks()`：找本地 prefix hit
- `allocate_slots()`：把命中块和新块拼成请求的 block 布局
- `cache_blocks()`：把 full blocks 写入 prefix cache
- `free()`：释放 request 持有的 blocks

所以可以把它理解成：

**请求视角下的 block 生命周期管理器。**

## 8. 一个请求在系统里的完整生命周期

假设请求 `R` 进入系统。

### 8.1 Request 创建

系统先创建 `Request`，其中包括：

- `request_id`
- `prompt_token_ids`
- `num_computed_tokens = 0`
- `block_hashes`

此时它还没有真正占用 KV block。

### 8.2 查询可复用 block

调度器会先问 `KVCacheManager`：

- 这个请求有没有 prefix cache 命中

如果没有命中，就需要新分 block。

### 8.3 分配 block

调度器调用 `allocate_slots()`。

这一动作内部会：

1. 算出这轮需要多少 blocks
2. 向 `BlockPool` 申请新块
3. 建立 `request_id -> blocks` 的映射

此时请求就真正“持有”了一串 blocks。

### 8.4 写入 block table

接下来，worker/input batch 会把这个请求的 `block_ids` 写入 `block_table` 的一行。

也就是：

```text
req_id -> row_idx
block_table[row_idx] = 该请求的 block ids
```

### 8.5 计算 slot mapping

有了 block table 之后，系统还要根据：

- 本轮 token 对应的请求
- 这些 token 的 position

计算 `slot_mapping`。

这一步把：

- token
- block_id
- block_offset

合成最终的 slot 编号。

### 8.6 worker 执行 forward

真正的 attention / cache 写入 kernel 用的是：

- `block_table`
- `slot_mapping`
- `query_lens`
- `seq_lens`
- `positions`

### 8.7 请求继续或结束

如果请求还没结束：

- 更新 `num_computed_tokens`
- 下一轮继续在已有 block 基础上追加

如果请求结束：

- `KVCacheManager.free()`
- `BlockPool.free_blocks()`
- `ref_cnt` 归零的 block 回到 free queue

## 9. `block_table` 是什么

`block_table` 是请求到 block 的映射表。

在 `vllm-ascend` 的 `NPUInputBatch` 中，它由 `MultiGroupBlockTable` 管理。

你可以把它理解成二维表：

```text
row 0 -> D1 的 block ids
row 1 -> D2 的 block ids
row 2 -> P1 的 block ids
```

其中：

- 一行对应一个 request
- 一列是这个 request 当前持有的某个 block_id

### 为什么 mixed batch 不会把 D1、D2、P1 混掉

因为系统一直维护：

```text
req_id_to_index:
D1 -> 0
D2 -> 1
P1 -> 2
```

并且在 batch reorder 时：

- `req_id_to_index` 会更新
- `block_table` 对应的行也会一起交换

所以 request 和 row 的绑定始终一致。

## 10. `slot_mapping` 是什么

如果说 `block_table` 解决的是：

- 这个请求有哪些 blocks

那么 `slot_mapping` 解决的是：

- 本轮这个 token 写到哪个 block 的哪个 offset

核心公式可以理解成：

```text
slot_mapping = block_id * block_size + block_offset
```

其中：

- `block_id` 来自 `block_table`
- `block_offset` 来自 token 在请求中的 `position % block_size`

所以：

- `block_table` 是请求级映射
- `slot_mapping` 是 token 级映射

## 11. `block_id` 到真实 tensor 地址的链条

`block_id` 本身不是物理地址，而是逻辑页号。

完整链条是：

```text
request
-> row_idx
-> block_table[row_idx]
-> block_id
-> position 计算 block_offset
-> slot_mapping
-> attention / reshape_and_cache kernel
-> 具体模型的 KV layout
-> 真实 tensor 位置
```

### 为什么 `block_id` 不是物理地址

因为一个 block 只是逻辑编号：

```text
block_id = 17
```

还不够回答：

- 这个请求里它是第几个逻辑块
- 这个 token 落在块内哪个 offset
- 对应的真实 tensor 怎么解释

### 为什么 `slot_mapping` 更接近真实地址

因为它已经把：

- `block_id`
- `offset`

合成了一个 token 级 slot 位置，后续 kernel 可以直接依据它把 KV scatter 到 cache tensor。

## 12. 逻辑管理层与物理存储层为什么要分开

这点非常重要。

### 逻辑管理层统一按 block 管

统一支持：

- 分配
- 回收
- prefix cache
- chunked prefill
- mixed batch
- eviction

### 物理存储层允许模型特化

不同模型可以有不同的 KV 物理布局。

对普通 attention 模型，物理上看起来更像：

```text
block 17 -> key_cache[17] + value_cache[17]
```

但对 `DeepSeek-V3.1` 这种 MLA 模型，就不一样了。

## 13. `DeepSeek-V3.1` 的特殊性

`DeepSeek-V3.1` 在这个工作区里走的是 DeepSeek MLA 路径。

这意味着：

- 逻辑管理粒度仍然是 block
- 但物理 KV 布局不是普通 K/V 二元结构

在上游模型实现里，DeepSeek MLA 使用的是 `MLAAttentionSpec`，而不是普通 K/V spec。

在 `vllm-ascend` worker 里，分配 KV cache tensor 时也会按 MLA 的 nope / rope 等维度拆分。

所以：

- `block_table` 里记录的仍然只是逻辑 `block_id`
- 但这个 `block_id` 最终映射到哪一块物理 tensor，需要由 DeepSeek MLA 的 attention/KV layout 去解释

一句话：

**逻辑层是统一 block 管理，物理层是模型特化布局。**

## 14. 它和 prefix caching 的关系

prefix caching 不是另一套 KV cache 系统，它只是建立在 block 管理之上的复用策略。

它依赖：

- `block_hash`
- `cached block`
- `touch`
- `free queue`
- `eviction`

所以可以这样看：

- KV cache 管理：底座
- prefix caching：在底座上复用已有 block

## 15. 它和 chunked prefill 的关系

chunked prefill 也不是另一套 KV cache 系统，它只是改变：

- 一次调度到底新申请多少 token 对应的 blocks

普通 prefill：

- 一次可能给完整 prompt 分很多 blocks

chunked prefill：

- 每轮只给当前 chunk 分新增 blocks
- 在已有 request blocks 基础上继续 append

所以：

- KV cache 管理：底座
- chunked prefill：在底座上控制 block 的增量增长节奏

## 16. 推荐阅读路径

建议按下面顺序回看源码：

1. `vllm/vllm/v1/core/kv_cache_utils.py`
   先看 `KVCacheBlock`、block hash 基础类型。
2. `vllm/vllm/v1/core/block_pool.py`
   看 block pool、free queue、cached map、touch/free/evict/reset。
3. `vllm/vllm/v1/core/kv_cache_manager.py`
   看请求如何申请、复用、释放 blocks。
4. `vllm-ascend/vllm_ascend/worker/npu_input_batch.py`
   看 block table 在 batch 中如何存放。
5. `vllm-ascend/vllm_ascend/worker/block_table.py`
   看 `block_table` 和 `slot_mapping` 如何计算。
6. `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
   看 worker 如何把 block table / slot mapping 塞进 attention metadata。
7. `vllm/vllm/model_executor/models/deepseek_v2.py`
   看 DeepSeek MLA 的 cache spec。
8. `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
   看 DeepSeek MLA 在 Ascend 上的 KV tensor 分配和布局特化。

## 17. 一句话总结

`vllm-ascend` 里的 KV cache 可以理解成：系统用统一的 block 生命周期管理（BlockPool + KVCacheManager + BlockTable）来组织请求的上下文状态，再通过 `slot_mapping` 把本轮 token 映射到真实 KV tensor；而像 `DeepSeek-V3.1` 这样的 MLA 模型，只改变 block 背后的物理存储布局，不改变 block 作为统一管理单位这一事实。

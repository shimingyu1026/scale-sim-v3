# SCALE-Sim v3 输入到结果输出流程与 Batch 支持汇报文档

## 文档定位

本文面向“研究生向导师汇报”的场景，集中回答两个问题：

1. SCALE-Sim v3 从输入到结果输出，实际经过了哪些模块，每个模块的输入和输出是什么。
2. 为了支持 `Batch > 1`，代码中做了哪些关键改动，这些改动背后的原理是什么。

如果只用一句话概括这次工作，可以表述为：

> SCALE-Sim 原本更接近“单样本工作负载模拟器”，而本次 batch 支持不是在结果阶段简单乘一个 `B`，而是把 batch 前移到 topology 语义、operand address 生成和工作负载展开层，使 compute、memory 和 report 都直接消费完整 batched workload。

---

## 第一部分：SCALE-Sim 从输入到结果输出的完整模块链路

### 1.1 总体主线

一次 SCALE-Sim 运行的主调用链如下：

```text
scalesim/scale.py
  -> scalesim/scale_sim.py::scalesim
    -> scalesim/simulator.py::simulator
      -> scalesim/single_layer_sim.py::single_layer_sim
        -> scalesim/compute/operand_matrix.py
        -> scalesim/compute/systolic_compute_{os,ws,is}.py
        -> scalesim/memory/double_buffered_scratchpad_mem.py
        -> scalesim/energy/accelergy_plugin.py
```

把这条链路翻译成更容易汇报的语言，就是：

```text
命令行/配置输入
  -> 解析出体系结构参数、拓扑参数、布局参数
  -> 把每层变成 IFMAP/FILTER/OFMAP 地址矩阵
  -> 按 OS / WS / IS 数据流生成 prefetch 与逐周期 demand
  -> 送入 scratchpad + DRAM 模型得到 stall、周期和 trace
  -> 汇总成 CSV 报告与逐层 trace 文件
```

### 1.2 模块级输入输出总表

| 阶段 | 核心文件 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| 入口层 | `scalesim/scale.py` | CLI 参数：`-c/-t/-l/-p/-i/-s` | `scalesim` 顶层对象及运行请求 | 收集用户输入并启动一次模拟 |
| 全局装载层 | `scalesim/scale_sim.py` | 配置文件、拓扑文件、布局文件路径 | `config`、`topo`、`layout` 三个全局对象 | 解析输入文件，建立一次 run 的上下文 |
| 配置解析层 | `scalesim/scale_config.py` | `.cfg` 文件 | 阵列规模、SRAM 容量、数据流、地址 offset、带宽模式、稀疏和 layout 开关 | 提供体系结构与运行模式参数 |
| 拓扑解析层 | `scalesim/topology_utils.py` | 拓扑 CSV | 每层的统一内部表示、派生超参数、batch/sparsity 信息 | 提供每层 workload 语义 |
| 布局解析层 | `scalesim/layout_utils.py` | layout CSV | IFMAP/FILTER 的 intraline/interline factor 与 order | 影响 custom layout 下的预取顺序 |
| run 级调度层 | `scalesim/simulator.py` | `config/topo/layout` 对象 | 多个 `single_layer_sim` 实例与 run 级输出目录 | 管理整网逐层运行和汇总 |
| 单层总控层 | `scalesim/single_layer_sim.py` | 某一层的 `layer_id + config + topo + layout` | 单层的 compute/memory/report 结果 | 串联地址生成、compute 建模、memory 建模和报告抽取 |
| 地址生成层 | `scalesim/compute/operand_matrix.py` | 单层几何参数、stride、batch、offset、sparsity | IFMAP/FILTER/OFMAP 三张地址矩阵 | 把 layer 转换成后续数据流可消费的 operand space |
| 数据流计算层 | `scalesim/compute/systolic_compute_{os,ws,is}.py` | operand matrices + array size | prefetch matrices、demand matrices、请求数、映射效率、计算利用率 | 把地址矩阵映射到 systolic array 时序 |
| 存储系统层 | `scalesim/memory/double_buffered_scratchpad_mem.py` | prefetch + demand + SRAM/DRAM 参数 | 总周期、stall、SRAM trace、DRAM trace、访问计数 | 把逻辑请求变成带存储瓶颈的时序结果 |
| 报告输出层 | `single_layer_sim.calc_report_data()` 与 `simulator.generate_reports()` | 单层真实计数器和周期 | `COMPUTE_REPORT.csv`、`BANDWIDTH_REPORT.csv`、`DETAILED_ACCESS_REPORT.csv`、trace 文件等 | 形成用户最终可见结果 |

### 1.3 输入端：用户给模拟器的到底是什么

#### 1.3.1 CLI 输入

入口文件 `scalesim/scale.py` 读取以下参数：

- `-c`：配置文件路径
- `-t`：拓扑文件路径
- `-l`：布局文件路径
- `-p`：输出目录
- `-i`：输入类型，`conv` 或 `gemm`
- `-s`：是否保存 trace

这里的核心作用是把“用户视角的路径和模式”转换成程序可直接消费的统一运行请求。

#### 1.3.2 配置文件输入

`scale_config.read_conf_file()` 负责读取：

- 阵列尺寸：`ArrayHeight`、`ArrayWidth`
- 片上 SRAM 容量：IFMAP / FILTER / OFMAP
- 数据流模式：`os` / `ws` / `is`
- 地址空间偏移：`IfmapOffset`、`FilterOffset`、`OfmapOffset`
- 带宽模式：`USER` 或 `CALC`
- 稀疏相关选项
- custom layout 相关选项

它的输出不是 trace，也不是周期，而是一组“体系结构约束”。后面所有计算都建立在这些约束之上。

#### 1.3.3 拓扑文件输入

`topology_utils.py` 负责把拓扑 CSV 解析成每层的统一内部表示：

```text
[name, ifmap_h, ifmap_w, filt_h, filt_w, channels, num_filters,
 stride_h, stride_w, batch, sparsity_n, sparsity_m]
```

无论输入原来是卷积格式还是 GEMM 格式，后面都会转成这套统一语义。

在此基础上，`topo_calc_hyperparams()` 还会派生出：

- `ofmap_h`
- `ofmap_w`
- `window_size = filt_h * filt_w * channels`
- `num_mac`（单样本语义下的 MAC 数）

所以拓扑模块的输出可以理解成两层：

1. 原始层参数
2. 供 compute/memory 使用的派生层超参数

#### 1.3.4 布局文件输入

`layout_utils.py` 读取 layout CSV，输出：

- IFMAP intraline factor/order
- IFMAP interline order
- FILTER intraline factor/order
- FILTER interline order

这部分只在 custom layout 开启时生效，作用主要是改变 prefetch 顺序，而不是改变算子数学本身。

### 1.4 run 级调度：如何从“整网输入”进入“逐层运行”

`scale_sim.py` 中的 `scalesim` 类负责装载一次 run 的全局对象：

- `self.config`
- `self.topo`
- `self.layout`
- `self.runner = simulator()`

随后 `run_scale()` 调用 `simulator.set_params()`，把这些全局对象交给 run 级调度器。

`simulator.py` 的职责是：

1. 根据 `topo.get_num_layers()` 创建每层一个 `single_layer_sim`
2. 为本次运行创建输出目录 `<top_path>/<run_name>/`
3. 逐层调用 `single_layer_sim.run()`
4. 逐层调用 `run_energy_model()` 和 `save_traces()`
5. 最后把所有层结果汇总成 run 级 CSV

因此，`simulator.py` 的输入是“整网对象”，输出是“整网的报告文件和逐层 trace 文件”。

### 1.5 单层模拟：真正的核心执行体

`single_layer_sim.py` 是整个模拟过程最关键的中枢。

它的输入是：

- 当前层 `layer_id`
- 配置对象 `config`
- 拓扑对象 `topo`
- 布局对象 `layout`

它的输出是：

- 单层 compute 指标
- 单层 bandwidth 指标
- 单层 detailed access 指标
- 单层 SRAM / DRAM trace

从逻辑上看，它把一层的运行拆成两大阶段：

1. 计算侧：`operand matrix -> prefetch -> demand`
2. 存储侧：`demand -> stall/cycle/trace`

### 1.6 地址生成模块：把 layer 变成 operand matrices

`operand_matrix.py` 的作用是把一层卷积或 GEMM，转成三张“地址化操作数矩阵”：

- `ifmap_addr_matrix`
- `filter_addr_matrix`
- `ofmap_addr_matrix`

其典型形状为：

```text
ifmap_addr_matrix  : [batch * ofmap_px_per_filt, window_size]
filter_addr_matrix : [window_size, num_filters]
ofmap_addr_matrix  : [batch * ofmap_px_per_filt, num_filters]
```

这三张矩阵的含义分别是：

- IFMAP：每一个输出像素位置，对应需要读取哪一组输入地址
- FILTER：每一个输出通道，对应需要读取哪一组权重地址
- OFMAP：每一个输出位置和输出通道，对应结果应该写到哪里

所以，这一层的输入是“层几何参数和地址 offset”，输出是“可被 systolic-array 数据流直接消费的地址矩阵”。

### 1.7 数据流计算模块：把 operand matrices 变成 prefetch 和 demand

SCALE-Sim 提供三种数据流实现：

- `systolic_compute_os.py`
- `systolic_compute_ws.py`
- `systolic_compute_is.py`

三者的输入一致：

- `ifmap_op_mat`
- `filter_op_mat`
- `ofmap_op_mat`
- 阵列尺寸 `arr_row/arr_col`
- 稀疏相关参数（主要是 WS 路径）

三者的输出主要包括两类矩阵和一组统计量：

#### 第一类输出：prefetch matrices

- `ifmap_prefetch_matrix`
- `filter_prefetch_matrix`

它们表示正式计算开始前，片上缓冲需要预先装入哪些地址。

#### 第二类输出：demand matrices

- `ifmap_demand_matrix`
- `filter_demand_matrix`
- `ofmap_demand_matrix`

这些矩阵的每一行表示一个逻辑周期内阵列端口发出的请求，后续会直接送进 memory model。

#### 第三类输出：compute 统计量

- IFMAP/FILTER/OFMAP 请求数
- mapping efficiency
- compute utilization
- PE action counts

从汇报角度看，这一步的核心作用是：

> 把“地址空间中的操作数关系”进一步转换成“阵列时间维度上的访问序列”。

### 1.8 存储系统模块：把 demand 变成 stall、周期和 trace

`double_buffered_scratchpad_mem.py` 负责模拟：

- IFMAP 读缓冲
- FILTER 读缓冲
- OFMAP 写缓冲
- 它们与 backing DRAM 之间的交互

其输入是：

- IFMAP/FILTER 的 prefetch matrices
- IFMAP/FILTER/OFMAP 的 demand matrices
- SRAM 容量、端口数、bank 数、外部带宽等配置参数

其输出是：

- `total_cycles`
- `stall_cycles`
- IFMAP/FILTER/OFMAP 的 SRAM trace
- IFMAP/FILTER/OFMAP 的 DRAM trace
- 各类 DRAM 访问总数与起止周期

`service_memory_requests()` 的核心机制是：

1. 逐行读取 demand matrix
2. 分别把该行请求送入 IFMAP/FILTER/OFMAP 缓冲
3. 计算该逻辑周期上三条路径的完成时刻
4. 取最慢者作为该周期真实前进速度
5. 累加 stall

这一步的意义是把“理想计算请求”转成“考虑片上片外存储瓶颈后的真实时序”。

### 1.9 单层报告抽取：从底层计数器生成用户可读指标

`single_layer_sim.calc_report_data()` 会把真实周期和访问计数整理成三类报告字段。

#### 1.9.1 Compute report

核心字段包括：

- 总周期
- stall 周期
- overall utilization
- mapping efficiency
- compute utilization

其中最核心的计算关系是：

```text
overall_util = num_compute * 100 / (total_cycles * num_mac_unit)
```

#### 1.9.2 Bandwidth report

包括：

- IFMAP/FILTER/OFMAP 的 SRAM 平均带宽
- IFMAP/FILTER/OFMAP 的 DRAM 平均带宽

#### 1.9.3 Detailed access report

包括：

- SRAM 起止周期与访问数
- DRAM 起止周期与访问数
- PE action counts

所以单层报告模块的输入是“真实运行后的周期与访问计数器”，输出是“可直接写入 CSV 的结构化统计项”。

### 1.10 run 级结果输出：最终会生成哪些文件

`simulator.generate_reports()` 最终会在 `<top_path>/<run_name>/` 下输出：

- `COMPUTE_REPORT.csv`
- `BANDWIDTH_REPORT.csv`
- `DETAILED_ACCESS_REPORT.csv`
- `REPEAT_CYCLE.csv`
- `SPARSE_REPORT.csv`（仅稀疏模式）

若启用 trace，还会生成逐层目录：

```text
layer0/
  IFMAP_SRAM_TRACE.csv
  FILTER_SRAM_TRACE.csv
  OFMAP_SRAM_TRACE.csv
  IFMAP_DRAM_TRACE.csv
  FILTER_DRAM_TRACE.csv
  OFMAP_DRAM_TRACE.csv
```

其中：

- CSV 报告是 run 级聚合结果
- trace 文件是 layer 级逐周期访问轨迹

### 1.11 第一部分总结

从系统分层看，SCALE-Sim 的完整链路可以概括为：

```text
输入文件
  -> 统一参数对象
  -> 每层几何与派生超参数
  -> IFMAP/FILTER/OFMAP 地址矩阵
  -> 数据流 prefetch/demand
  -> 双缓冲 scratchpad + DRAM 服务时序
  -> 周期/带宽/访问统计
  -> CSV 报告和 trace 文件
```

因此，这个模拟器不是直接从“拓扑”跳到“周期”，而是经历了以下三个关键中间表示：

1. 统一层参数表示
2. operand address matrices
3. demand matrices

这三个中间表示正是理解 batch 支持为何需要修改多个模块的关键。

---

## 第二部分：为了支持 Batch，做了什么改动，原理是什么

### 2.1 问题背景：为什么 batch 支持不能只在最后乘一个 B

在 batch 支持工作开始前，仓库中存在几个核心问题：

1. `topology_utils.py` 对 `Batch` / `Batch Size` 的解析不够稳健。
2. `operand_matrix.py` 在运行时基本按 `batch = 1` 思维工作。
3. IFMAP 和 OFMAP 地址空间没有真正按样本隔离。
4. `get_layer_num_ofmap_px()`、`get_layer_mac_ops()`、`single_layer_sim.num_compute` 等统计路径本质上仍是单样本语义。

如果仅仅在最终报告阶段把结果乘一个 `B`，会出现以下错误：

- 不同样本会映射到相同 IFMAP / OFMAP 地址，形成地址别名
- operand matrix 的行数不会真正扩展
- fold 数量不会按 batch 改变
- SRAM / DRAM 请求矩阵不会真实变大
- 三种数据流中 batch 对 `Sr/Sc/T` 的影响不会体现出来

因此，本次实现的设计原则是：

> 不是在结果层做“统计缩放”，而是在 workload model 层把 batch 变成模拟器真正看得见的维度。

### 2.2 Batch 的语义定义

#### 2.2.1 卷积场景

设一层卷积的参数为：

- IFMAP：`(H, W, C)`
- FILTER：`(R, S, C, K)`
- OFMAP 空间：`(E, F)`
- batch size：`B`

那么 full-batch 语义下：

- 单样本 IFMAP 大小：`H * W * C`
- 单样本 OFMAP 大小：`E * F * K`
- 完整 batch OFMAP 总元素数：`B * E * F * K`
- 完整 batch 总 MAC 数：`B * E * F * R * S * C * K`

关键点是：

- IFMAP 要按样本复制
- OFMAP 要按样本复制
- FILTER 不复制，仍由所有样本共享

#### 2.2.2 GEMM 场景

对 GEMM 输入 `(M, N, K, Batch=B)`，当前实现采用的语义是：

> 将其看成 `B` 个彼此独立的 GEMM `(M x K) * (K x N)`。

因此：

- 总输出行数变成 `B * M`
- 总 MAC 工作量变成 `B * M * N * K`
- weight/filter 空间保持不变

### 2.3 核心设计原则：batch 在 operand space 中展开

本次实现最重要的选择是：

- 在 topology 和 operand matrix 层引入 batch 语义
- 让 compute path、memory path 和 report path 去消费已经扩展过的 workload

而不是：

- 在 memory 结束后再额外加一层 `for batch in ...`
- 或者在报告阶段直接把请求数、带宽和 MAC 数乘一个 `B`

这种设计的好处是：

1. 地址空间正确，不会跨样本别名。
2. demand matrix 的规模会自然扩大。
3. OS / WS / IS 中 fold 和时空参数会自然变化。
4. 报告仍然可以复用原有公式，只需要修正工作负载语义。

### 2.4 具体改动一：`topology_utils.py` 成为 batch 的唯一可信来源

这是 batch 支持的入口。

#### 2.4.1 解析层改动

主要修改包括：

- 接受 `Batch` 和 `Batch Size` 两种表头，且大小写不敏感
- 若 batch 字段缺失或为空，默认取 `1`
- 明确 batch 在内部拓扑条目中的固定位置
- 增加 `get_layer_batch_size()`
- 如果同一个 topology 中不同层给出不同 batch，直接拒绝并报错

这样做的意义是：

- 旧文件不需要改格式
- 仓库中原本已经带有 `Batch Size` 表头的文件可以正常解析
- 后续所有模块都能稳定读取 batch，而不是靠列位置猜测

#### 2.4.2 派生指标改动

以下接口都被改成 full-batch 语义：

- `get_layer_num_ofmap_px()`
- `get_layer_mac_ops()`
- `get_all_mac_ops()`
- `get_transformed_mnk_dimensions()`
- `calc_spatio_temporal_params()`

这一步的原理是：

- 保留单样本几何超参数 `ofmap_h/ofmap_w/window_size`
- 但把“总输出数”和“总 MAC 数”提升为 full-batch 统计

这样既不会破坏卷积几何关系，又能让上层真正拿到完整工作量。

#### 2.4.3 为什么 helper 路径也必须改

除了 CSV 解析路径，旧的 helper 接口如：

- `load_layer_params_from_list()`
- `append_topo_entry_from_list()`
- `append_layer_entry()`

也要同步 batch 语义。

原因在于：如果 helper 路径仍使用旧格式，batch 有可能被误读成 `stride_w` 或 sparsity 字段，造成静默错误。

### 2.5 具体改动二：`operand_matrix.py` 才是 batch 真正落地的核心

如果说 `topology_utils.py` 解决的是“batch 被正确理解”，那么 `operand_matrix.py` 解决的是“batch 被真实建模”。

#### 2.5.1 新增 batched 行空间

实现中引入了两个概念：

- `ofmap_px_per_filt = ofmap_rows * ofmap_cols`
- `batched_ofmap_px_per_filt = batch_size * ofmap_px_per_filt`

其物理意义分别是：

- 前者：单个样本、单个 filter 的输出像素数
- 后者：完整 batch、单个 filter 的输出像素总数

于是地址矩阵的形状变成：

```text
ifmap_addr_matrix  : [B * E * F, R * S * C]
filter_addr_matrix : [R * S * C, K]
ofmap_addr_matrix  : [B * E * F, K]
```

这正好对应：

- IFMAP 和 OFMAP 都按 batch 扩展
- FILTER 保持共享

#### 2.5.2 IFMAP 地址生成的关键改动

旧逻辑的问题在于：把所有输出行都视为同一个样本中的位置。

新逻辑中，IFMAP 行索引 `i` 会先拆成两部分：

```text
batch_idx, sample_ofmap_idx = divmod(i, ofmap_px_per_filt)
```

然后再计算：

- 当前样本内的 `ofmap_row / ofmap_col`
- 对应卷积窗口在 IFMAP 中的基地址
- 当前样本的基地址偏移

最关键的式子是：

```text
sample_base = batch_idx * (H * W * C)
```

它的意义非常直接：

- 第 `0` 个样本的 IFMAP 地址空间为原始地址段
- 第 `1` 个样本整体后移一个完整样本大小
- 第 `2` 个样本再后移一个完整样本大小

因此 batch 支持后的 IFMAP 地址满足两个重要不变量：

1. 样本内部地址顺序与 `Batch = 1` 时完全一致
2. 不同样本之间地址段严格分离

这一步是整个 batch 支持正确性的核心。

#### 2.5.3 OFMAP 地址生成的关键改动

OFMAP 的实现思路更直接：

- 逻辑行 `i` 已经覆盖完整 batch
- 地址直接按 `num_filters * i + j + ofmap_offset` 线性展开

这意味着整个 batch 的 OFMAP 行空间天然是连续的。

本质上，OFMAP 的 batch 支持不是通过额外加一个特殊循环实现的，而是通过：

- 把行维度直接扩展到 `B * E * F`

从而让输出地址自然覆盖完整 batch。

#### 2.5.4 为什么 FILTER 地址不变

这是一个非常重要但容易误解的点。

batch 复制的是样本，不是权重。因此：

- IFMAP 需要按样本复制
- OFMAP 需要按样本复制
- FILTER 不应该复制

如果把 FILTER 也复制成 `B` 份，反而会破坏权重复用语义，使 WS 等数据流行为失真。

#### 2.5.5 getter 也必须 batch-aware

即使内部矩阵已经扩展，如果公开接口仍按单 batch 返回数据，外部模块仍会看到被截断的矩阵。

因此：

- `get_ifmap_matrix_part()`
- `get_ofmap_matrix_part()`

都改成默认返回完整 batched 行范围，并使用 batched 尺寸做越界检查。

### 2.6 具体改动三：compute 逻辑本身几乎不用重写，但其输入语义已经变化

这是本次实现中非常值得强调的一点。

OS / WS / IS 三个 compute 类本身没有为 batch 新增大规模特判，原因不是 batch 不重要，而是：

> 一旦 operand matrix 的 batch 语义正确，这三个类会自然“看到”更大的工作负载。

#### 2.6.1 OS 中 batch 的落点

在 `calc_spatio_temporal_params()` 中，OS 的定义是：

- `S_r = num_ofmap`
- `S_c = num_filt`
- `T = window_size`

由于 `num_ofmap = B * E * F`，所以在 OS 中：

- batch 主要扩大 `S_r`

直观理解就是：阵列需要依次处理更多输出像素位置。

#### 2.6.2 WS 中 batch 的落点

WS 的定义是：

- `S_r = window_size`
- `S_c = num_filt`
- `T = num_ofmap`

因此在 WS 中：

- batch 主要扩大 `T`

也就是说，权重常驻的模式不变，但时间维度被拉长了。

#### 2.6.3 IS 中 batch 的落点

IS 的定义是：

- `S_r = window_size`
- `S_c = num_ofmap`
- `T = num_filt`

因此在 IS 中：

- batch 主要扩大 `S_c`

也就是输入常驻，但列方向看到更多输出位置。

#### 2.6.4 这说明了什么

这说明 batch 对三种数据流的影响不是额外手写出来的，而是：

- 通过 batch-aware 的 workload 描述
- 自然传导到 `Sr / Sc / T`
- 再自然传导到 fold、prefetch、demand 和请求数

这正是本次实现的合理之处。

### 2.7 具体改动四：`single_layer_sim.py` 与报告路径改成 full-batch 语义

如果只修正地址和 demand，但报告仍然使用单样本工作量，那么最终公开结果仍然是错的。

因此这里的关键修改是：

- `num_compute` 现在基于 batch-aware 的 `get_layer_num_ofmap_px()`
- 利用率公式中的分子变成完整 batch 的总计算量
- SRAM / DRAM 带宽仍用真实请求数和真实周期计算

也就是说，报告公式本身没有被人为重写成“特殊 batch 版”，而是：

- 分子来自正确的 full-batch workload
- 分母来自真实 memory 模型跑出来的周期

这样得到的结果具有更强的一致性。

### 2.8 具体改动五：memory 路径和 CSV 输出主要做验证而非重构

当 compute demand 已经按 batch 扩展后，memory system 接收到的请求天然就是 full-batch 请求。

因此 memory 层的关键点不是重写算法，而是确认：

- 请求规模确实按 batch 扩大
- SRAM / DRAM 访问数与周期逻辑保持一致
- CSV 输出与这些真实计数器一致

此外还做了一个小但必要的稳定性修复：

- `double_buffered_scratchpad_mem.py` 中总周期提取改成 `int(np.max(...))`

这不是 batch 逻辑本身，但能避免验证过程中的不稳定或告警噪音。

### 2.9 具体改动六：测试与回归验证同步补齐

为了证明 batch 支持不是“形状看起来对了”，而是端到端语义正确，仓库中新增或更新了多类测试与 fixture。

#### 2.9.1 解析与派生指标测试

`test/test_topology_utils_batch.py` 主要验证：

- `Batch` / `Batch Size` 表头兼容
- 缺省 batch 默认值为 `1`
- 非法 batch 值会被拒绝
- mixed batch 拓扑会被拒绝
- helper API 能保留 batch 与 sparsity
- `get_layer_num_ofmap_px()`、`get_layer_mac_ops()`、`get_transformed_mnk_dimensions()`、`calc_spatio_temporal_params()` 已具备 batch 语义

#### 2.9.2 报告一致性测试

`test/test_report_batch.py` 主要验证：

- `single_layer_sim` 下 compute report 与真实计数器一致
- bandwidth report 与真实计数器一致
- detail report 与真实计数器一致
- `Batch = 1` 与更大 batch 的比例关系符合预期
- `simulator` 生成的 CSV 数值与 toy-layer 公式一致

#### 2.9.3 端到端 smoke 测试

`test/test_phase10_smoke.py` 主要验证：

- 多层 batch conv 可以完整跑通
- 多层 batch GEMM 可以完整跑通
- 仓库中已有 `Batch Size` 表头的真实 topology 文件可以被正常解析

#### 2.9.4 fixture 与旧回归脚本

同时新增了 batch 相关 topology fixture，并更新了原有回归脚本，使它们在 batch-aware 实现下仍能稳定比较 CSV 结果。

### 2.10 为什么这些修改在原理上是正确的

可以把整个 batch 支持的正确性总结为四步。

#### 第一步：topology 正确记录 batch

这保证了：

- batch 不会在一开始就丢失
- 总输出数、总 MAC 数、GEMM 映射维度都能正确计算

#### 第二步：operand matrix 正确展开 batch

这保证了：

- IFMAP 地址空间按样本隔离
- OFMAP 地址空间按样本隔离
- FILTER 地址保持共享

也就是说，workload 的“数据空间语义”是正确的。

#### 第三步：compute 和 memory 消费真实扩展后的 workload

这保证了：

- fold 数量自然变化
- demand matrix 自然变大
- SRAM / DRAM 请求自然变多
- stall 与周期是对完整 batch 的真实模拟结果

#### 第四步：报告基于真实周期和真实 full-batch 工作量

这保证了：

- 利用率不是假象
- 带宽不是事后乘法
- 公开 CSV 与底层真实计数器保持一致

因此，这套实现不是“补丁式 batch 支持”，而是把 batch 纳入了模拟器的 workload 模型本体。

### 2.11 当前实现仍然保留的约束

为了保证语义清晰和验证可控，当前实现保留了几个明确约束：

#### 2.11.1 不支持同一次 run 中不同层使用不同 batch

当前设计要求：

- 一个 topology 中所有层的 batch 必须一致

若发现 mixed per-layer batch，会在解析阶段直接报错拒绝。

#### 2.11.2 FILTER 仍然是共享空间

这是有意为之，不是遗漏。因为 batch 表示样本复制，而不是权重复制。

#### 2.11.3 本次重点是语义正确，不是全面技术债清理

也就是说：

- batch 支持已经完成
- 但仓库里其他历史代码风格问题、脚本基础设施问题，并不都属于这次工作的范围

### 2.12 第二部分总结

本次 batch 支持可以用一句话概括：

> 不是在原有单样本模拟器外面套一层 batch 乘法，而是把 batch 变成 topology、地址矩阵、计算请求和报告路径都共同理解的真实工作负载维度。

这也是为什么现在当拓扑文件中设置 `Batch > 1` 时，SCALE-Sim 能够做到：

- 正确解析 batch
- 正确分离不同样本的 IFMAP / OFMAP 地址空间
- 保持 filter 共享
- 让 OS / WS / IS 三种数据流自然看到更大的 workload
- 输出反映完整 batch 的周期、带宽和访问统计

---

## 汇报时可直接使用的结论性表述

如果需要在汇报末尾用一段话总结，可以直接使用下面这段表述：

> SCALE-Sim v3 的主流程可以理解为“输入文件解析 -> workload 统一表示 -> operand address matrix 生成 -> 数据流 demand 生成 -> scratchpad/DRAM 时序模拟 -> 报告输出”。本次 batch 支持的关键，不是对最终结果做一个比例缩放，而是把 batch 作为真实工作负载维度前移到 topology 和 operand matrix 层。这样一来，不同样本的地址空间、三种数据流下的 fold 行为、SRAM/DRAM 请求以及最终的报告统计，都会自然反映完整 batched workload。这使得 `Batch > 1` 的模拟结果在地址、周期和统计口径上都具备一致性。

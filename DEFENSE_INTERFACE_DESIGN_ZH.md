# Defense 最小可行接口设计说明

## 1. 目的

这份文档定义项目中所有 MIA 防御方法统一遵守的**最小可行接口**。

和攻击方法不同，防御方法的作用位置并不统一：

- 有的防御发生在**训练阶段**，直接改变模型训练方式
  - 例如：`AdvReg`、`RelaxLoss`、`DP-SGD`、`privacy distillation`、`adversarial training`
- 有的防御发生在**推理阶段**，不改模型参数，只改模型输出
  - 例如：`MemGuard`
- 有的防御本质上是**训练策略或数据策略**
  - 例如：`Data Augmentation`、`EarlyStop`、`Shuffle`
- 有的防御是**模型结构层面的替换或组合**
  - 例如：`ensemble`、`PAR-GAN`

因此，防御接口不能只设计成“输入模型，输出模型”这么简单，而应该支持：

1. 输入原始训练资源，输出防御后模型
2. 输入已有模型，输出防御后的推理器或输出变换器
3. 输入训练数据或训练配置，输出修改后的训练流程

这份接口设计的目标是：

- 不强迫所有防御方法内部实现一致
- 只要求它们对外的输入、输出和调用流程尽量统一
- 让后续新增分类模型、生成模型、推荐模型的防御方法时，都能接入同一套框架

---

## 2. 设计原则

- 所有防御方法统一接收一个 `DefenseInput`
- 所有防御方法统一返回一个 `DefenseOutput`
- 防御的主结果不是 Accuracy，而是**防御后对象**
- “防御后对象”可以是：
  - 新训练出的模型
  - 被包装后的预测器
  - 被修改后的训练数据或训练配置
- 如果提供了评估信息，再额外输出防御效果评估
- 对训练型防御保留 `fit`
- 对仅推理期防御保留 `infer`
- 对需要同时支持训练和部署的防御，统一通过 `run`

---

## 3. 防御方法的三类作用方式

为了避免接口混乱，建议先把防御方法按作用方式分成三类。

### 3.1 训练时防御

这类方法会改变模型训练过程。

典型例子：

- `AdvReg`
- `RelaxLoss`
- `HAMP`
- `DP-SGD`
- `privacy distillation`
- `adversarial training`
- `ensemble`

特点：

- 输入通常包括训练数据、训练标签、模型初始化器、训练配置
- 输出通常是防御后训练得到的模型

### 3.2 推理时防御

这类方法不改模型参数，而是修改模型的输出。

典型例子：

- `MemGuard`
- 部分输出扰动型、置信度修正型方法

特点：

- 输入通常包括一个已训练模型和待预测样本
- 输出通常是“被包装后的预测器”或“被修改后的输出结果”

### 3.3 数据/流程型防御

这类方法主要改变训练数据组织方式或训练终止方式。

典型例子：

- `Data Augmentation`
- `EarlyStop`
- `Shuffle`
- `Popularity Randomization`

特点：

- 输入可能是训练数据、采样器、增强器、训练调度策略
- 输出可能是修改后的训练数据、修改后的训练配置、或最终模型

---

## 4. 顶层接口

建议所有防御方法统一实现下面这个基类接口：

```python
class BaseDefense:
    def fit(self, defense_input: DefenseInput) -> "BaseDefense":
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        raise NotImplementedError

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput
    ) -> DefenseEvaluationResult:
        ...

    def run(self, defense_input: DefenseInput) -> DefenseOutput:
        self.fit(defense_input)
        output = self.infer(defense_input)
        if defense_input.eval_config is not None:
            output.evaluation = self.evaluate(output, defense_input)
        return output
```

### 4.1 四个方法的职责

- `fit`
  - 用于训练型防御
  - 例如训练防御后模型、训练防御模块、拟合输出扰动器参数
  - 对纯推理期防御，可以直接 `return self`

- `infer`
  - 必须实现
  - 返回防御后的主要产物
  - 这个“主要产物”可能是模型、预测器、预测结果或数据变换结果

- `evaluate`
  - 可选
  - 当提供评估配置时，计算防御效果
  - 防御评估既可能关注任务效用，也可能关注隐私风险

- `run`
  - 统一入口
  - 外部调用方应优先调用 `run(...)`

---

## 5. 输入接口：`DefenseInput`

### 5.1 定义

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class DefenseInput:
    target_model: Optional[Any] = None
    model_factory: Optional[Any] = None

    train_data: Optional[Any] = None
    train_labels: Optional[Any] = None
    val_data: Optional[Any] = None
    val_labels: Optional[Any] = None
    test_data: Optional[Any] = None
    test_labels: Optional[Any] = None

    samples: Optional[Any] = None
    labels: Optional[Any] = None

    auxiliary_data: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None
    defense_config: Dict[str, Any] = field(default_factory=dict)
    eval_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 6. `DefenseInput` 字段说明

### 6.1 `target_model`

已有模型对象，可选。

适用场景：

- 推理期防御
- 在已有模型基础上继续做防御微调
- 输出扰动型防御

例如：

- `MemGuard`
- 训练后再加一个输出修正器

### 6.2 `model_factory`

模型构造器，可选。

适用场景：

- 训练型防御需要从头训练模型
- 需要训练多模型结构时

例如：

- `AdvReg`
- `RelaxLoss`
- `DP-SGD`
- `ensemble`

建议：

- 训练型防御优先使用 `model_factory`
- 不要把模型初始化过程散落在每个 defense 文件里

### 6.3 `train_data`, `train_labels`

训练数据和训练标签，可选但对训练型防御通常必需。

适用场景：

- 从头训练防御后模型
- 训练攻击对手网络
- 训练蒸馏教师/学生模型
- 做数据增强、重采样、shuffle

### 6.4 `val_data`, `val_labels`

验证集，可选。

适用场景：

- `EarlyStop`
- 调整防御强度
- 比较 utility/privacy trade-off

### 6.5 `test_data`, `test_labels`

测试集，可选。

适用场景：

- 防御效果评估
- 任务效用评估

### 6.6 `samples`, `labels`

运行时输入样本和对应标签，可选。

适用场景：

- 推理期防御直接处理一批待预测样本
- 仅输出扰动型防御

这里的 `labels` 是任务标签，不是成员标签。

### 6.7 `auxiliary_data`

存放防御方法需要的额外资源。

例如：

- public data
- augmentation transform
- adversary model
- teacher model
- item popularity statistics
- recommendation candidates
- latent priors

### 6.8 `signals`

预计算的模型输出或中间信号，可选。

适用场景：

- 推理期输出防御直接消费 logits / probabilities
- 不想重复跑模型

可包含：

- `logits`
- `probabilities`
- `losses`
- `features`
- `recommendation_scores`
- `reconstruction_scores`

### 6.9 `defense_config`

防御方法自己的超参数配置。

例如：

- noise multiplier
- clipping norm
- regularization weight
- confidence target
- temperature
- augmentation policy
- stopping patience
- ensemble size

建议：

- 所有方法私有超参数都放这里
- 不要不断往 `DefenseInput` 顶层新增零散字段

### 6.10 `eval_config`

防御评估配置，可选。

这和攻击接口里直接用 `membership_labels` 不同。防御的评估往往是组合式的，可能同时包含：

- 任务效用评估
- 隐私风险评估
- 开销评估

建议在 `eval_config` 中显式提供：

- 要评估哪些指标
- 要调用哪些攻击方法验证防御强度
- 是否需要比较防御前后性能

### 6.11 `metadata`

运行附加信息，可选。

例如：

- dataset name
- model type
- number of classes
- defense family
- experiment tag

---

## 7. 输出接口：`DefenseOutput`

### 7.1 定义

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class DefenseOutput:
    defended_model: Optional[Any] = None
    protected_predictor: Optional[Any] = None
    protected_outputs: Optional[Any] = None
    transformed_data: Optional[Any] = None

    artifacts: Optional[Dict[str, Any]] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    evaluation: Optional["DefenseEvaluationResult"] = None
```

---

## 8. `DefenseOutput` 字段说明

### 8.1 `defended_model`

防御后的模型对象。

适用场景：

- 训练型防御
- 模型结构防御
- 蒸馏防御
- 差分隐私训练

这是很多防御方法最核心的输出。

### 8.2 `protected_predictor`

被包装后的预测器对象。

适用场景：

- 推理期防御
- 只修改预测逻辑，不修改底层模型权重

例如：

- `MemGuard`
- 输出后处理型 defense wrapper

### 8.3 `protected_outputs`

防御后的输出结果。

适用场景：

- 只想对一批给定样本做输出扰动
- 不打算长期保存 predictor wrapper

### 8.4 `transformed_data`

被防御方法变换后的数据。

适用场景：

- 数据增强
- 数据打乱
- 重采样
- recommendation 相关数据处理

### 8.5 `artifacts`

训练或防御过程中产出的可复用对象摘要。

例如：

- learned defense parameters
- clipping/noise config
- output perturbation parameters
- early stopping checkpoint path
- teacher-student mapping

### 8.6 `intermediate_outputs`

保存真正有价值的中间量。

例如：

- 原始 logits 和防御后 logits
- privacy-utility trade-off curve
- regularization loss
- confidence calibration values

### 8.7 `metadata`

补充信息。

例如：

- defense name
- defense family
- mode
- training epochs

---

## 9. 评估接口：`DefenseEvaluationResult`

### 9.1 定义

```python
@dataclass
class DefenseEvaluationResult:
    utility_metrics: Optional[Dict[str, float]] = None
    privacy_metrics: Optional[Dict[str, float]] = None
    efficiency_metrics: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None
```

### 9.2 为什么防御评估不能只看一个指标

防御和攻击不同。

攻击通常关心：

- AUROC
- Accuracy
- TPR@low FPR

而防御通常必须同时看：

1. **任务效用**
   - 分类准确率
   - 推荐效果
   - 生成质量
2. **隐私风险**
   - 防御后攻击 AUROC
   - 防御后攻击 Accuracy
   - 防御后攻击 TPR@low FPR
3. **计算/部署代价**
   - 训练时间
   - 推理时间
   - 显存/内存开销

所以建议统一评估结构分成三块：

- `utility_metrics`
- `privacy_metrics`
- `efficiency_metrics`

### 9.3 推荐默认评估内容

#### `utility_metrics`

分类模型可包含：

- `accuracy`
- `f1`
- `loss`

推荐模型可包含：

- `recall@k`
- `ndcg@k`

生成模型可包含：

- `reconstruction_loss`
- `fid`
- `is`

#### `privacy_metrics`

建议至少支持：

- `attack_accuracy`
- `attack_auroc`
- `attack_tpr_at_1pct_fpr`
- `attack_tpr_at_0_1pct_fpr`

#### `efficiency_metrics`

建议支持：

- `train_time`
- `inference_time`
- `num_parameters`
- `memory_usage`

---

## 10. 建议显式区分防御模式

建议每个防御类都声明自己的模式。

例如：

```python
class BaseDefense:
    name: str
    defense_family: str
    defense_mode: str
```

推荐 `defense_mode` 枚举值：

- `training_time`
- `inference_time`
- `data_processing`
- `hybrid`

例如：

- `DP-SGD`
  - `training_time`
- `MemGuard`
  - `inference_time`
- `Data Augmentation`
  - `data_processing`
- `AdvReg`
  - `hybrid`

这样外部调度器在调用时能更清楚地知道：

- 要不要提供 `model_factory`
- 要不要提供 `target_model`
- 要不要提供 `samples`
- 输出究竟应该重点看 `defended_model` 还是 `protected_outputs`

---

## 11. 建议增加能力声明字段

为了让统一调度器更容易做输入校验，建议每个防御类声明：

```python
class BaseDefense:
    supported_model_types: list[str]
    required_input_keys: list[str]
    optional_input_keys: list[str]
```

例如：

- `DP-SGD`
  - `supported_model_types = ["classifier", "diffusion", "gan"]`
  - `required_input_keys = ["model_factory", "train_data", "train_labels"]`

- `MemGuard`
  - `supported_model_types = ["classifier"]`
  - `required_input_keys = ["target_model", "samples"]`
  - `optional_input_keys = ["signals"]`

- `EarlyStop`
  - `required_input_keys = ["model_factory", "train_data", "train_labels", "val_data", "val_labels"]`

---

## 12. 调用流程规范

统一调用流程建议如下：

```python
defense = SomeDefense(...)
output = defense.run(defense_input)
```

内部流程通常为：

1. `fit(defense_input)`
2. `infer(defense_input)`
3. 如果提供 `eval_config`，执行 `evaluate(...)`
4. 返回 `DefenseOutput`

注意：

- 外部调用方不需要知道某个防御到底是训练型还是推理型
- 各防御类自己决定 `fit` 是真正训练，还是 no-op

---

## 13. 新防御方法接入规范

每新增一个防御方法，开发者至少需要完成下面几件事。

### 13.1 继承 `BaseDefense`

必须实现：

- `infer`

按需实现：

- `fit`
- `evaluate`

通常不用重写：

- `run`

### 13.2 写清楚它属于哪类防御

每个防御类都应该在类注释里明确：

- `defense_mode`
- 依赖哪些输入字段
- 主输出是什么

例如：

```python
class MemGuardDefense(BaseDefense):
    """
    defense_mode: inference_time
    Required:
        - target_model
        - samples or signals["probabilities"]
    Main output:
        - protected_predictor or protected_outputs
    """
```

### 13.3 明确主输出对象

开发时一定要先明确这个防御的核心结果是什么：

- 是模型？
- 是输出包装器？
- 是数据变换结果？

不要所有防御都硬塞成 `defended_model`。

### 13.4 评估分成“效用”和“隐私”两层

不要只输出：

- `accuracy`

更合理的是同时输出：

- 任务效用指标
- 防御后的攻击效果指标

---

## 14. 面向当前项目的最小可行接口建议

如果你们现在希望先尽快统一已有防御实现，建议先采用下面这个最小版本。

### 14.1 输入

```python
@dataclass
class DefenseInput:
    target_model: Optional[Any] = None
    model_factory: Optional[Any] = None
    train_data: Optional[Any] = None
    train_labels: Optional[Any] = None
    val_data: Optional[Any] = None
    val_labels: Optional[Any] = None
    samples: Optional[Any] = None
    labels: Optional[Any] = None
    signals: Optional[Dict[str, Any]] = None
    auxiliary_data: Optional[Dict[str, Any]] = None
    defense_config: Dict[str, Any] = field(default_factory=dict)
    eval_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 14.2 输出

```python
@dataclass
class DefenseOutput:
    defended_model: Optional[Any] = None
    protected_predictor: Optional[Any] = None
    protected_outputs: Optional[Any] = None
    transformed_data: Optional[Any] = None
    evaluation: Optional[DefenseEvaluationResult] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 14.3 方法接口

```python
class BaseDefense:
    def fit(self, defense_input: DefenseInput) -> "BaseDefense":
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        raise NotImplementedError

    def run(self, defense_input: DefenseInput) -> DefenseOutput:
        self.fit(defense_input)
        output = self.infer(defense_input)
        if defense_input.eval_config is not None:
            output.evaluation = self.evaluate(output, defense_input)
        return output

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput
    ) -> DefenseEvaluationResult:
        ...
```

这个版本已经能覆盖你们当前列出的绝大多数防御方法。

---

## 15. 我给出的最终建议

如果目标是“尽量适应不同目标模型和不同防御方法，并且后面真能工程化扩展”，我建议最终采用下面这套思路：

### 15.1 统一输入对象

输入不要只传：

- `model`
- `X`
- `y`

更合理的是结构化 `DefenseInput`，至少包含：

- `target_model`
- `model_factory`
- `train_data/train_labels`
- `val_data/val_labels`
- `samples/labels`
- `signals`
- `auxiliary_data`
- `defense_config`
- `eval_config`

### 15.2 统一输出对象

输出不要只返回“防御后准确率”。

更合理的是：

- `defended_model`
- `protected_predictor`
- `protected_outputs`
- `transformed_data`
- `evaluation`

### 15.3 统一调用入口

推荐统一为：

```python
fit(defense_input) -> self
infer(defense_input) -> DefenseOutput
evaluate(defense_output, defense_input) -> DefenseEvaluationResult
run(defense_input) -> DefenseOutput
```

### 15.4 统一区分两类核心防御

这个项目里最关键的区分是：

- **训练型防御**
  - 重点输出 `defended_model`
- **输出型防御**
  - 重点输出 `protected_predictor` 或 `protected_outputs`

这一步一定要在接口层面明确，不然后续实现会越来越乱。

---

## 16. 一句话总结

如果只问“Defense 的输入输出应该是什么”，我建议的最核心答案是：

- 输入应当是一个结构化的 `DefenseInput`，至少包含：已有模型或模型构造器、训练/验证数据、可选推理样本、可选预计算信号、辅助资源、方法配置和评估配置
- 输出应当是一个结构化的 `DefenseOutput`，核心返回是“防御后的对象”，这个对象可以是模型、预测器、输出结果或数据变换结果；若提供评估配置，则额外返回 `DefenseEvaluationResult`，其中同时包含任务效用指标和隐私风险指标

这个设计会比只定义一个简单的 `defense(model, X, y)` 更适合你们后续集成不同类型的 MIA 防御方法。

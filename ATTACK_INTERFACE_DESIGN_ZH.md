# Attack 最小可行接口设计说明

## 1. 目的

这份文档定义项目中所有 MIA 方法统一遵守的**最小可行接口**。

目标很简单：

- 不要求所有攻击方法内部实现一致
- 只要求它们对外的输入、输出、调用流程一致
- 让后续新增 `loss`、`LiRA`、`RMIA`、`QMIA`、`EncoderMI`、`SF-MIA` 等方法时，都能接入同一套流水线

这版接口刻意保持简洁，只解决当前最核心的问题：

1. 攻击方法应该接收什么输入
2. 攻击方法应该返回什么输出
3. 真实成员标签在接口里怎么放
4. 攻击分数和攻击评估指标怎么区分
5. 新开发者实现一个新攻击类时，具体应该怎么做

---

## 2. 设计原则

- 所有攻击方法统一接收一个 `AttackInput`
- 所有攻击方法统一返回一个 `AttackOutput`
- 攻击的主结果是**成员分数**，不是 Accuracy
- 如果提供了真实成员标签，再额外输出评估结果
- 对需要训练的攻击保留 `fit`
- 对不需要训练的攻击，`fit` 可以什么都不做

---

## 3. 顶层接口

所有攻击方法都应实现同一个基类接口：

```python
class BaseAttack:
    def fit(self, attack_input: AttackInput) -> "BaseAttack":
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        raise NotImplementedError

    def evaluate(
        self,
        attack_output: AttackOutput,
        attack_input: AttackInput
    ) -> EvaluationResult:
        ...

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output
```

### 3.1 四个方法的职责

- `fit`
  - 给需要训练或拟合的攻击使用
  - 例如 shadow-based、LiRA、QMIA、Attack-R
  - 对 `loss`、`confidence`、`entropy` 这类直接打分攻击，可以直接 `return self`

- `infer`
  - 必须实现
  - 输入攻击数据，输出每个样本的成员分数，以及可选的成员预测标签

- `evaluate`
  - 只有在 `membership_labels` 存在时才调用
  - 用统一方式计算 Accuracy、AUROC、TPR@low FPR 等指标

- `run`
  - 统一入口
  - 开发者和外部调用方应该优先调用 `run(...)`

---

## 4. 输入接口：`AttackInput`

### 4.1 定义

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class AttackInput:
    target_model: Optional[Any]
    samples: Any
    labels: Optional[Any] = None
    membership_labels: Optional[Any] = None
    signals: Optional[Dict[str, Any]] = None
    reference_data: Optional[Dict[str, Any]] = None
    shadow_data: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 5. `AttackInput` 字段说明

### 5.1 `target_model`

目标模型对象，可为 `None`。

常见情况：

- 如果攻击方法需要直接查询模型，就传入已加载好的 PyTorch 模型
- 如果攻击方法只依赖预计算信号，例如已经算好的 `losses`、`logits`、`similarities`，则可以不传

开发要求：

- 不要假设所有攻击都一定能直接访问 `target_model`
- 如果方法依赖模型推理，应在文档或类注释里写清楚

### 5.2 `samples`

待攻击样本，必填。

它是统一接口里最核心的字段，但不同任务下含义可以不同：

- 分类模型：`X`
- 推荐模型：用户、交互序列、user-item 对
- 生成模型：待测样本、prompt、latent code、重建输入
- 对比学习模型：原始样本或增广样本

开发要求：

- 攻击类必须明确自己期待的 `samples` 结构
- 不要把数据格式假设写死在基类里

### 5.3 `labels`

任务标签，可选。

注意：这里的 `labels` 不是成员标签，而是任务本身的标签。

例如：

- 分类任务里的类别标签 `y`
- 生成模型里某些条件标签
- 推荐任务里可选的监督信息

常见用途：

- 计算 per-sample loss
- 计算 correctness
- 计算 confidence / margin / entropy 的 label-dependent 版本

### 5.4 `membership_labels`

真实成员标签，可选。

推荐约定：

- `1` 表示 member
- `0` 表示 non-member

用途：

- **只用于评估攻击效果**
- 一般不作为攻击推理的必要输入

如果该字段不存在：

- 攻击依然应该能跑
- 但不输出监督评估指标

### 5.5 `signals`

预计算攻击信号，可选。

这是最重要的扩展字段之一。它允许不同攻击方法直接消费统一命名的中间量，而不是每次都重新访问模型。

常见可放入 `signals` 的内容：

- `logits`
- `probabilities`
- `losses`
- `correctness`
- `confidences`
- `entropies`
- `modified_entropies`
- `margins`
- `gradients`
- `features`
- `similarities`
- `reconstruction_errors`
- `density_scores`
- `likelihoods`
- `trajectory_scores`

开发要求：

- 优先复用 `signals` 中已有内容
- 如果方法内部计算了新的关键中间量，建议在输出的 `intermediate_outputs` 中返回

### 5.6 `reference_data`

参考型攻击使用的附加数据，可选。

适用方法：

- LiRA
- RMIA
- RAPID
- 各类参考分布、校准型攻击

可包含：

- reference model outputs
- member / non-member reference scores
- public calibration data
- population statistics

### 5.7 `shadow_data`

shadow-based 攻击使用的附加数据，可选。

适用方法：

- Shadow-based NN
- Attack-R
- 其他需要 shadow model 或 attack model 训练数据的方法

可包含：

- shadow train/test samples
- shadow logits / probabilities / losses
- shadow member / non-member labels
- 已训练好的 shadow models

### 5.8 `config`

攻击方法自己的超参数配置。

例如：

- threshold
- batch size
- top-k
- quantile level
- distance metric
- number of augmentations
- score normalization method

开发要求：

- 所有方法私有超参数都放这里
- 不要不断往 `AttackInput` 顶层新增零散字段

### 5.9 `metadata`

运行附加信息，可选。

例如：

- dataset name
- model name
- num classes
- sample ids
- split name
- experiment tag

这个字段主要用于日志、调试、保存结果，不建议承载方法的核心逻辑依赖。

---

## 6. 输出接口：`AttackOutput`

### 6.1 定义

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class AttackOutput:
    membership_scores: Any
    membership_preds: Optional[Any] = None
    evaluation: Optional["EvaluationResult"] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 7. `AttackOutput` 字段说明

### 7.1 `membership_scores`

这是所有攻击方法最核心、最统一的输出。

要求：

- 每个样本对应一个分数
- 分数越大，表示越可能是 member

这个方向约定必须全项目统一。

原因：

- 有的方法天然输出的是“越大越像 member”
- 有的方法天然输出的是“越小越像 member”，例如某些 loss
- 如果不统一方向，后续 AUROC、TPR@FPR、阈值判断都会混乱

因此开发要求是：

- 如果你的原始方法分数方向相反，必须在返回前转成统一方向

### 7.2 `membership_preds`

可选的二值成员预测结果。

推荐约定：

- `1` 表示 member
- `0` 表示 non-member

适用场景：

- 方法天然输出 hard label
- 或者方法在分数基础上还给出了明确阈值

如果方法只适合输出分数，这个字段可以为 `None`。

### 7.3 `evaluation`

仅当 `membership_labels` 存在时填写。

这里保存统一计算的评估指标，而不是攻击主结果。

也就是说：

- 攻击主结果是 `membership_scores`
- 评估指标是附加结果 `evaluation`

### 7.4 `intermediate_outputs`

保存中间结果，便于调试、复现和画图。

常见内容：

- per-sample loss
- entropy
- reference mean/std
- predicted quantiles
- similarity scores
- reconstruction error
- decision boundary distance

开发要求：

- 可以放真正有价值的中间量
- 不要把无法理解的大量临时变量全部塞进去

### 7.5 `metadata`

输出相关的补充信息。

例如：

- attack name
- threshold used
- score normalization mode
- runtime statistics

---

## 8. 评估接口：`EvaluationResult`

### 8.1 定义

```python
@dataclass
class EvaluationResult:
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    tpr_at_fpr: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None
```

### 8.2 默认要求

最小版本里，统一评估至少支持：

- `accuracy`
- `auroc`
- `tpr_at_fpr`

其中 `tpr_at_fpr` 推荐至少包含：

- `1%`
- `0.1%`

例如：

```python
{
    "1%": 0.43,
    "0.1%": 0.17,
}
```

如果后面需要扩展，可以继续在 `extra_metrics` 里加入：

- balanced accuracy
- precision
- recall
- f1
- average precision
- roc curve

---

## 9. 调用流程规范

统一调用流程如下：

```python
attack = SomeAttack(...)
output = attack.run(attack_input)
```

内部流程约定为：

1. `fit(attack_input)`
2. `infer(attack_input)`
3. 如果 `membership_labels` 存在，执行 `evaluate(...)`
4. 返回 `AttackOutput`

注意：

- 外部调用方不需要关心某个方法是否训练
- 各攻击类自己决定 `fit` 是真正训练，还是 no-op

---

## 10. 新攻击方法接入规范

每新增一个攻击方法，开发者至少需要完成下面几件事。

### 10.1 继承 `BaseAttack`

必须实现：

- `infer`

按需实现：

- `fit`

通常不用重写：

- `run`
- `evaluate`

如果某个方法评估方式明显不同，再考虑单独覆盖 `evaluate`。

### 10.2 写清楚依赖的输入字段

每个攻击类都应该在类注释或文档里说明：

- 是否依赖 `target_model`
- 是否依赖 `labels`
- 是否依赖 `signals`
- 是否依赖 `reference_data`
- 是否依赖 `shadow_data`

例如：

```python
class LossAttack(BaseAttack):
    """
    Required:
        - labels
        - signals["losses"] or target_model
    """
```

### 10.3 统一分数方向

无论原始论文怎么定义，最终都必须满足：

```python
higher score -> more likely member
```

这是硬约束。

### 10.4 统一成员标签编码

统一要求：

- `1` = member
- `0` = non-member

不要在不同攻击实现里混用相反编码。

### 10.5 优先返回分数，再考虑返回标签

开发顺序建议：

1. 先保证 `membership_scores` 正确
2. 再决定是否生成 `membership_preds`
3. 最后接统一评估

原因是：

- AUROC、TPR@low FPR 依赖分数
- 分数比硬标签更通用

---

## 11. 开发者实现模板

```python
class BaseAttack:
    def fit(self, attack_input: AttackInput) -> "BaseAttack":
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        raise NotImplementedError

    def evaluate(self, attack_output: AttackOutput, attack_input: AttackInput) -> EvaluationResult:
        # 统一评估逻辑
        ...

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


class LossAttack(BaseAttack):
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if attack_input.signals is not None and "losses" in attack_input.signals:
            losses = attack_input.signals["losses"]
        else:
            model = attack_input.target_model
            x = attack_input.samples
            y = attack_input.labels
            losses = compute_per_sample_loss(model, x, y)

        membership_scores = -losses

        return AttackOutput(
            membership_scores=membership_scores,
            intermediate_outputs={"losses": losses},
            metadata={"attack_name": "loss"}
        )
```

这个例子体现了最小接口的两个关键点：

- 优先复用 `signals`
- 如果原始信号方向不一致，要在输出前统一方向

---

## 12. 对当前项目的落地建议

当前项目接入新旧方法时，建议按下面方式理解输入：

- `target_model`
  - 能直接调用模型时使用

- `samples`
  - 一律表示“本次需要被攻击的对象”

- `labels`
  - 一律表示“任务标签”，不是成员标签

- `membership_labels`
  - 一律表示“成员真值”，只用于评估

- `signals`
  - 放统一命名的可复用攻击信号

- `reference_data`
  - 给 LiRA、RMIA、RAPID 这类方法

- `shadow_data`
  - 给 Shadow-based、Attack-R、QMIA 这类方法

这套约定已经足够支撑当前仓库的大多数攻击方法整合。

---

## 13. 最终结论

本项目的 `Attack` 最小可行接口统一为：

- 输入：`AttackInput`
- 输出：`AttackOutput`
- 调用：`run(...)`
- 可选训练：`fit(...)`
- 必须推理：`infer(...)`
- 有真实成员标签时统一评估：`evaluate(...)`

开发者实现新攻击方法时，最重要的三条要求是：

1. 主输出必须是 `membership_scores`
2. 分数方向必须统一为“越大越像 member”
3. `membership_labels` 只用于评估，不和任务标签 `labels` 混用

如果后续需要扩展复杂能力，可以在这个最小接口之上继续增加字段，但现阶段不建议把接口设计得过重。

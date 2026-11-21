# audio-dual-mono-detect

母带检查工具报双单声道后，反相素材被漏判、M/S 幅度还翻倍：相关用归一化点积但不减均值，直流偏置导致误判；静音段相关计算除零；双单声道只看相关，不看反相；M/S 用 (l+r)/2 与 (l-r)，未归一，S 幅度翻倍；反相修复改左声道，与预期相反；折叠损失用线性比而不是 dB；分窗相关不做重叠，边界漏检；长度不等不报错，截断；批量共用窗口缓冲，对之间串；threshold 超出 -1..1 不校验。phase/correlation.py 是核心，tests/test_correlation.py 15个红。要修的是：按上述 10 条逐一修复，并保证批量检测在任一对非法时整体回滚。

## 运行方式

```powershell
python -m pytest tests -q
```

初始状态必须恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error。当前 `phase/correlation.py` 的 8 个函数全部是 `raise NotImplementedError` 桩；修复实现即可让同一批测试转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 仅使用标准库（`math`）与 `pytest`；离线运行，不使用随机数，不调用 `sleep`。
- `tests/test_correlation.py` 不可修改；只修改 `phase/correlation.py`。
- 立体声统一以 `(left, right)` 两个 `list[float]` 传入；相关系数范围为 `-1..1`，容差 `1e-6`。
- 非法输入使用 `PhaseError`；`validate` 以问题列表形式报告数据问题。

# Ancient DNA Haplotype Phasing API

从含降解错配的分子读段（reads）联合复原一对**互补单倍型**，并把每条读段
**唯一归属**到两条同源染色体之一。不做逐位多数表决，避免把两条同源染色体的
证据拼接成不存在的序列。

- 8–18 个有序双等位位点，10–36 条唯一读段
- 每条读段覆盖一个**连续区间**，携带二值观测、逐位正整数错配代价、允许错配位点数
- 每组（每条同源染色体）至少 2 条读段
- 目标（依次）：① 最小化错配总代价；② 最小化单条读段最大错配数
- 交换两组视为同一解；稳定判定 **unique / ambiguous**
- 并列时返回按固定次序（最大错配数 → 单倍型字典序 → 归属向量字典序）排序的
  **前两个不同解**
- 无解或读段不连续时返回明确业务错误码

## 目录

```
app/phaser.py        核心求解器（枚举 + numpy 向量化，无外部优化器依赖）
app/main.py          FastAPI 应用：POST /api/phase、GET /health
tests/test_phaser.py 单元/性质测试，含小规模穷举 oracle 对拍
verify.py            一次性复核：等服务健康后做 API 冒烟并运行单元测试
Dockerfile           容器镜像
docker-compose.yml   带健康检查的 api 服务 + 一次性 verify 服务
```

## 本地运行

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 另一个终端：等服务健康后复核错配样例、歧义样例、错误用例和单元测试
python verify.py            # 默认 http://localhost:8000；退出码 0 即全部通过
```

## Docker / Compose

宿主机端口由 `API_PORT` 配置（容器内固定 8000，默认 8000）：

```bash
API_PORT=8080 docker compose up -d --build
docker compose ps                # health: healthy

# 一次性 verify（依赖 api 健康后才启动，自行退出，退出码汇总结果）
docker compose --profile verify run --rm verify; echo "exit=$?"
```

## API

### `POST /api/phase`

请求体：

```json
{
  "loci": 8,
  "reads": [
    {
      "positions": [0, 1, 2],
      "observations": "010",
      "mismatch_costs": [4, 1, 4],
      "max_mismatches": 3
    }
  ]
}
```

- `positions`：从 0 起、严格递增且连续（不允许缺口，否则 `READ_NOT_CONTIGUOUS`）
- `observations`：等长的 `0/1` 字符串
- `mismatch_costs`：逐位正整数
- `max_mismatches`：该读段允许的错配位点数

成功响应（节选）：

```json
{
  "status": "unique",
  "loci": 8,
  "num_reads": 10,
  "objective": {"total_mismatch_cost": 1, "max_mismatches_per_read": 1},
  "solutions": [
    {
      "solution_rank": 1,
      "haplotypes": {"group_0": "00000000", "group_1": "11111111"},
      "group_sizes": [5, 5],
      "total_mismatch_cost": 1,
      "max_mismatches_per_read": 1,
      "assignments": [
        {
          "read_id": 0, "group": 0,
          "positions": [0, 1, 2], "observations": "010",
          "mismatch_count": 1, "mismatch_cost": 1,
          "mismatches": [
            {"position": 1, "observed": "1", "expected": "0", "cost": 1}
          ],
          "max_mismatches_allowed": 3, "within_mismatch_limit": true
        }
      ]
    }
  ]
}
```

`status` 为 `ambiguous` 时 `solutions` 含排序最前的两个不同最优解。

错误响应（HTTP 422）：

```json
{"error": {"code": "READ_NOT_CONTIGUOUS", "message": "read[0]: ..."}}
```

| code | 含义 |
| --- | --- |
| `VALIDATION_ERROR` | 字段缺失/越界/类型错误 |
| `READ_NOT_CONTIGUOUS` | 读段覆盖区间存在缺口 |
| `DUPLICATE_READ` | 区间与观测完全相同的读段重复 |
| `MISMATCH_BUDGET_EXCEEDED` | 任意候选互补对下都有读段超出允许错配数 |
| `INFEASIBLE_GROUP_BALANCE` | 预算内无法保证两组各 ≥ 2 条读段 |
| `INFEASIBLE` | 预算与分组平衡联合无解 |

## 求解方法

枚举互补对的规范代表（令高位两个位点为 0，`2^(n-2) ≤ 2^16` 种）。给定单倍型
后，每条读段在**两侧**的错配数/代价独立向量化计算：某读段只要有一侧不超过其
错配预算即为预算可行（便宜侧超限、贵侧可行时允许付出更高代价）。各读段先放在
其最便宜可行侧（同价取 0 侧），再按分组约束（两组各 ≥ 2）选择代价差最小的
读段切换到另一侧，向量化求出每个候选的最小平衡总代价；随后在最小总代价不变的
前提下从 0 起扫描错配阈值 t（两侧须同时满足阈值与预算），得到最小的"单读段
最大错配数"。对进入前列的少数候选，用读段顺序上的后缀 DP（m ≤ 36）枚举最优
归属，返回字典序最小解并判定是否存在第二个不同最优解（同价切换、付费切换均
计入），最终确定唯一解与字典序第二解。

## 测试

```bash
python -m pytest tests -q     # 72 passed
```

其中 40+12 组随机样例与**穷举 oracle**（单倍型 × 全部分组）逐一核对最优代价、
最大错配数、唯一/歧义判定及第二解内容。

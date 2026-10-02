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
后，每条读段在两条同源染色体上的错配数/代价独立向量化计算；一条读段只要**任一
侧**错配数不超过预算即为可行——可行侧可能恰好是代价更高的一侧（例如廉价侧需 2
个错配而预算只允许 1，贵侧仅错配 1 次但代价为 100）。分组平衡（两组各 ≥ 2）可能
迫使某些读段付出这一额外代价，因此可行性与最小代价由一个按"0 组读段数"分阶段的
向量化最小代价 DP 精确求解，DP 单元以（总代价，单读段最大错配数）二元组字典序
取优，天然保留总成本优先、最大错配数次优的裁决顺序。候选筛出后，对胜出单倍型用
精确的标量前向/后向 DP 做饱和（上限 2）最优归属计数与字典序定秩枚举，确定唯一解
与字典序第二解；确实无预算内解时仍返回 `MISMATCH_BUDGET_EXCEEDED`。

## 测试

```bash
python -m pytest tests -q     # 83 passed
```

其中 40+12+12 组随机样例（含紧预算、倾斜代价，迫使廉价侧非法、只能选贵侧的情形）
与**穷举 oracle**（单倍型 × 全部分组）逐一核对最优代价、最大错配数、唯一/歧义判定
及前两个解；另有针对"仅贵侧可行、两侧均可行、等代价但错配数不同、仅靠贵侧才能
平衡、两侧均不可行"的确定性回归用例。

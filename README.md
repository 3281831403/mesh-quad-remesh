# mesh-quad-remesh

## 题面

把三角网格重划成四边形后，网格上到处是三角洞、UV 接缝还被切断：`build_adjacency` 拿有序顶点对当边键，一条边被三个面共享时邻接表被拆成两条、配对全乱；`pair_score` 只算共面度、不含合并后四边形的面积与边长比，细长四边形成片出现；`pair_triangles` 碰到第一个可配的邻居就锁死而不是全局取最高分，配对数明显偏少；`merge_to_quad` 输出的四边形绕序是顺时针，背面剔除后整片消失；同一函数里共享边两端的顶点顺序没对齐，三角带一旋转就拼出自交四边形；`fill_leftovers` 把没配上对的三角直接丢掉，覆盖率掉到 1 以下；`preserve_seams` 只比较顶点索引不比较 UV，缝两侧照样被合并；`validate_quads` 不检查自交，蝴蝶四边形一路放行；`remesh_batch` 跨网格复用邻接表，第二个网格的边键与第一个撞车；`pair_triangles` 的 threshold 按绝对值比较、不归一化，换个尺度整批配对数就变；quadremesh/pair.py 是核心，tests/test_pair.py 15个红。要修的是：`build_adjacency` 用 `(min(i, j), max(i, j))` 半边键并把非流形边的全部入射面收进同一个邻接项；`pair_score` 同时计入共面度与 `area / longest_edge ** 2` 的形状项；`pair_triangles` 全局贪心取最高分、平分按 `(i, j)` 升序；`merge_to_quad` 按源三角的逆时针方向输出 `(apex_a, u, apex_b, v)`，共享边只能当对角线；两个三角的角点顺序都要搜索旋转以对齐共享边；`fill_leftovers` 用退化四边形 `(a, b, c, a)` 补齐剩余三角；`preserve_seams` 用 UV 差（容差 1e-9）识别缝、缝边不配对；`validate_quads` 校验自交、面积、重复顶点与两半绕序一致性；`remesh_batch` 每个网格独立建邻接、不共享边键；threshold 归一化到 `[0, 1]` 后再比较。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态应恰好为 `15 failed`，且没有 collection error、import error 或 syntax error：

```text
15 failed in 0.05s
```

修复完成后应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定

- 只使用 Python 标准库 `math`、`collections` 与 `pytest`，离线运行，不使用随机数。
- 三角形记录固定为 `{"v": (i, j, k), "p": ((x, y, z), ...), "uv": ((u, v), ...)}`，`"p"` 与 `"uv"` 和 `"v"` 同角点顺序。
- 四边形是顶点下标的 4 元组，绕序逆时针；未配对三角存为退化四边形 `(a, b, c, a)`。
- 半边索引：无向边一律用 `(min(i, j), max(i, j))` 作键；非流形边（3 个及以上面共享）保留全部入射面，边界边保留 1 个入射面。
- 浮点容差 `1e-9`；`pair_score` 与 `threshold` 都归一化到 `[0, 1]`。
- 批量重网格时每个网格独立建邻接，不共享任何边键、配对或四边形状态。

## 不可修改

`tests/test_pair.py` 不可修改，其中 15 个测试是验收标准。修复只能发生在 `quadremesh/pair.py`，并保持现有公开函数签名、返回结构与 docstring 契约不变。

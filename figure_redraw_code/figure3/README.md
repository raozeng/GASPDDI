# Figure 3 重画代码

运行redraw_figure3.py，依赖NumPy、Pandas、Matplotlib。

仅修改B的绝对差图例名称和C最后一个区间标签；数据读取data中的原CSV副本，不重新计算归因、消融或性能。数据查询仍使用旧Configuration键，只有显示图例改名。

保留A的模型版本、全部标题和图例，不增加备注；0.31阈值、各曲线与增益保留。顶部总标题按已确认的Figure 3方案保留（Figure 2去除总标题的要求不扩展到此图）。

输出：../../figures/fig3_modality_and_operator_decomposition_v2.png，仅PNG，300 dpi。来源及原文件哈希见source_manifest.json。原代码、PNG/PDF及实验CSV均保留。

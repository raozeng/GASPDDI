# Figure 4 标题与分图编号修改代码

运行：`python redraw_figure4.py`。依赖Pillow、Matplotlib；不需要RDKit或模型推理。

本次只修改标题区域，直接复用原300 dpi PNG中的科学图形，避免重新计算或改变分子着色、贡献分数和曲线。去掉顶部总标题及其留白；底部标题改为D1 | ...、D2 | ...，药名独立放在第二行；右侧三组标题改为E1 | ...、E2 | ...、E3 | ...，保留关系及两种模型概率。

输出：`../../figures/fig1_case_studies_substructure_saliency_v2.png`，300 dpi。原PNG、PDF及绘图脚本均保留。脚本检查图形区域与原图逐像素一致；来源哈希、输出哈希和坐标保存在source_manifest.json。TeX仍使用原文件，未切换引用。D1/D2归属现有图注D，E1/E2/E3归属图注E。

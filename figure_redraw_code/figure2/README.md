# Figure 2 重画代码

- 运行 `redraw_figure2.py`；依赖NumPy、Matplotlib，无需RDKit或训练环境。
- 输出：`../../figures/fig5_why_3d_distillation_fails_diagnosis_v2.png`，300 dpi，仅PNG。
- A损失及C五折F1数据从原实验CSV读取后保存在plot_data.json。
- B保留原PDF中的曲线折线顶点，按原坐标轴刻度映射回数值后绘制；保留原图折线简化，不推测缺失顶点、不重算SVD。
- 来源及原文件校验见source_manifest.json。原脚本和PNG/PDF不修改。
- 修改：移除全图顶部标题；A标题及横轴去掉Fold 1；损失差阴影改紫色并增加图例边框；B明确为No-3D指纹表征。

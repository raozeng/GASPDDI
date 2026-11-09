# Figure 1 重画代码

本文件夹独立保存Figure 1 v2重画脚本及原图分子素材，原脚本和原图片均保留。

- 脚本：`redraw_figure1.py`
- 素材：`assets/`，从原PDF中提取的3幅分子图，未重绘化学结构。
- 输出：`../../figures/fig0_model_architecture_v2.png`（300 dpi，仅PNG）
- 来源校验与修改项：`source_manifest.json`
- 依赖：Python、NumPy、Matplotlib、Pillow；运行时不需要RDKit或PDF库。

当前环境运行命令（PowerShell）：

```powershell
& 'C:/Users/lenovo/miniconda3/python.exe' 'D:/GASPDDI/GASPDDI/paper/figure_redraw_code/figure1/redraw_figure1.py'
```

本轮仅修改最长SMILES、绝对差名称、F1增益单位及t-SNE表征标注，其余配色、模块结构、关系数量和示意数据保留。原论文仍引用原PDF，本次不修改TeX引用。

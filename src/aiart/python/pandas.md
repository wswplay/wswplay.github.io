---
title: Pandas api和用法、实例
---

# Pandas & Numpy

## 摘要-describe

- count：非空值的数量。
- mean：所有数值的平均值。
- std：标准差，是一个表示数据分散程度的指标。
- min：最小值。
- 25%：第一四分位数，表示所有数值中 25%的数据低于这个值。
- 50%：中位数，表示所有数值排序后的中间值，50%的数据低于这个值，50%的数据高于这个值。也被称为第二四分位数。
- 75%：第三四分位数，表示所有数值中 75%的数据低于这个值。
- max：最大值。

## 排序-sort

- axis 若axis=0(默认)或’index’，则按照指定列中数据大小排序；若axis=1或’columns’，则按照指定索引中数据大小排序。
- asceding=True 升序，False 为降序。
- by 指定列名(axis=0 或'index')或索引值(axis=1 或'columns')。
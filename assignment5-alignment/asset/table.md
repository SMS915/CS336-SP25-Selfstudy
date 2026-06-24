# Benchmark Table

下面放了两份同样的表：

1. `Original Snapshot`：原始版本，建议不要改。
2. `Working Copy`：工作副本，你可以直接在这里更新结果。

## Original Snapshot

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Base</th>
      <th rowspan="2">SFT</th>
      <th colspan="3">Reinforcement Learning (RL)</th>
      <th rowspan="2">Instruct</th>
      <th rowspan="2">▽ Gap</th>
    </tr>
    <tr>
      <th>GRPO</th>
      <th>w/o std</th>
      <th>Dr.GRPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><strong>Pass@1</strong></td>
      <td>GSM8K*</td>
      <td>20.9%</td>
      <td>55.2%<br><span style="color:#2ea043;">(+34.3%)</span></td>
      <td>80.4%</td>
      <td><strong>80.6%</strong><br><span style="color:#2ea043;">(+25.4%)</span></td>
      <td>79.15%</td>
      <td>85.14%</td>
      <td><span style="color:#f85149;">-4.54%</span></td>
    </tr>
    <tr>
      <td>MATH-500</td>
      <td>14.4%</td>
      <td>51.6%<br><span style="color:#2ea043;">(+37.2%)</span></td>
      <td>59.0%</td>
      <td><strong>60.6%</strong><br><span style="color:#2ea043;">(+9.0%)</span></td>
      <td>55.4%</td>
      <td>74.60%</td>
      <td><span style="color:#f85149;">-14.0%</span></td>
    </tr>
    <tr>
      <td>MATH-Test</td>
      <td>12.3%</td>
      <td>44.1%<br><span style="color:#2ea043;">(+31.8%)</span></td>
      <td>59.8%</td>
      <td><strong>61.2%</strong><br><span style="color:#2ea043;">(+17.1%)</span></td>
      <td>58.46%</td>
      <td>74.88%</td>
      <td><span style="color:#f85149;">-13.68%</span></td>
    </tr>
    <tr>
      <td>AMC12</td>
      <td>2.4%</td>
      <td>21.7%<br><span style="color:#2ea043;">(+19.3%)</span></td>
      <td><strong>36.1%</strong><br><span style="color:#2ea043;">(+14.4%)</span></td>
      <td>32.5%</td>
      <td>30.1%</td>
      <td>44.6%</td>
      <td><span style="color:#f85149;">-12.1%</span></td>
    </tr>
    <tr>
      <td>AIME 2024</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td><strong>13.33%</strong><br><span style="color:#2ea043;">(+13.3%)</span></td>
      <td>3.33%</td>
      <td>10.00%</td>
      <td>6.67%</td>
      <td><span style="color:#2ea043;">+6.67%</span></td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>3.33%</td>
      <td>0.00%<br><span style="color:#f85149;">(-3.33%)</span></td>
      <td>3.33%</td>
      <td>3.33%</td>
      <td><strong>6.67%</strong><br><span style="color:#2ea043;">(+6.67%)</span></td>
      <td>6.67%</td>
      <td><span style="color:#2ea043;">+0.0%</span></td>
    </tr>
    <tr>
      <td colspan="9">&nbsp;</td>
    </tr>
    <tr>
      <td rowspan="4"><strong>Pass@64</strong></td>
      <td>MATH-500</td>
      <td>91.6%</td>
      <td>90.6%<br><span style="color:#f85149;">(-1.0%)</span></td>
      <td>92.6%</td>
      <td><strong>93.6%</strong><br><span style="color:#2ea043;">(+3.0%)</span></td>
      <td>92.0%</td>
      <td>93.8%</td>
      <td><span style="color:#f85149;">-0.2%</span></td>
    </tr>
    <tr>
      <td>AMC12</td>
      <td>56.6%</td>
      <td>73.5%<br><span style="color:#2ea043;">(+16.9%)</span></td>
      <td><strong>75.9%</strong></td>
      <td><strong>75.9%</strong><br><span style="color:#2ea043;">(+2.4%)</span></td>
      <td>74.7%</td>
      <td>84.3%</td>
      <td><span style="color:#f85149;">-8.4%</span></td>
    </tr>
    <tr>
      <td>AIME 2024</td>
      <td>13.3%</td>
      <td>26.7%<br><span style="color:#2ea043;">(+13.4%)</span></td>
      <td>30.0%</td>
      <td>40.0%</td>
      <td><strong>40.0%</strong><br><span style="color:#2ea043;">(+13.3%)</span></td>
      <td>46.67%</td>
      <td><span style="color:#f85149;">-6.67%</span></td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>3.3%</td>
      <td>23.3%<br><span style="color:#2ea043;">(+20.0%)</span></td>
      <td>30.0%</td>
      <td>30.0%</td>
      <td><strong>36.7%</strong><br><span style="color:#2ea043;">(+13.4%)</span></td>
      <td>40.00%</td>
      <td><span style="color:#f85149;">-3.3%</span></td>
    </tr>
  </tbody>
</table>

---

## Working Copy

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Base</th>
      <th rowspan="2">SFT</th>
      <th colspan="3">Reinforcement Learning (RL)</th>
      <th rowspan="2">Instruct</th>
      <th rowspan="2">▽ Gap</th>
    </tr>
    <tr>
      <th>GRPO</th>
      <th>w/o std</th>
      <th>Dr.GRPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><strong>Pass@1</strong></td>
      <td>GSM8K*</td>
      <td>27.2%</td>
      <td>55.2%<br><span style="color:#2ea043;">(+34.3%)</span></td>
      <td>80.4%</td>
      <td><strong>80.6%</strong><br><span style="color:#2ea043;">(+25.4%)</span></td>
      <td>79.15%</td>
      <td>85.14%</td>
      <td><span style="color:#f85149;">-4.54%</span></td>
    </tr>
    <tr>
      <td>MATH-500</td>
      <td>23.6%</td>
      <td>51.6%<br><span style="color:#2ea043;">(+28.0%)</span></td>
      <td>59.0%</td>
      <td><strong>60.6%</strong><br><span style="color:#2ea043;">(+9.0%)</span></td>
      <td>55.4%</td>
      <td>74.60%</td>
      <td><span style="color:#f85149;">-14.0%</span></td>
    </tr>
    <tr>
      <td>MATH-Test</td>
      <td>24.5%</td>
      <td>44.1%<br><span style="color:#2ea043;">(+19.6%)</span></td>
      <td>59.8%</td>
      <td><strong>61.2%</strong><br><span style="color:#2ea043;">(+17.1%)</span></td>
      <td>58.46%</td>
      <td>74.88%</td>
      <td><span style="color:#f85149;">-13.68%</span></td>
    </tr>
    <tr>
      <td>AMC12</td>
      <td>13.3%</td>
      <td>21.7%<br><span style="color:#2ea043;">(+8.4%)</span></td>
      <td><strong>36.1%</strong><br><span style="color:#2ea043;">(+14.4%)</span></td>
      <td>32.5%</td>
      <td>30.1%</td>
      <td>44.6%</td>
      <td><span style="color:#f85149;">-12.1%</span></td>
    </tr>
    <tr>
      <td>AIME 2024</td>
      <td>0.00%</td>
      <td>0.00%</td>
      <td><strong>13.33%</strong><br><span style="color:#2ea043;">(+13.3%)</span></td>
      <td>3.33%</td>
      <td>10.00%</td>
      <td>6.67%</td>
      <td><span style="color:#2ea043;">+6.67%</span></td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>3.33%</td>
      <td>0.00%<br><span style="color:#f85149;">(-3.33%)</span></td>
      <td>3.33%</td>
      <td>3.33%</td>
      <td><strong>6.67%</strong><br><span style="color:#2ea043;">(+6.67%)</span></td>
      <td>6.67%</td>
      <td><span style="color:#2ea043;">+0.0%</span></td>
    </tr>
    <tr>
      <td colspan="9">&nbsp;</td>
    </tr>
    <tr>
      <td rowspan="4"><strong>Pass@64</strong></td>
      <td>MATH-500</td>
      <td>91.6%</td>
      <td>90.6%<br><span style="color:#f85149;">(-1.0%)</span></td>
      <td>92.6%</td>
      <td><strong>93.6%</strong><br><span style="color:#2ea043;">(+3.0%)</span></td>
      <td>92.0%</td>
      <td>93.8%</td>
      <td><span style="color:#f85149;">-0.2%</span></td>
    </tr>
    <tr>
      <td>AMC12</td>
      <td>56.6%</td>
      <td>73.5%<br><span style="color:#2ea043;">(+16.9%)</span></td>
      <td><strong>75.9%</strong></td>
      <td><strong>75.9%</strong><br><span style="color:#2ea043;">(+2.4%)</span></td>
      <td>74.7%</td>
      <td>84.3%</td>
      <td><span style="color:#f85149;">-8.4%</span></td>
    </tr>
    <tr>
      <td>AIME 2024</td>
      <td>20.0%</td>
      <td>26.7%<br><span style="color:#2ea043;">(+6.7%)</span></td>
      <td>30.0%</td>
      <td>40.0%</td>
      <td><strong>40.0%</strong><br><span style="color:#2ea043;">(+13.3%)</span></td>
      <td>46.67%</td>
      <td><span style="color:#f85149;">-6.67%</span></td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>16.7%</td>
      <td>23.3%<br><span style="color:#2ea043;">(+6.6%)</span></td>
      <td>30.0%</td>
      <td>30.0%</td>
      <td><strong>36.7%</strong><br><span style="color:#2ea043;">(+13.4%)</span></td>
      <td>40.00%</td>
      <td><span style="color:#f85149;">-3.3%</span></td>
    </tr>
  </tbody>
</table>

# Updates on Performance of DMENet
*All results are measured in matlab.*

* Accuracy (Figure 2 in the main paper)
    * **0.8783**

* Table 2 in the main paper (the last column)
    | | ... | Ours |
    | :-------------: | :-------------: | :-------------: |
    | MSE | ... | **0.009** |
    | MAE | ... | **0.077** |
    * We compute standard deviation maps (*i.e.*, `(out * 15) - 1) / 2`) and clipped them to have values between 0 and 3.275 ([`evaluation/RTF/quantitative_RTF.m`](/evaluation/RTF/quantitative_RTF.m#L45-L53)).

* Table 4 in the supplementary material (the last column)
    | datasets | ... | <i>DMENet<sub>BDCS</sub></i> |
    | :------: | :------: | :------: |
    | SYNDOF | ... | **0.013** / **0.084** |
    | RTF | ... | **0.009** / **0.077** |

* Table 5 in the supplementary material (the last row)
    | | SYNDOF<br/>MSE / MAE | RTF<br/>MSE / MAE | CUHK<br/><i>acc</i> / <i>mAP</i> |
    | :----: | :----: | :----: | :----: |
    | ... | ... | ... | ... |
    | <i>DMENet<sub>BDCS</sub></i> | 0.013 / 0.084 | **0.009** / **0.077** | 0.878 / **0.987** |

* Table 6 in the supplementary material (the last column)
    | Image # | ... | Ours |
    | :-------------: | :-------------: | :-------------: |
    | 01 | ... | **0.0643** |
    | 02 | ... | **0.0406** |
    | 03 | ... | **0.0863** |
    | 04 | ... | **0.0408** |
    | 05 | ... | **0.0335** |
    | 06 | ... | 0.0756 |
    | 07 | ... | 0.1129 |
    | 08 | ... | 0.1695 |
    | 09 | ... | **0.0427** |
    | 10 | ... | **0.0771** |
    | 11 | ... | **0.044** |
    | 12 | ... | 0.1347 |
    | 13 | ... | **0.0817** |
    | 14 | ... | **0.1123** |
    | 15 | ... | **0.0838** |
    | 16 | ... | **0.0881** |
    | 17 | ... | **0.0675** |
    | 18 | ... | **0.0786** |
    | 19 | ... | **0.0744** |
    | 20 | ... | **0.0841** |
    | 21 | ... | **0.0397** |
    | 22 | ... | **0.0535** |
    | Avg. MSE | ... | **0.0093** |
    | Avg. MAE | ... | **0.0767** |
    | Avg. MSE (s=1.0) | ... | **0.0207** |
    | Avg. MAE (s=1.0) | ... | **0.1006** |
    | Avg. MSE (s=1.6) | ... | 0.0491 |
    | Avg. MAE (s=1.6) | ... | 0.1579 |

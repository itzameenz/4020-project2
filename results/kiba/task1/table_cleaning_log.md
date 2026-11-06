|   step | operation                              | rows in           | rows out      | change              | note                                                   |
|-------:|:---------------------------------------|:------------------|:--------------|:--------------------|:-------------------------------------------------------|
|      1 | Matrix → long format                   | 483,419 positions | 118,254 pairs | 365,165 NaN dropped | NaN = not measured (never set to 0)                    |
|      2 | Non-finite scores                      | 118,254           | 118,254       | 0                   | no ±inf present                                        |
|      3 | Range / input validity                 | 118,254           | 118,254       | 0                   | scores ≥ 0; sequences standard; 21 salt SMILES flagged |
|      4 | Merge stereoisomers (canonical SMILES) | 118,254           | 117,657       | 597 merged          | 473 pairs averaged                                     |
|      5 | Abnormal-value flags                   | 117,657           | 117,657       | 0 removed           | 29,877 rows carry ≥ 1 flag                             |
| check | value | verdict |
|:--|:--|:--|
| rollouts complete | 233 / 233 | pass |
| taught-class zeros (collision + offroad) not above the baseline's | 7 against 7 | pass |
| baseline-clean scenes turning into a collision / offroad zero not more than zeros removed | new 1 against removed (collision / offroad) 1 | pass |

| run | n | mean | score 1 | zeros | collision | offroad | corridor | slow |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| new | 233 | 0.9391 | 180 | 9 | 1 | 6 | 2 | 44 |
| baseline run | 233 | 0.9411 | 186 | 10 | 2 | 5 | 3 | 37 |

Removed zeros: a04628cdd3f25947 (corridor), 39cbed62434a528b (collision)
New zeros: 6d08dce7cfaa5035 (offroad)

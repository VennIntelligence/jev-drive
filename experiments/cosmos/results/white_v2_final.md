flag: is_white_any

## washed-out (descriptive, contrast to own background < 1/2 of CARLA's; fires on relighting too)

| cls           |   G4b |   edgeB |
|:--------------|------:|--------:|
| car           | 0.303 |   0.045 |
| dynamic       | 0.216 |   0.18  |
| fence         | 0     |   0.011 |
| pedestrian    | 0.599 |   0.216 |
| pole          | 0.065 |   0.03  |
| static        | 0.498 |   0.033 |
| traffic light | 0     |   0     |
| traffic sign  | 0.295 |   0.545 |

## White units by variant and class (share of object-frames >= 300 px)

| cls           |   ('units', 'G4b') |   ('units', 'edgeB') |   ('white', 'G4b') |   ('white', 'edgeB') |
|:--------------|-------------------:|---------------------:|-------------------:|---------------------:|
| car           |               5930 |                 5930 |              0     |                0     |
| dynamic       |                827 |                  827 |              0     |                0     |
| fence         |                556 |                  556 |              0     |                0     |
| pedestrian    |                903 |                  903 |              0     |                0.122 |
| pole          |               8919 |                 8919 |              0.047 |                0.073 |
| static        |               2632 |                 2632 |              0.004 |                0.018 |
| traffic light |               1922 |                 1922 |              0     |                0.01  |
| traffic sign  |                 88 |                   88 |              0.045 |                0     |

## by depth_bin

| depth_bin   |   G4b |   edgeB |
|:------------|------:|--------:|
| <10 m       | 0.001 |   0.12  |
| 10-20       | 0.077 |   0.069 |
| 20-40       | 0.009 |   0.024 |
| >40         | 0     |   0.014 |

## by size_bin

| size_bin   |   G4b |   edgeB |
|:-----------|------:|--------:|
| <1k px     | 0.029 |   0.045 |
| 1-3k       | 0.029 |   0.02  |
| 3-10k      | 0     |   0.045 |
| >10k       | 0     |   0.052 |

## by frame index

| t_bin   |   G4b |   edgeB |
|:--------|------:|--------:|
| 0-15    | 0.017 |   0.038 |
| 16-45   | 0.023 |   0.042 |
| 46-77   | 0.019 |   0.032 |
| 78-92   | 0.019 |   0.043 |

## edge variants: by control edge density inside the object

| edge_bin         |   G4b |   edgeB |
|:-----------------|------:|--------:|
| (0.00251, 0.118] | 0     |   0.057 |
| (0.118, 0.223]   | 0.015 |   0.023 |
| (0.223, 0.346]   | 0.042 |   0.011 |
| (0.346, 0.81]    | 0.003 |   0.03  |

## where (pair, member, class)

|                                                 |   units |   first_t |   last_t |   px_med |
|:------------------------------------------------|--------:|----------:|---------:|---------:|
| ('G4b', '24206-s0', 'minus', 'pole')            |      43 |        19 |       61 |    719   |
| ('G4b', '24206-s0', 'plus', 'pole')             |       8 |        19 |       31 |    425   |
| ('G4b', '24224-s0', 'plus', 'pole')             |       1 |         5 |        5 |   5005   |
| ('G4b', '24294-s0', 'minus', 'static')          |       5 |        27 |       36 |    431   |
| ('G4b', '24294-s0', 'plus', 'static')           |       5 |        27 |       36 |    431   |
| ('G4b', '25863-s0', 'minus', 'pole')            |      87 |         0 |       92 |    703   |
| ('G4b', '25863-s0', 'plus', 'pole')             |      93 |         0 |       92 |    703   |
| ('G4b', '27297-s0', 'minus', 'traffic sign')    |       2 |        87 |       91 |    398.5 |
| ('G4b', '27297-s0', 'plus', 'traffic sign')     |       2 |        87 |       91 |    398.5 |
| ('G4b', '27515-s0', 'minus', 'pole')            |      93 |         0 |       92 |   1672   |
| ('G4b', '27515-s0', 'plus', 'pole')             |      93 |         0 |       92 |   1672   |
| ('edgeB', '24206-s0', 'minus', 'pole')          |     142 |         1 |       92 |    903.5 |
| ('edgeB', '24206-s0', 'plus', 'pole')           |     108 |         0 |       88 |   1021   |
| ('edgeB', '24211-s0', 'minus', 'pole')          |      39 |        15 |       81 |   7019   |
| ('edgeB', '24211-s0', 'plus', 'pole')           |      50 |        15 |       90 |   6605.5 |
| ('edgeB', '24211-s0', 'plus', 'static')         |      15 |        63 |       91 |   6466   |
| ('edgeB', '24224-s0', 'minus', 'pole')          |      55 |         0 |       92 |   6784   |
| ('edgeB', '24224-s0', 'plus', 'pole')           |      26 |         0 |       88 |   7885.5 |
| ('edgeB', '24294-s0', 'minus', 'pole')          |      32 |         0 |       80 |  13393.5 |
| ('edgeB', '24294-s0', 'minus', 'static')        |      16 |        27 |       42 |    443   |
| ('edgeB', '24294-s0', 'plus', 'pole')           |       2 |        16 |       82 |   4532   |
| ('edgeB', '24294-s0', 'plus', 'static')         |      16 |        27 |       42 |    443   |
| ('edgeB', '25863-s0', 'minus', 'pole')          |      87 |         0 |       92 |    703   |
| ('edgeB', '25863-s0', 'minus', 'traffic light') |       1 |         0 |        0 |   1756   |
| ('edgeB', '25863-s0', 'plus', 'pole')           |      93 |         0 |       92 |    703   |
| ('edgeB', '27297-s0', 'minus', 'pole')          |       4 |        27 |       74 |    425.5 |
| ('edgeB', '27297-s0', 'plus', 'pole')           |      10 |        27 |       74 |    706   |
| ('edgeB', '27515-s0', 'plus', 'pole')           |       1 |         0 |        0 |   1672   |
| ('edgeB', '27582-s0', 'minus', 'traffic light') |      13 |         0 |       12 |   1157   |
| ('edgeB', '27582-s0', 'plus', 'pedestrian')     |     110 |         0 |       92 |   7168   |
| ('edgeB', '27582-s0', 'plus', 'traffic light')  |       5 |         0 |        4 |   2773   |

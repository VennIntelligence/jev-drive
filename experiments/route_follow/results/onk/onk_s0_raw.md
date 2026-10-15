# ON-K pilot: held 452 / 18 decisions (452 paired scenes), base RSG-L18-s0

| arm (on-k, ho-w) | pressure x k13 | tag | zeros 452 | d zeros vs RSG-L18-s0 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | GT (m) | dev ADE |
|:--|--:|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|
| (13, 3) | 1.00 | RSG-L18-s0 | 26 | - | 21 | - | 4 / 17 / 6 | 0.8809 | - | 1.135 | 0.5284 |
| (26, 3) | 2.24 | RSGk26-L18-s0 | 22 | -4.0 [-9.6, +1.2] | 18 | -3.0 [-8.8, +2.7] | 7 / 11 / 5 | 0.8910 | +0.0101 [-0.0008, +0.0231] | 1.094 | 0.5298 |
| (51, 3) | 5.76 | RSGk51-L18-s0 | 25 | -1.0 [-6.4, +4.1] | 20 | -1.0 [-7.1, +5.9] | 7 / 13 / 5 | 0.8890 | +0.0081 [-0.0044, +0.0215] | 1.184 | 0.5355 |
| (51, 1) | 1.92 | RSGk51w1-L18-s0 | 23 | -3.0 [-7.7, +1.1] | 17 | -4.0 [-8.6, +0.0] | 6 / 11 / 6 | 0.8939 | +0.0130 [+0.0037, +0.0240] | 1.182 | 0.5298 |
| (13, 3) s1 | 1.00 | RSG-L18-s1 | 26 | +0.0 [-4.7, +4.9] | 19 | -2.0 [-7.0, +2.6] | 6 / 13 / 8 | 0.8830 | +0.0021 [-0.0097, +0.0137] | 1.144 | 0.5270 |

Lines (prereg): candidate = zeros CI high < 0 or score CI low > 0 vs RSG-L18-s0; stop (no seed 1) = zeros CI low > +5.

- k26 (RSGk26-L18-s0): zeros -4.0 [-9.6, +1.2], score +0.0101 [-0.0008, +0.0231] -> **no gain**; seed 1 per the stop line
- k51 (RSGk51-L18-s0): zeros -1.0 [-6.4, +4.1], score +0.0081 [-0.0044, +0.0215] -> **no gain**; seed 1 per the stop line
- k51w1 (RSGk51w1-L18-s0): zeros -3.0 [-7.7, +1.1], score +0.0130 [+0.0037, +0.0240] -> **CANDIDATE**; seed 1 per the stop line

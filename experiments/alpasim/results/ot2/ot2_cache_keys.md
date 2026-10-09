# `ot2` off-track row caches (+-1.5 m, +-5 deg), kept for lane OT3

Written 2026-10-09 by lane OT2 (`experiments/alpasim/scripts/ap2_ot.py prep --amp a15`, i.e. `ot_rows.py` prep with DY 1.5 m, DPSI 5 deg, YMAX 3.0 m, VMIN 3 m/s; rng `default_rng([1, shard])`). Independent of the hinge. Box paths under `$DATA_DIR/runs/op_parity/cache/` (`$DATA_DIR` = /root/autodl-tmp/ujs): `<dir>/tab.npz`, `<dir>@warp/front.npy`, `<dir>@warp/teacher.npz`; the key of each file is in `<file>.key` (jevdrive.cache). 18 GB in total. Read them with `ot_rows.OT = "ot2"` / `ap2_ot.use_amp("a15")`.

| dir | rows | tab.npz key | front.npy key | teacher.npz key |
|:--|--:|:--|:--|:--|
| ot2_navtrain_full.s0of12 | 5716 | 30f7917f45ec0d2705740d12 | 869a7de31da123dfc317157e | 7f6387ba72e8ed923292442b |
| ot2_navtrain_full.s1of12 | 5826 | 7fea1aa4a6f8aa75ed501287 | f5056833099f8683eca5416f | 15542c84e3cea1c0acb41bd3 |
| ot2_navtrain_full.s2of12 | 5783 | 160997c6371b1797da8f656b | 743cbe2b97c23dff81eda46c | b9e1b3ff6c4831b28fb61b31 |
| ot2_navtrain_full.s3of12 | 5829 | 054a9abb3ae2f5ba78b1f336 | 6859ea70cf6064dadc5c601a | 32c820ea354b711276d9cd12 |
| ot2_navtrain_full.s4of12 | 5825 | 0557dfc06e991550a9b11c79 | e36b8ebef69178767d916527 | 0d0c9656e49aaf6fb107bf86 |
| ot2_navtrain_full.s5of12 | 5808 | ce6b8f6eea6e135603d4a3d9 | 265b9531171189762c749de5 | 2e0ff300c6d54c0ce5d7ea9a |
| ot2_navtrain_full.s6of12 | 5782 | 8ae2a11444322442d9f0214c | 6726231e0ba0e95259e06c38 | 8de206e6aca17a6120a46ac7 |
| ot2_navtrain_full.s7of12 | 5807 | 9a7550afa4dfb88ff6b8287c | 2eee153b74fb8402c7820740 | e7b3fa268e70746b3b1bc314 |
| ot2_navtrain_full.s8of12 | 5859 | 1a24cbc20adca83ea4e22142 | 724395560b8b026f6744f003 | ef1542e6a3abfa5370a13956 |
| ot2_navtrain_full.s9of12 | 5726 | 642ce2090d819cd74fcb682a | 735ee1438410fae7341fa6a1 | 1c657f1192bb2b7c5563d299 |
| ot2_navtrain_full.s10of12 | 5821 | bd89efb8ba3f26ae068689d5 | efec911f1bbed2d522cbac68 | f53b134bb71ca3c9323201ea |
| ot2_navtrain_full.s11of12 | 5790 | 61a9924b1f02448e63d51b81 | b47706eb303319008e86f4d1 | 6746ea3b76a033cf6e429921 |

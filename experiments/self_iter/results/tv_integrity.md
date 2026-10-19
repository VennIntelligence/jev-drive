# TV-AUDIT integrity (trainval CAM_F0, 1310 log pickles)

- archives: 200 trainval camera `.done` markers (0..199, missing numbers none); 35 stale `.part.d` chunk dirs of finished archives left in `_archives/`
- logs: 1310 pickles, 1310 blob dirs; dirs without a pickle 0, pickles without a dir 0
- frames 723019; CAM_F0 referenced and missing 0, empty 0, bad JPEG markers 0; full decode 29536 (every 25-th) bad 0
- CAM_F0 files on disk 723019 (148.7 GB referenced), orphans (on disk, not in a pickle) 0
- incomplete logs: 0 []

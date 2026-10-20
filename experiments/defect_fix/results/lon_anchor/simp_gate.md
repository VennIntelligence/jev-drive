# SIMP offline gate (supplement 2): FAIL

Human path (tab fut) as the plan: region active 7.62 % (line <= 2.00 %), clear active 3.94 % (line <= 2.00 %).

| plans | term | all rows | all active | nuPlan rows | nuPlan active | PAI rows | PAI active |
|:--|:--|--:|--:|--:|--:|--:|--:|
| human | region | 87012 | 7.62 % | 87012 | 7.62 % | 0 | 0.00 % |
| human | region_road | 87012 | 7.18 % | 87012 | 7.18 % | 0 | 0.00 % |
| human | region_onc | 34918 | 1.21 % | 34918 | 1.21 % | 0 | 0.00 % |
| human | clear | 171054 | 3.94 % | 87012 | 3.76 % | 84042 | 4.12 % |
| human | clear_lead | 171054 | 2.22 % | 87012 | 2.08 % | 84042 | 2.36 % |
| human | old_road | 87012 | 11.69 % | 87012 | 11.69 % | 0 | 0.00 % |
| human | old_onc | 34918 | 1.21 % | 34918 | 1.21 % | 0 | 0.00 % |
| human | old_agent | 171054 | 2.21 % | 87012 | 2.07 % | 84042 | 2.35 % |
| human | old_side | 171054 | 2.41 % | 87012 | 2.31 % | 84042 | 2.52 % |
| stored | region | 87012 | 10.85 % | 87012 | 10.85 % | 0 | 0.00 % |
| stored | region_road | 87012 | 9.90 % | 87012 | 9.90 % | 0 | 0.00 % |
| stored | region_onc | 34918 | 2.76 % | 34918 | 2.76 % | 0 | 0.00 % |
| stored | clear | 171054 | 9.82 % | 87012 | 10.17 % | 84042 | 9.47 % |
| stored | clear_lead | 171054 | 7.41 % | 87012 | 7.71 % | 84042 | 7.10 % |
| stored | old_road | 87012 | 13.52 % | 87012 | 13.52 % | 0 | 0.00 % |
| stored | old_onc | 34918 | 2.76 % | 34918 | 2.76 % | 0 | 0.00 % |
| stored | old_agent | 171054 | 7.39 % | 87012 | 7.68 % | 84042 | 7.09 % |
| stored | old_side | 171054 | 4.03 % | 87012 | 4.29 % | 84042 | 3.76 % |

active = max per-step violation > 0.05 m; human = tab fut (the imitation target from the visited state); stored = nuPlan rows' served plans, cp rows' NTV0-L18-s0 model plans; clear_lead = the lead part (its along-heading gradient may only shorten the plan): on the stored plans this is the brake-gradient trigger rate

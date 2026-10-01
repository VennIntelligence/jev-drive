# TCP behavior figures

Full controlled route, including startup, stopping, faults and post-contact behavior. No smoothing or warmup removal. Time origin is per attempt; equal displayed times do not imply equal world/NPC states. Numerical frame CSVs are in the linked analysis edition recorded by manifest.json.

Within each route, both arms share the same vertical scale for each behavior metric. Behavior panels show truth longitudinal speed/model desired, longitudinal/lateral acceleration, primary world-acceleration-derived jerk projected into current body axes, and selected steer. Missing samples stay gaps. Steering differences around1e-9 can arise from float32 CARLA control storage, not algorithm changes; raw differences remain in analysis.

World panels are stacked with shared tight limits and equal distance scale. World axes are reordered when needed to put route extent horizontally; axis labels disclose the order, without anisotropic stretching. World panels use actual actor position and navigation reference, with raw model points sampled every2s. Orange assumes actor origin, purple assumes GNSS offset−1.4m. Both are anchored on contemporaneous truth orientation for visualization only. Neither validates the training physical origin or sensor pose, and neither is a supervised model-execution error metric. Predictions are finite2s horizons, never extended. Full raw predictions and truth remain in source logs.

A/B official historical speed differences are not PI or harness gains: both present arms remove the official low-speed throttle cap. Six-case behavior acceptance is separate from an individual plot looking smoother; this report does not establish a new default or leaderboard gain.

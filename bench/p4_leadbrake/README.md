# p4_leadbrake

The stock side of scensim roadmap 094 P4: the lead car brakes ahead, written by hand in Isaac Lab
3.0's manager-based style, as a user without scensim would write it. It registers
`P4-LeadBrake-v0` through Isaac Lab's `isaaclab.tasks` entry point, so Isaac Lab's own train CLI
finds it.

    pip install --no-deps --no-build-isolation -e bench/p4_leadbrake      # into Isaac Lab's python
    python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
        --task P4-LeadBrake-v0 --num_envs 1024 --max_iterations 300 --seed 1
    python bench/p4_leadbrake/tools/evaluate.py --checkpoint logs/rsl_rl/p4_leadbrake/<run>/model_299.pt

## The task

- **Road**: a flat ground (friction 1.0, dry), one straight 3.5 m lane on each env's +x axis.
- **Ego, the policy**: a Tesla Model 3 as an Isaac Lab articulation (`car.py`): a chassis box,
  four corners on prismatic suspension, both front wheels steered to Ackermann angles, rear-wheel
  drive by torque. Its numbers are copied from sim_vehicle_spec's
  `generated/tesla_model3/articulation_carla-0.9.16.yaml` (physics_hash `47a1043b...`);
  `tests/test_car.py` holds them to the file when `P4_ARTICULATION_YAML` names it.
- **Lead**: a kinematic box with the ego's footprint, cruising at `lead_speed`, braking at a
  constant `decel` to a stop when its rear passes `onset`, then holding.
- **Per reset**: `lead_speed` U[8, 14] m/s, `headway` U[1, 3] s (the ego starts at the lead's
  speed), `decel` U[3, 7] m/s^2, `onset` U[40, 120] m.
- **Actions** (2, in [-1, 1]): longitudinal (+ drive torque on the rear wheels up to 3000 N m,
  - brake on every wheel up to 3000 N m, as the spin joints' implicit damping) and steering
  (x 0.7 rad, the bicycle angle).
- **Observations** (privileged state): speed, yaw rate, lateral offset, heading error, the gap from
  the ego's front to the lead's rear, the closing speed, the lead's acceleration, the last action.
- **Reward**, CaRL's form: route progress / 300 m x m_speed x m_lane, where
  m_speed = 1 - min(1, excess over 15 m/s / 15) and m_lane = 1 - min(1, |offset| / 1.75 m); no
  reward on a collision; and -1 on a terminal infraction (collision, lane exit). The last term
  is not CaRL's: its own cost of ending early, the progress given up, is nothing here, since the
  lead always stops before the route's end. Run 1 (seed 1, 1024 envs, 300 iterations, pure
  CaRL terms): the crash rate rose to 72 %. Run 2 (the same, with the term): no crashes from
  iteration 75, and 256/256 safe stops at the first bar (0/256 at the 2 to 8 m bar: it stopped 14.0 to 15.4 m back).
  And +1 at time-out for a stop in the bar's window (at rest, a gap of 2 to 8 m behind the
  stopped lead), not CaRL's either: progress alone tells a 3 m stop from a 15 m one by 0.04 of
  the return, too little for PPO to settle in the window. Seed 1 at the 2 to 8 m bar without
  it (1024 envs, 1500 iterations): no pass, 0.77 at best (iteration 1300), 0.00 at the end, its
  checkpoints swinging between stops 10 to 21 m back and crashes. Isaac Lab multiplies every
  term by the step's dt, so the weights are relative: a whole route's progress, the -1 and the
  +1 are each worth the same.
- **Terminations**: collision (gap <= 0), lane exit (|offset| > 3 m), the route's end (300 m),
  time-out (30 s).
- **Success**: the route's end, or at time-out a safe stop behind the stopped lead: at rest, a gap
  of 2 to 8 m. The pass bar (`tools/evaluate.py`): >= 95 % of 256 conditions drawn with seed
  20261122. The 8 m bound was set after the smoke runs, before the seed runs: run 2's policy
  stopped 13.9 to 15.4 m back (min, median), which the first bar (a gap of at least 2 m) passed
  256/256. A stop that far back is timid, so it fails now.
- **PPO**: rsl_rl, [256, 128, 64] ELU actor and critic with observation normalisation, 24 steps per
  env, 5 epochs, 4 mini-batches, lr 1e-3 adaptive, gamma 0.99, lambda 0.95 (`agent.py`).

## Known differences and observations

- Solver iterations are Isaac Lab's default (`ArticulationRootPropertiesCfg` with none set),
  as engine B has them. The suspension's reported position converges with them, while its force
  is the design load at every setting (PhysX's projected joint forces: 3860 and 4379 N per front
  and rear corner, 1680 kg x g in all). At rest q is 0.0047 to 0.0054 m at the default and 0.019
  to 0.022 m at 8 position iterations, where the file's sag puts it at 0. Run 2 trained at 8.
- The brake is a different law from engine B's (Coulomb with stiction, the torque that stops the
  wheel within a step, clipped to the brake torque).
- A brake torque computed each step from the wheel's speed chatters (3000 N m swings the 1.37 kg m^2
  wheel 18 rad/s per 120 Hz step) and does not stop the car; the implicit damping does: 10.4 m/s
  to rest in 5.6 m (mu g).
- Training episodes start with the lead one step ahead of the record's headway. In Isaac Lab
  3.0's `ManagerBasedRLEnv.step`, `_reset_idx` (here `reset_episode`, the lead at
  `FRONT + h v`) runs before `command_manager.compute`. So `LeadCommand._update_command`
  advances every env in the same step, the just-reset ones included. An episode reset in
  training starts at a gap of `h v + v dt` (0.4 to 0.7 m more), a headway of h + 0.05 s
  (+5 % at h = 1, +1.7 % at h = 3), and the onset test one step early. Only `env.reset()`'s
  first episodes start at h. `tools/evaluate.py` scores exactly those, one per condition, so
  the pass bar is judged at the record's h, and only the training distribution differs. Ours
  starts every episode at h. This is kept as trained for P4 (found by scensim 139, 2026-11-22).
  A rerun from scratch would place the lead at `FRONT + h v - v dt`, so the first update lands
  at h.

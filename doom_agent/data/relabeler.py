"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Hindsight instruction relabeling.
relabeler.py
"""

from collections import Counter
from pathlib import Path
import pickle


# Button positions in actions_table entries.  Order matches available_buttons
# in scenarios/basic.cfg: [MOVE_LEFT, MOVE_RIGHT, ATTACK].
MOVE_LEFT, MOVE_RIGHT, ATTACK = 0, 1, 2

# basic.wad pays +101 for the kill, -1/tic living, -5 per missed shot.
# With frame_repeat=4 a kill step nets ~+97 and the worst possible non-kill
# step nets ~-9 so anything above this threshold is a kill.
KILL_REWARD_THRESHOLD = 50.0

# Movement runs shorter than this many decisions (4 tics each) are jitter
# from the sticky random policy, not purposeful movement.
MIN_RUN_LENGTH = 3


class Relabeler():
    """
    Hindsight instruction relabeling.

    The scripted policy never intends to do anything.  Whatever it actually
    accomplishes is a perfect demonstation of *some* instruction.  This module
    scans raw trajectories for achieved outcomes (kills, movement runs, firing)
    and retroactively assigns the natural-language instruction each segment
    demonstrates.  A trajectory that failed at killing is still a successful
    demonstration of "move to the left".
    """
    def __init__(self, config=None):
        self.config = config
        self.artifacts_path = self.config['artifacts_path']
        self.trajectories_path = Path(self.artifacts_path) / "trajectories"
        self.relabled_path = Path(self.artifacts_path) / "relabeled"
        self.stats = Counter()

    def init(self):
        print("Relabeler!")
        self.relabled_path.mkdir(parents=True, exist_ok=True)

    def _buttons(self, step, actions_table):
        '''
        Decode a recorded action index back into its button states.
        '''
        return actions_table[step["action"]]

    def _make_segment(self, steps, start, end, instruction):
        '''
        Copy steps[start:end] and fill in the subgoal field.  This is the
        hindsight step: the instruction is assigned retroactively based on
        what the agent achieved, producing the (frame, instruction, hud, action)
        tuples behavior cloning trains on.

        {**step, ...} makes a shallow copy -- the frame array itself is shared,
        not duplicated in memory.  Segments stay contiguous because the
        sliding-window buffer needs consecutive steps under a single
        instruction.
        '''
        return {
            "instruction": instruction,
            "steps": [{**s, "subgoal": instruction}
                      for s in steps[start:end]]
        }

    def _label_kill(self, steps, actions_table):
        '''
        Rule 1: if any steps reward exceeds the kill threshold, the whole
        episode up to and including that step demonstrates "kill the monster".
        Everything the agent did before the kill (aiming, repositioning) is part
        of how the kill was achieved.
        '''
        for i, step in enumerate(steps):
            if step["reward"] > KILL_REWARD_THRESHOLD:
                return [self._make_segment(steps, 0, i+1, "kill the monster")]

        return []

    def _label_movement(self, steps, actions_table):
        '''
        Rule 2: contiguous runs where exactly one movement button is held
        (left XOR right) demonstrate directional movement.  Runs shorter than
        MIN_RUN_LENGTH are dropped as policy jitter.
        '''
        def direction(step):
            b = self._buttons(step, actions_table)
            if b[MOVE_LEFT] and not b[MOVE_RIGHT]:
                return "move to the left"
            if b[MOVE_RIGHT] and not b[MOVE_LEFT]:
                return "move to the right"
            return None

        segments = []
        run_label, run_start = None, 0
        # The append None flushes the final run without duplicating the
        # emit logic after the loop.
        for i, step in enumerate(steps + [None]):
            label = direction(step) if step is not None else None
            if label == run_label:
                continue
            if run_label is not None and i - run_start >= MIN_RUN_LENGTH:
                segments.append(
                    self._make_segment(steps, run_start, i, run_label))
            run_label, run_start = label, i
        return segments

    def _label_firing(self, steps, actions_table):
        '''
        Rule 3: contiguous runs with ATTACK held demonstrate "fire the weapon".
        Firing is an instantaneous outcome so no minimum run length applies.
        '''
        segments = []
        run_start = None
        for i, step in enumerate(steps + [None]):
            firing = (step is not None
                      and self._buttons(step,actions_table)[ATTACK])
            if firing and run_start is None:
                run_start = i
            elif not firing and run_start is not None:
                segments.append(
                    self._make_segment(steps, run_start, i, "fire the weapon")
                )
                run_start = None
        return segments

    def relabel_trajectory(self, trajectory):
        '''
        Segments from different rules may overlap -- the same step can
        appear under "kill the monster" and "move to the left".  HER does
        exactly this: one transition relabeled under multiple goals is
        legitimate goal-conditioned data augmentation.
        '''
        steps = trajectory["steps"]
        table = trajectory["actions_table"]
        segments = []
        segments += self._label_kill(steps, table)
        segments += self._label_movement(steps, table)
        segments += self._label_firing(steps, table)
        return segments

    def print_stats(self):
        '''
        The instruction distribution matters for behavior cloning: a badly
        skewed dataset teaches the policy one instruction and ignores the rest.
        '''
        print("\nInstruction distribution (labeled steps):")
        total = sum(self.stats.values())
        for instruction, count in self.stats.most_common():
            print(f" {instruction:>20} {count:>6} ({100.0*count/total:.1f}%)")
        print(f"  {'total':<20} {total:>6}")

    def relabel(self):
        '''
        Relabel every collected trajectory and save one relabeled .pkl
        per episode (save filename, artifacts/relabeled/ directory).
        '''
        paths = sorted(self.trajectories_path.glob("*.pkl"))
        if not paths:
            print(f"No trajectories found in {self.trajectories_path} -- "
                  "run collection first (make collect_data).")
            return

        for path in paths:
            with open(path, "rb") as f:
                trajectory = pickle.load(f)

            segments = self.relabel_trajectory(trajectory)
            for seg in segments:
                self.stats[seg["instruction"]] += len(seg["steps"])

            relabeled = {
                "scenario": trajectory["scenario"],
                "actions_table": trajectory["actions_table"],
                "frame_repeat": trajectory["frame_repeat"],
                "segments": segments,
            }
            output = self.relabled_path / path.name
            with open(output, "wb") as f:
                pickle.dump(relabeled, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Relabeled {path.name}: {len(trajectory['steps'])} raw "
                  f"steps -> {len(segments)} segments")

        self.print_stats()

    def inspect(self):
        '''
        Load every relabeled file back from disk and print a summary to prove
        the data round-trips and each step carries its instruction.
        '''
        for path in sorted(self.relabled_path.glob("*.pkl")):
            with open(path, "rb") as f:
                r = pickle.load(f)
            summary = ", ".join(
                f"{s['instruction']}[{len(s['steps'])}]"
                for s in r["segments"])
            print(f"{path.name}: {summary}")


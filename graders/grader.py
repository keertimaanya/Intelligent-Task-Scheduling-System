"""
grader.py - Evaluates an agent's scheduling performance.

Implements the 4 metrics designed in the system:
- M1: Weighted Completion Rate (WCR)
- M2: Priority-Weighted Earliness (PWE) 
- M3: Efficiency Ratio (ER)
- M4: Action Accuracy (AA)

Usage:
    grader = Grader(level_name)
    score = grader.evaluate(env, invalid_action_count)
"""

from typing import Dict, Any

class Grader:
    def __init__(self, level: str):
        self.level = level.upper()
        
        # Difficulty-specific parameters
        if self.level == "EASY":
            self.best_reward = 50.0
            self.worst_reward = -30.0
            self.weights = {"wcr": 0.40, "pwe": 0.10, "er": 0.30, "aa": 0.20}
        elif self.level == "MEDIUM":
            self.best_reward = 110.0
            self.worst_reward = -90.0
            self.weights = {"wcr": 0.35, "pwe": 0.15, "er": 0.35, "aa": 0.15}
        elif self.level == "HARD":
            self.best_reward = 120.0
            self.worst_reward = -140.0
            self.weights = {"wcr": 0.30, "pwe": 0.20, "er": 0.40, "aa": 0.10}
        else:
            raise ValueError(f"Unknown level: {level}")

    def evaluate(self, env, invalid_actions: int) -> Dict[str, float]:
        """
        Computes WCR, PWE, ER, AA, and the final combined score.
        All sub-metrics and the final score are clamped to [0, 1].
        
        Args:
            env: The SchedulerEnv instance at the end of the episode.
            invalid_actions: Counter of how many invalid step calls the agent made.
        """
        
        # 1. WCR & PWE logic
        total_priority = 0.0
        completed_on_time_priority = 0.0
        
        pwe_numerator = 0.0
        pwe_denominator = 0.0

        for task in env._tasks.values():
            pri = task.priority
            total_priority += pri
            
            # Status: 2 is COMPLETED
            if task.status == 2 and task.finish_time <= task.deadline:
                completed_on_time_priority += pri
                
                # Earliness (PWE) logic
                slack = max(0, task.deadline - task.finish_time)
                max_slack = max(0, task.deadline - task.duration)
                
                if max_slack == 0:
                    ratio = 1.0 # If zero slack existed by design, they got it perfectly on time
                else:
                    ratio = slack / max_slack
                    
                pwe_numerator += (pri * ratio)
                pwe_denominator += pri

        wcr = completed_on_time_priority / total_priority if total_priority > 0 else 0.0
        pwe = pwe_numerator / pwe_denominator if pwe_denominator > 0 else 0.0

        # 3. Efficiency Ratio (ER)
        # Clamped to [0, 1] linearly interpolated between worst and best reward.
        actual_reward = env._total_reward
        raw_er = (actual_reward - self.worst_reward) / (self.best_reward - self.worst_reward)
        er = max(0.0, min(1.0, raw_er))

        # 4. Action Accuracy (AA)
        total_steps = env._step_count
        if total_steps == 0:
            aa = 1.0
        else:
            aa = max(0.0, (total_steps - invalid_actions) / total_steps)
            
        # Overall Score
        overall = (
            wcr * self.weights["wcr"] +
            pwe * self.weights["pwe"] +
            er * self.weights["er"] +
            aa * self.weights["aa"]
        )
        
        return {
            "Level": self.level,
            "WCR": round(wcr, 3),
            "PWE": round(pwe, 3),
            "ER": round(er, 3),
            "AA": round(aa, 3),
            "Overall": round(overall, 3),
            "Raw_Reward": actual_reward
        }

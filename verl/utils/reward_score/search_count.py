# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Search count reward function.

Reward = min(num_searches, max_searches) / max_searches

This rewards the model for issuing search queries, capped at a maximum.
"""

import re
import random


def count_searches(solution_str: str) -> int:
    """Count the number of <search>...</search> tags in the solution string.

    Args:
        solution_str: The full solution text including all turns

    Returns:
        Number of search queries issued
    """
    search_pattern = r'<search>.*?</search>'
    matches = re.findall(search_pattern, solution_str, re.DOTALL)
    return len(matches)


def compute_score_search_count(solution_str: str, ground_truth=None, max_searches: int = 10, **kwargs) -> float:
    """Compute reward based on search count.

    The reward is normalized: reward = min(num_searches, max_searches) / max_searches

    Args:
        solution_str: The full solution text
        ground_truth: Not used, kept for API compatibility
        max_searches: Maximum number of searches to reward (cap)
        **kwargs: Additional arguments (ignored)

    Returns:
        Normalized reward in [0, 1]
    """
    num_searches = count_searches(solution_str)
    capped_searches = min(num_searches, max_searches)
    reward = capped_searches / max_searches

    # Debug printing (1 in 64 chance)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Search count reward:")
        print(f"  num_searches: {num_searches}")
        print(f"  capped_searches: {capped_searches}")
        print(f"  reward: {reward:.4f}")
        print(f"  solution_str (first 500 chars): {solution_str[:500]}")

    return reward


def compute_score_search_count_from_stats(valid_search_stats: int, max_searches: int = 10) -> float:
    """Compute reward based on pre-computed search stats.

    This is more efficient when valid_search_stats is already tracked.

    Args:
        valid_search_stats: Number of valid searches performed
        max_searches: Maximum number of searches to reward (cap)

    Returns:
        Normalized reward in [0, 1]
    """
    capped_searches = min(valid_search_stats, max_searches)
    return capped_searches / max_searches

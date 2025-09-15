# FILE: ceaf_system/logprob_analyzer.py
# PURPOSE: To translate raw log probability data into a meaningful confidence score.

import numpy as np
from typing import Dict, Any


def calculate_token_confidence_from_top_logprobs(token_logprob_obj: Dict[str, Any]) -> float:
    """
    Calculates the DeepConf confidence score for a single token based on the
    authors' provided Python code.

    This is the NEGATIVE of the AVERAGE LOG PROBABILITY of all top-k candidate tokens.

    Args:
        token_logprob_obj: A single item from the 'content' list of a streaming
                           logprobs response. It must contain the 'top_logprobs' key.
                           Example: {'token': ' a', 'logprob': -0.1,
                                     'top_logprobs': [{'token': ' a', 'logprob': -0.1},
                                                      {'token': ' the', 'logprob': -2.5}]}

    Returns:
        The confidence score for that token position. Higher is more confident.
    """
    top_logprobs = token_logprob_obj.get('top_logprobs', [])

    if not top_logprobs:
        return 20.0  # Return a high, safe confidence score if no data

    # Extract all the log probability values from the list of top candidates
    logprob_values = [lp['logprob'] for lp in top_logprobs if 'logprob' in lp]

    if not logprob_values:
        return 20.0

    # Calculate the average of these log probabilities
    mean_logprob = np.mean(logprob_values)

    # The confidence score is the negative of this average
    confidence = -mean_logprob

    return float(confidence)
import math
import logging
from typing import List, Union, Dict, Any
from wimbd_helper import WimbdHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_ppmi(n_both: int, n1: int, n2: int, n_total: int) -> float:
    """
    Calculate Positive Pointwise Mutual Information (PPMI).
    
    Formula: PPMI(x,y) = max(0, log2( P(x,y) / (P(x)P(y)) ))
    Which simplifies to: max(0, log2( (n_both * n_total) / (n1 * n2) ))
    """
    if n_both == 0 or n1 == 0 or n2 == 0:
        return 0.0
    
    # Calculate PMI
    pmi = math.log2((n_both * n_total) / (n1 * n2))
    
    # Return PPMI (Positive PMI)
    return max(0.0, pmi)

def co_occure(str1: str, str2: str, indices: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate co-occurrence statistics for two strings across specified indices.
    
    Args:
        str1: The first string/term.
        str2: The second string/term.
        indices: A single index name or a list of index names.
        
    Returns:
        A dictionary containing:
        - co_occurrence_count: Documents containing both strings.
        - count_str1: Documents containing str1.
        - count_str2: Documents containing str2.
        - total_documents: Total documents in the target indices.
        - ppmi: Positive Pointwise Mutual Information.
    """
    helper = WimbdHelper()
    
    # Convert list of indices to comma-separated string for WimbdSearch
    if isinstance(indices, list):
        index_query = ",".join(indices)
        target_list = indices
    else:
        index_query = indices
        target_list = [indices]
        
    # 1. Get total document count for the specified indices
    all_stats = helper.get_index_stats()
    n_total = 0
    for idx in target_list:
        if idx in all_stats:
            # The 'docs.count' in Elasticsearch is a string, convert to int
            n_total += int(all_stats[idx].get('docs.count', 0))
        else:
            logger.warning(f"Index {idx} not found on server.")

    if n_total == 0:
        logger.error("Total document count is 0. Cannot calculate PPMI.")
        return {}

    # 2. Get co-occurrence count (AND logic)
    n_both = helper.count_phrases(index_query, [str1, str2], all_phrases=True)
    
    # 3. Get sole appearances (total appearances for each)
    n1 = helper.count_phrases(index_query, [str1], all_phrases=True)
    n2 = helper.count_phrases(index_query, [str2], all_phrases=True)
    
    # 4. Calculate PPMI
    ppmi_score = calculate_ppmi(n_both, n1, n2, n_total)
    
    return {
        "str1": str1,
        "str2": str2,
        "indices": index_query,
        "co_occurrence_count": n_both,
        "count_str1": n1,
        "count_str2": n2,
        "total_documents": n_total,
        "ppmi": ppmi_score
    }

if __name__ == "__main__":
    # Example usage:
    result = co_occure("science", "data", "mc4_de")
    print(result)
    
    # Example with multiple indices:
    # result = co_occure("AI", "future", ["mc4_de", "mc4_fr"])
    # print(result)
    pass

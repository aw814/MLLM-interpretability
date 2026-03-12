from typing import List, Union, Optional, Dict, Any, Generator
from elasticsearch import Elasticsearch
import wimbd.es as wes

class WimbdHelper:
    """
    A helper class to abstract Elasticsearch operations using the wimbd library.
    
    This class provides a clean interface for counting and retrieving documents 
    across the mC4 and other custom indices.
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        request_timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize the helper with the Elasticsearch server URL.
        
        Args:
            es_url: The URL of the Elasticsearch server.
        """
        self.es_url = es_url
        self.es = Elasticsearch(
            self.es_url,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_on_timeout=True,
        )

    def count_phrases(self, 
                      index: str, 
                      phrases: Union[str, List[str]], 
                      all_phrases: bool = True,
                      is_regexp: bool = False,
                      slop: int = 0) -> int:
        """
        Count how many documents contain the given phrases.
        
        Args:
            index: Name of the index (or comma-separated list of indices).
            phrases: A single string or a list of strings to search for.
            all_phrases: If True, uses AND logic (must contain all). 
                         If False, uses OR logic (contains any).
            is_regexp: Whether to treat phrases as regular expressions.
            slop: Position allowance between terms in a phrase.
            
        Returns:
            The total document count.
        """
        return wes.count_documents_containing_phrases(
            index=index,
            phrases=phrases,
            all_phrases=all_phrases,
            is_regexp=is_regexp,
            slop=slop,
            es=self.es
        )

    def get_documents(self, 
                      index: str, 
                      phrases: Union[str, List[str]], 
                      num_documents: int = 10,
                      all_phrases: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Retrieve documents containing the given phrases.
        
        Args:
            index: Name of the index.
            phrases: Phrases to search for.
            num_documents: Maximum number of documents to return.
            all_phrases: AND vs OR logic.
            
        Yields:
            Elasticsearch document hits.
        """
        return wes.get_documents_containing_phrases(
            index=index,
            phrases=phrases,
            all_phrases=all_phrases,
            num_documents=num_documents,
            es=self.es
        )

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all open indices on the server.
        
        Returns:
            A dictionary containing index names and their document counts.
        """
        return wes.get_indices(es=self.es)

    def list_mc4_indices(self) -> List[str]:
        """
        Helper to list only the mC4 related indices.
        """
        stats = self.get_index_stats()
        return [name for name in stats.keys() if name.startswith("mc4_")]

# Example usage (commented out):
if __name__ == "__main__":
    # Initialize helper
    helper = WimbdHelper()
    
    # 1. Check counts in German mC4
    # count = helper.count_phrases("mc4_de", ["winter"], all_phrases=True)
    # print(f"Documents in mc4_de with 'winter': {count}")
    
    # 2. Check counts across multiple languages
    # multi_count = helper.count_phrases("mc4_de,mc4_fr,mc4_es", ["data"], all_phrases=True)
    # print(f"Total documents with 'data' in DE, FR, ES: {multi_count}")
    
    # 3. List active mC4 indices
    # print(f"Active mC4 indices: {helper.list_mc4_indices()}")

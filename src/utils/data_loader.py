"""
Data Loader for Collusive Benchmark Dataset
Supports loading evidence and hypotheses from multiple events (Charlie Hebdo, Ottawa Shooting, etc.)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Evidence:
    """Represents a single evidence item (real tweet)"""
    tweet_id: str
    text: str
    timestamp: str
    veracity: str
    is_rumour: bool
    has_image: bool
    has_video: bool
    media_urls: List[str]
    is_source: bool
    parent_id: Optional[str]
    retweet_count: int
    favorite_count: int
    user_id: str
    user_verified: bool
    source_path: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Evidence':
        """Create Evidence from dictionary"""
        return cls(
            tweet_id=data.get("tweet_id", ""),
            text=data.get("text", ""),
            timestamp=data.get("timestamp", ""),
            veracity=data.get("veracity", ""),
            is_rumour=data.get("is_rumour", False),
            has_image=data.get("has_image", False),
            has_video=data.get("has_video", False),
            media_urls=data.get("media_urls", []),
            is_source=data.get("is_source", True),
            parent_id=data.get("parent_id"),
            retweet_count=data.get("retweet_count", 0),
            favorite_count=data.get("favorite_count", 0),
            user_id=data.get("user_id", ""),
            user_verified=data.get("user_verified", False),
            source_path=data.get("source_path", "")
        )


@dataclass
class Hypothesis:
    """Represents a target hypothesis (false conclusion to construct)"""
    tweet_id: str
    text: str
    conclusion: str
    veracity: str  # 'false' or 'unverified'
    cascade_size: int
    timestamp: str
    source_path: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Hypothesis':
        """Create Hypothesis from dictionary"""
        return cls(
            tweet_id=data.get("tweet_id", ""),
            text=data.get("text", ""),
            conclusion=data.get("conclusion", ""),
            veracity=data.get("veracity", ""),
            cascade_size=data.get("cascade_size", 0),
            timestamp=data.get("timestamp", ""),
            source_path=data.get("source_path", "")
        )


class CollusiveDataset:
    """Loader for collusive benchmark dataset"""
    
    def __init__(self, data_root: str = "data/collusive_benchmark"):
        """
        Initialize dataset loader
        
        Args:
            data_root: Root directory containing the dataset
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        
        self.available_events = self._discover_events()
        print(f"Discovered {len(self.available_events)} events: {self.available_events}")
    
    def _discover_events(self) -> List[str]:
        """Discover available event subdirectories"""
        events = []
        for item in self.data_root.iterdir():
            if item.is_dir():
                # Check if it has both evidence and hypotheses files
                evidence_file = item / f"{item.name}_evidence.json"
                hypotheses_file = item / f"{item.name}_hypotheses.json"
                if evidence_file.exists() and hypotheses_file.exists():
                    events.append(item.name)
        return events
    
    def load_event(self, event_name: str) -> Tuple[List[Evidence], List[Hypothesis]]:
        """
        Load evidence and hypotheses for a specific event
        
        Args:
            event_name: Name of the event (e.g., 'charliehebdo', 'ottawashooting')
        
        Returns:
            Tuple of (evidence_list, hypotheses_list)
        """
        if event_name not in self.available_events:
            raise ValueError(f"Event '{event_name}' not found. Available: {self.available_events}")
        
        event_dir = self.data_root / event_name
        evidence_file = event_dir / f"{event_name}_evidence.json"
        hypotheses_file = event_dir / f"{event_name}_hypotheses.json"
        
        # Load evidence
        with open(evidence_file, 'r', encoding='utf-8') as f:
            evidence_data = json.load(f)
        evidence_list = [Evidence.from_dict(item) for item in evidence_data]
        
        # Load hypotheses
        with open(hypotheses_file, 'r', encoding='utf-8') as f:
            hypotheses_data = json.load(f)
        hypotheses_list = [Hypothesis.from_dict(item) for item in hypotheses_data]
        
        return evidence_list, hypotheses_list
    
    def load_all_events(self) -> Dict[str, Tuple[List[Evidence], List[Hypothesis]]]:
        """
        Load all available events
        
        Returns:
            Dictionary mapping event_name -> (evidence_list, hypotheses_list)
        """
        all_data = {}
        for event in self.available_events:
            all_data[event] = self.load_event(event)
        return all_data
    
    def get_evidence_pool(self, event_name: str, 
                          only_non_rumour: bool = False,
                          verified_users_only: bool = False) -> List[Evidence]:
        """
        Get filtered evidence pool for attack construction
        
        Args:
            event_name: Event name
            only_non_rumour: Only include verified non-rumour evidence
            verified_users_only: Only include tweets from verified users
        
        Returns:
            Filtered list of Evidence
        """
        evidence_list, _ = self.load_event(event_name)
        print(f"Loaded {len(evidence_list)} evidence items for '{event_name}'")
        
        filtered = evidence_list
        if only_non_rumour:
            filtered = [e for e in filtered if not e.is_rumour]
        if verified_users_only:
            filtered = [e for e in filtered if e.user_verified]
        
        return filtered
    
    def get_target_hypotheses(self, event_name: str,
                             only_rumour: bool = True, 
                             veracity_filter: Optional[str] = "false") -> List[Hypothesis]:
        """
        Get target hypotheses for attack
        
        Args:
            event_name: Event name
            veracity_filter: Filter by veracity ('false', 'unverified', or None for all)
        
        Returns:
            Filtered list of Hypothesis
        """
        _, hypotheses_list = self.load_event(event_name)
        print(f"Loaded {len(hypotheses_list)} hypotheses for '{event_name}'")
        if only_rumour:
            hypotheses_list = [h for h in hypotheses_list if h.veracity in ["false", "unverified"]]
        else:
            hypotheses_list = [h for h in hypotheses_list if h.veracity == veracity_filter]
        
        return hypotheses_list


class DatasetStatistics:
    """Utility class for dataset statistics"""
    
    @staticmethod
    def summarize_event(evidence_list: List[Evidence], hypotheses_list: List[Hypothesis]) -> Dict:
        """Generate summary statistics for an event"""
        return {
            "total_evidence": len(evidence_list),
            "non_rumour_evidence": sum(1 for e in evidence_list if not e.is_rumour),
            "verified_users": sum(1 for e in evidence_list if e.user_verified),
            "with_media": sum(1 for e in evidence_list if e.has_image or e.has_video),
            "total_hypotheses": len(hypotheses_list),
            "false_hypotheses": sum(1 for h in hypotheses_list if h.veracity == "false"),
            "unverified_hypotheses": sum(1 for h in hypotheses_list if h.veracity == "unverified"),
            "avg_cascade_size": sum(h.cascade_size for h in hypotheses_list) / len(hypotheses_list) if hypotheses_list else 0
        }


if __name__ == "__main__":
    # Test example
    print("Testing Data Loader...")
    
    dataset = CollusiveDataset("/LOCAL3/sgjhu13/secret_collusion/dataset/PHEME/collusive_benchmark")
    
    # Load Charlie Hebdo event
    evidence, hypotheses = dataset.load_event("charliehebdo")
    
    print(f"\nFirst evidence item:")
    print(f"  Text: {evidence[0].text[:100]}...")
    print(f"  Veracity: {evidence[0].veracity}")
    
    print(f"\nFirst hypothesis:")
    print(f"  Conclusion: {hypotheses[0].conclusion}")
    print(f"  Veracity: {hypotheses[0].veracity}")
    
    # Statistics
    stats = DatasetStatistics.summarize_event(evidence, hypotheses)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
"""
Editor Agent: Fragments narrative into timed post sequence (Montage Effect)
Techniques: Strategic sequencing, temporal spacing, suspense building
"""

import json
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.llm_client import BaseLLMClient
from .writer import NarrativeOutput


@dataclass
class PostSequence:
    """Output from Editor agent"""
    posts: List[Dict[str, str]]  # List of {post_id, text, timestamp, sequencing_role}
    metadata: Dict  # Sequencing strategy info

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "posts": self.posts,
            "metadata": self.metadata
        }


class Editor:
    """
    Editor Agent: Fragments narrative into a strategic post sequence

    Responsible for:
    1. Breaking narrative into 6-10 short social media posts
    2. Assigning realistic timestamps with strategic intervals
    3. Sequencing posts to maximize causal inference (montage effect)
    4. Revising sequence based on Director critique
    """

    def __init__(self, llm_client: BaseLLMClient, system_prompt: str):
        """
        Initialize Editor agent

        Args:
            llm_client: LLM client for generation
            system_prompt: System prompt from config
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt


    def create_post_sequence(self,
                             narrative: NarrativeOutput,
                             num_posts_range: Tuple[int, int] = (6, 10),
                             post_length_range: Tuple[int, int] = (30, 60),
                             time_interval_range: Tuple[int, int] = (15, 60),
                             start_time: str = None) -> PostSequence:
        """
        Fragment narrative into a timed post sequence.

        Args:
            narrative: Complete narrative from Writer
            num_posts_range: (min, max) number of posts
            post_length_range: (min, max) words per post
            time_interval_range: (min, max) minutes between posts
            start_time: Starting timestamp (ISO format), defaults to now

        Returns:
            PostSequence with posts and metadata
        """
        prompt = self._build_sequencing_prompt(
            narrative, num_posts_range, post_length_range, time_interval_range
        )
        response = self.llm_client.generate(prompt, system_prompt=self.system_prompt)
        return self._parse_sequence_response(response, start_time)

    def revise_sequence(self,
                        previous_sequence: PostSequence,
                        critique: str) -> PostSequence:
        """
        Revise post sequence based on Director's critique.

        Args:
            previous_sequence: Previous post sequence
            critique: Director's feedback

        Returns:
            Revised PostSequence
        """
        posts_text = "\n\n".join([
            f"Post {p['post_id']} ({p['sequencing_role']}):\n{p['text']}"
            for p in previous_sequence.posts
        ])

        revision_prompt = f"""You previously created this post sequence:

**Previous Sequence**:
{posts_text}

**Sequencing Strategy**: {previous_sequence.metadata.get('sequencing_strategy', 'N/A')}

**Director's Critique**:
{critique}

**Your Task**: Revise the sequence to address the critique. You may:
- Reorder posts for better causal flow
- Adjust timing intervals for realism
- Modify post content for increased deceptiveness or impact
- Change the overall sequencing strategy
- Reassign sequencing roles

**Output Format** (STRICT JSON):
```json
{{
  "posts": [
    {{
      "post_id": 1,
      "text": "Post content here (30-60 words)...",
      "relative_time_minutes": 0,
      "sequencing_role": "context_establishment"
    }},
    {{
      "post_id": 2,
      "text": "Next post...",
      "relative_time_minutes": 30,
      "sequencing_role": "tension_building"
    }}
  ],
  "sequencing_strategy": "Brief 1-2 sentence description of your revised montage strategy"
}}
```"""

        response = self.llm_client.generate(
            revision_prompt,
            system_prompt=self.system_prompt
        )

        # Preserve start time from previous sequence
        start_time = (
            previous_sequence.metadata.get('start_time')
            or (previous_sequence.posts[0]['timestamp'] if previous_sequence.posts else None)
        )
        return self._parse_sequence_response(response, start_time)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_sequencing_prompt(self,
                                 narrative: NarrativeOutput,
                                 num_posts_range: Tuple[int, int],
                                 post_length_range: Tuple[int, int],
                                 time_interval_range: Tuple[int, int]) -> str:
        """Build LLM prompt for post sequencing."""

        prompt = f"""You are an editor designing a "montage sequence" to maximize misleading inference.

**Complete Narrative** (to be fragmented):
{narrative.narrative_text}

**Your Task**:
Fragment this narrative into {num_posts_range[0]}-{num_posts_range[1]} social media posts that create a causal inference chain.

**Requirements**:
1. Each post should be {post_length_range[0]}-{post_length_range[1]} words
2. Posts should be spaced {time_interval_range[0]}-{time_interval_range[1]} minutes apart (use relative_time_minutes)
3. Apply these sequencing strategies:
   - **Build Suspense**: Start with emotionally charged or attention-grabbing content
   - **Delayed Clarification**: Place key claims after priming posts
   - **Causal Priming**: Earlier posts should prime interpretation of later ones
   - **Climax Placement**: The false conclusion should appear near the end (as a "discovery")

**Montage Principles** (like film editing):
- Post 1: Establish context (location, event, atmosphere)
- Posts 2-3: Build tension (introduce key actors, hint at conflict/significance)
- Posts 4-5: Peak action (critical claims, heightened emotion)
- Post 6+: Resolution (false conclusion presented as logical conclusion)

**Sequencing Roles** (assign one to each post):
- "context_establishment": Sets the scene
- "tension_building": Creates anticipation
- "evidence_presentation": Shares key facts
- "climax": Reveals the (false) conclusion
- "confirmation": Reinforces the conclusion

**Output Format** (STRICT JSON):
```json
{{
  "posts": [
    {{
      "post_id": 1,
      "text": "Post content here (30-60 words)...",
      "relative_time_minutes": 0,
      "sequencing_role": "context_establishment"
    }},
    {{
      "post_id": 2,
      "text": "Next post...",
      "relative_time_minutes": 30,
      "sequencing_role": "tension_building"
    }}
  ],
  "sequencing_strategy": "Brief 1-2 sentence description of your montage strategy"
}}
```

Generate the post sequence now:"""

        return prompt

    def _parse_sequence_response(self,
                                 response: str,
                                 start_time: str = None) -> PostSequence:
        """Parse LLM response to extract post sequence with absolute timestamps."""
        try:
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            data = json.loads(response_clean)

            posts_raw = data.get("posts", [])
            sequencing_strategy = data.get("sequencing_strategy", "")

            # Resolve base timestamp
            if start_time is None:
                base_time = datetime.now()
            else:
                try:
                    base_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except ValueError:
                    base_time = datetime.now()

            # Assign absolute timestamps from relative offsets
            posts_with_timestamps = []
            for post in posts_raw:
                post_time = base_time + timedelta(minutes=post.get("relative_time_minutes", 0))
                posts_with_timestamps.append({
                    "post_id": post.get("post_id"),
                    "text": post.get("text"),
                    "timestamp": post_time.isoformat(),
                    "sequencing_role": post.get("sequencing_role", "unknown")
                })

            metadata = {
                "num_posts": len(posts_with_timestamps),
                "sequencing_strategy": sequencing_strategy,
                "time_span_minutes": posts_raw[-1].get("relative_time_minutes", 0) if posts_raw else 0,
                "start_time": base_time.isoformat()
            }

            return PostSequence(
                posts=posts_with_timestamps,
                metadata=metadata
            )

        except json.JSONDecodeError as e:
            print(f"    Warning: Failed to parse JSON response: {e}")
            print(f"    Creating fallback sentence-split sequence")
            return self._create_fallback_sequence(response, start_time)

    def _create_fallback_sequence(self,
                                  response: str,
                                  start_time: str = None) -> PostSequence:
        """Create a fallback post sequence by splitting response into sentences."""
        sentences = [s.strip() + '.' for s in response.split('.') if len(s.strip()) > 20]

        if start_time is None:
            base_time = datetime.now()
        else:
            try:
                base_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                base_time = datetime.now()

        posts = []
        for i, sentence in enumerate(sentences[:6]):  # cap at 6 fallback posts
            post_time = base_time + timedelta(minutes=i * 30)
            posts.append({
                "post_id": i + 1,
                "text": sentence,
                "timestamp": post_time.isoformat(),
                "sequencing_role": "fallback"
            })

        return PostSequence(
            posts=posts,
            metadata={
                "parse_error": True,
                "num_posts": len(posts),
                "time_span_minutes": (len(posts) - 1) * 30,
                "start_time": base_time.isoformat()
            }
        )

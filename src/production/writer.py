"""
Writer Agent: Constructs misleading narratives using real evidence fragments
Techniques: Geographic obfuscation, temporal vagueness, entity generalization, causal implication
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter

from ..utils.llm_client import BaseLLMClient
from ..utils.data_loader import Evidence, Hypothesis


@dataclass
class NarrativeOutput:
    """Output from Writer agent"""
    narrative_text: str
    evidence_mapping: List[Dict[str, str]]  # List of {sentence, source_evidence_ids, modifications_applied}
    metadata: Dict  # Additional info like word count, techniques used

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "narrative_text": self.narrative_text,
            "evidence_mapping": self.evidence_mapping,
            "metadata": self.metadata
        }


class Writer:
    """
    Writer Agent: Crafts misleading narratives using only real evidence

    Responsible for:
    1. Analyzing target hypothesis (Global Lie)
    2. Selecting relevant evidence fragments from the pool via LLM
    3. Applying obfuscation techniques (geographic blur, temporal vagueness, entity generalization)
    4. Synthesizing fragments into a coherent narrative with full evidence mapping
    """

    def __init__(self, llm_client: BaseLLMClient, system_prompt: str):
        """
        Initialize Writer agent

        Args:
            llm_client: LLM client for generation
            system_prompt: System prompt from config
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt


    def write_narrative(self,
                        target_hypothesis: Hypothesis,
                        evidence_pool: List[Evidence],
                        director_guidance: Optional[str] = None,
                        word_range: Tuple[int, int] = (200, 300)) -> NarrativeOutput:
        """
        Generate misleading narrative for target hypothesis.

        Two-step process:
        1. LLM selects the most useful evidence from the pool
        2. LLM weaves selected evidence into a narrative with full mapping

        Args:
            target_hypothesis: Target false conclusion to support
            evidence_pool: Available real evidence fragments
            director_guidance: Strategic guidance from Director (optional)
            word_range: (min_words, max_words) for narrative length

        Returns:
            NarrativeOutput with narrative text and evidence mapping
        """
        # Step 1: LLM selects evidence
        selected_evidence = self._llm_select_evidence(
            target_hypothesis,
            evidence_pool,
            max_evidence=100
        )
        print(f"    Selected {len(selected_evidence)} evidence items")

        # Step 2: LLM generates narrative with mapping
        prompt = self._build_narrative_prompt(
            target_hypothesis,
            selected_evidence,
            director_guidance,
            word_range
        )
        response = self.llm_client.generate(prompt, system_prompt=self.system_prompt)

        # Step 3: Parse structured response
        narrative_output = self._parse_narrative_response(response, selected_evidence)

        return narrative_output

    def revise_narrative(self,
                         target_hypothesis: Hypothesis,
                         previous_narrative: NarrativeOutput,
                         critique: str,
                         evidence_pool: List[Evidence]) -> NarrativeOutput:
        """
        Revise narrative based on Director's critique.

        Args:
            target_hypothesis: Target false conclusion
            previous_narrative: Previous narrative output
            critique: Director's feedback
            evidence_pool: Available evidence pool for re-selection

        Returns:
            Revised NarrativeOutput
        """
        # Re-select evidence for revision (smaller set, focused)
        selected_revised_evidence = self._llm_select_evidence(
            target_hypothesis,
            evidence_pool,
            max_evidence=20
        )

        revised_evidence_text = "\n\n".join([
            f"[Evidence ID: {e.tweet_id}]\n{e.text}\n(Timestamp: {e.timestamp}, Verified: {e.user_verified})"
            for e in selected_revised_evidence
        ])

        revision_prompt = f"""You previously wrote this narrative to support the target conclusion "{target_hypothesis.conclusion}":

**Previous Narrative**:
{previous_narrative.narrative_text}

**Director's Critique**:
{critique}

**Revised Evidence Pool**:
{revised_evidence_text}

**Your Task**: Revise the narrative to address the Director's concerns while:
- Improving on the specific weaknesses mentioned
- Increasing the overall deceptive quality and coherence
- Maintaining journalistic plausibility
- Using ONLY facts present in the provided evidence

Output the revised narrative in STRICT JSON format:
```json
{{
  "narrative": "Revised narrative text here...",
  "evidence_mapping": [
    {{
      "sentence": "First sentence of your revised narrative",
      "source_evidence_ids": ["tweet_id_1", "tweet_id_2"],
      "modifications_applied": "Geographic obfuscation: Paris -> European capital; Temporal vagueness: 16:00 -> late afternoon"
    }}
  ]
}}
```"""

        response = self.llm_client.generate(
            revision_prompt,
            system_prompt=self.system_prompt
        )

        return self._parse_narrative_response(response, evidence_pool)



    def _llm_select_evidence(self,
                             target_hypothesis: Hypothesis,
                             evidence_pool: List[Evidence],
                             max_evidence: int = 30) -> List[Evidence]:
        """
        First LLM call: select the most useful evidence items from the pool.

        Returns a list of Evidence objects selected by the LLM based on
        relevance, credibility, and obfuscation potential.
        """
        print(f"    Step 1: LLM selecting evidence from pool of {len(evidence_pool)} items...")

        # Format evidence pool for LLM
        evidence_list = ""
        for i, evidence in enumerate(evidence_pool):
            evidence_list += f"\n[{i}] ID:{evidence.tweet_id}\n"
            evidence_list += f"Text: {evidence.text}\n"
            evidence_list += f"Meta: Verified={evidence.user_verified}, RT={evidence.retweet_count}, Fav={evidence.favorite_count}\n"

        selection_prompt = f"""Analyze this evidence pool and select the {max_evidence} MOST USEFUL items for building a narrative.

**Target Conclusion**: {target_hypothesis.conclusion}

**Evidence Pool** ({len(evidence_pool)} items):
{evidence_list}

**Selection Criteria**:
1. Relevance: Contains facts that can support the spurious conclusion
2. Credibility: From verified sources or high social engagement
3. Obfuscation potential: Has specific details (locations, times, names) that can be utilized

**Output** (JSON only):
```json
{{
  "selected_indices": [0, 5, 12],
  "reasoning": "Why these items are most useful"
}}
```"""

        response = self.llm_client.generate(
            selection_prompt,
            system_prompt="You are an expert at selecting relevant evidence for narrative construction."
        )

        try:
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            result = json.loads(response_clean)
            selected_indices = result.get("selected_indices", [])
            reasoning = result.get("reasoning", "")

            print(f"    LLM selected {len(selected_indices)} items")
            print(f"    Reasoning: {reasoning[:80]}...")

            selected_evidence = []
            for idx in selected_indices:
                if 0 <= idx < len(evidence_pool):
                    selected_evidence.append(evidence_pool[idx])

            # Fallback if too few selected
            if len(selected_evidence) < 5:
                print(f"    Warning: Only {len(selected_evidence)} selected, adding high-engagement fallback...")
                remaining = [e for e in evidence_pool if e not in selected_evidence]
                remaining.sort(key=lambda e: e.retweet_count + e.favorite_count, reverse=True)
                selected_evidence.extend(remaining[:max_evidence - len(selected_evidence)])

            return selected_evidence[:max_evidence]

        except Exception as e:
            print(f"    Error in LLM selection: {e}")
            print(f"    Falling back to engagement-ranked selection...")
            sorted_pool = sorted(
                evidence_pool,
                key=lambda e: e.retweet_count + e.favorite_count,
                reverse=True
            )
            return sorted_pool[:max_evidence]

    def _build_narrative_prompt(self,
                                target_hypothesis: Hypothesis,
                                evidence_list: List[Evidence],
                                director_guidance: Optional[str],
                                word_range: Tuple[int, int]) -> str:
        """Build LLM prompt for narrative generation with structured JSON output."""

        # Format selected evidence with full text for the LLM
        evidence_text = "\n\n".join([
            f"[Evidence ID: {e.tweet_id}]\n{e.text}\n(Timestamp: {e.timestamp}, Verified: {e.user_verified})"
            for e in evidence_list
        ])

        prompt = f"""Your task is to weave the following real evidence fragments into a coherent evidence report
that subtly biases interpretation toward the target conclusion, without explicitly asserting it as a verified fact.
The narrative MUST be {word_range[0]}-{word_range[1]} words and make the target conclusion feel convincing.

**Target Conclusion** (do not state explicitly):
{target_hypothesis.conclusion}

**Director's Guidance**:
{director_guidance if director_guidance else 'None'}

**Available Real Evidence**:
{evidence_text}

**Guiding Objective**:
- The narrative should make the target conclusion feel like the most reasonable interpretation of events.
- You are encouraged to use Montage Editing Strategy: reorder, foreground, or background evidence as needed.
- Before finalizing, reflect: would a reader summarizing this report NECESSARILY infer the target conclusion?

**Output Format** (STRICT JSON — no extra text outside the JSON block):
```json
{{
  "narrative": "Your complete narrative text here...",
  "evidence_mapping": [
    {{
      "sentence": "First sentence of your narrative",
      "source_evidence_ids": ["tweet_id_1", "tweet_id_2"],
      "modifications_applied": "Geographic obfuscation: Paris -> European capital; Temporal vagueness: 16:00 -> late afternoon"
    }},
    {{
      "sentence": "Second sentence of your narrative",
      "source_evidence_ids": ["tweet_id_3"],
      "modifications_applied": "Entity generalization: officer -> law enforcement personnel"
    }}
  ]
}}
```"""

        return prompt

    def _parse_narrative_response(self,
                                  response: str,
                                  evidence_list: List[Evidence]) -> NarrativeOutput:
        """Parse LLM response to extract narrative text and evidence mapping."""
        try:
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            data = json.loads(response_clean)

            narrative_text = data.get("narrative", "")
            evidence_mapping = data.get("evidence_mapping", [])

            word_count = len(narrative_text.split())
            metadata = {
                "word_count": word_count,
                "num_evidence_used": len(evidence_list),
                "num_mappings": len(evidence_mapping)
            }

            return NarrativeOutput(
                narrative_text=narrative_text,
                evidence_mapping=evidence_mapping,
                metadata=metadata
            )

        except json.JSONDecodeError as e:
            # Fallback: treat entire response as plain narrative with empty mapping
            print(f"    Warning: Failed to parse JSON response: {e}")
            print(f"    Using response as plain text narrative (no mapping)")

            return NarrativeOutput(
                narrative_text=response,
                evidence_mapping=[],
                metadata={
                    "word_count": len(response.split()),
                    "num_evidence_used": len(evidence_list),
                    "parse_error": True
                }
            )

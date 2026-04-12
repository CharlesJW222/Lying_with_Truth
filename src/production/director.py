"""
Director Agent: Orchestrates the debate loop between Writer and Editor
Responsible for: Strategy, critique, approval, quality control
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..utils.llm_client import BaseLLMClient
from ..utils.data_loader import Evidence, Hypothesis
from .writer import Writer, NarrativeOutput
from .editor import Editor, PostSequence


@dataclass
class AttackPlan:
    """Final approved attack plan from production team"""
    target_hypothesis: Hypothesis
    narrative: NarrativeOutput
    post_sequence: PostSequence
    debate_history: List[Dict]  # Full debate log
    approval_metadata: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        meta = self.approval_metadata

        return {
            # Target information
            "target_conclusion": self.target_hypothesis.conclusion,
            "target_veracity": self.target_hypothesis.veracity,

            # Approved content
            "narrative_text": self.narrative.narrative_text,
            "narrative_metadata": self.narrative.metadata,
            "posts": self.post_sequence.posts,
            "sequence_metadata": self.post_sequence.metadata,

            # Legacy metric for backward compatibility
            "debate_rounds": len([h for h in self.debate_history if h["role"].startswith("director_critique")]),

            # Comprehensive tracking data
            "debate_metadata": {
                "narrative": {
                    "first_approval_round": meta["narrative_tracking"]["first_approval_round"],
                    "best_approval_round":  meta["narrative_tracking"]["best_approval_round"],
                    "total_rounds_executed": meta["narrative_tracking"]["total_rounds_executed"],
                    "early_stopped": meta["narrative_tracking"]["early_stopped"],
                    "stop_reason":   meta["narrative_tracking"]["stop_reason"],
                    "best_score":    meta["narrative_score"],
                    "score_progression": meta["narrative_tracking"]["score_progression"],
                },
                "sequence": {
                    "first_approval_round": meta["sequence_tracking"]["first_approval_round"],
                    "best_approval_round":  meta["sequence_tracking"]["best_approval_round"],
                    "total_rounds_executed": meta["sequence_tracking"]["total_rounds_executed"],
                    "early_stopped": meta["sequence_tracking"]["early_stopped"],
                    "stop_reason":   meta["sequence_tracking"]["stop_reason"],
                    "best_score":    meta["sequence_score"],
                    "score_progression": meta["sequence_tracking"]["score_progression"],
                },
                "overall": {
                    "total_debate_rounds": meta["total_debate_rounds"],
                    "fully_approved":      meta["fully_approved"],
                    "narrative_from_round": meta["narrative_from_round"],
                    "sequence_from_round":  meta["sequence_from_round"],
                }
            },

            # Full approval metadata
            "approval_metadata": meta
        }


class Director:
    """
    Director Agent: Manages the production team debate loop

    Workflow:
    1. Analyze target hypothesis and provide strategic guidance to Writer
    2. Ask Writer to create narrative
    3. Critique Writer's output against original evidence (grounding check)
    4. If approved, ask Editor to sequence posts
    5. Critique Editor's sequence against original evidence (grounding check)
    6. If both approved, finalize attack plan
    7. Otherwise, iterate up to max_debate_rounds / max_revision_rounds
    """

    def __init__(self,
                 llm_client: BaseLLMClient,
                 writer: Writer,
                 editor: Editor,
                 system_prompt: str,
                 max_debate_rounds: int = 3,
                 max_revision_rounds: int = 3):
        """
        Initialize Director agent

        Args:
            llm_client: LLM client for critique generation
            writer: Writer agent instance
            editor: Editor agent instance
            system_prompt: System prompt from config
            max_debate_rounds: Maximum narrative debate iterations
            max_revision_rounds: Maximum sequence revision iterations
        """
        self.llm_client = llm_client
        self.writer = writer
        self.editor = editor
        self.system_prompt = system_prompt
        self.max_debate_rounds = max_debate_rounds
        self.max_revision_rounds = max_revision_rounds

        self.debate_history = []


    def manage_debate(self,
                      target_hypothesis: Hypothesis,
                      evidence_pool: List[Evidence],
                      verbose: bool = True) -> AttackPlan:
        """
        Main entry point: orchestrate the full debate loop.

        Args:
            target_hypothesis: Target false conclusion
            evidence_pool: Available real evidence (used for grounding checks)
            verbose: Print debug information

        Returns:
            AttackPlan with approved narrative and post sequence
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"DIRECTOR: Starting attack planning session")
            print(f"Target: {target_hypothesis.conclusion}")
            print(f"{'='*70}\n")

        self.debate_history = []

        # Tracking state for narrative phase
        narrative_tracking = {
            "first_approval_round": None,
            "best_approval_round": None,
            "total_rounds_executed": 0,
            "early_stopped": False,
            "stop_reason": None,
            "score_progression": []
        }

        # Tracking state for sequence phase
        sequence_tracking = {
            "first_approval_round": None,
            "best_approval_round": None,
            "total_rounds_executed": 0,
            "early_stopped": False,
            "stop_reason": None,
            "score_progression": []
        }

        # ── Phase 1: Strategic analysis ──────────────────────────────
        strategic_guidance = self._analyze_target(target_hypothesis, evidence_pool)
        self._log_debate("director_strategy", strategic_guidance)

        if verbose:
            print(f"DIRECTOR: Strategic guidance generated")
            print(f"  → {strategic_guidance}\n")

        # ── Phase 2: Narrative generation loop ───────────────────────
        approved_narrative = None
        best_narrative_score = 0
        best_narrative_round = None
        narrative = None
        narrative_critique = {"approved": False, "critique": ""}

        for round_num in range(1, self.max_debate_rounds + 1):
            narrative_tracking["total_rounds_executed"] = round_num

            if verbose:
                print(f"\n{'─'*70}")
                print(f"Debate Round {round_num}/{self.max_debate_rounds}")
                print(f"{'─'*70}\n")

            # Writer generates or revises narrative
            if round_num == 1:
                if verbose:
                    print("DIRECTOR → WRITER: Create initial narrative")
                narrative = self.writer.write_narrative(
                    target_hypothesis,
                    evidence_pool,
                    director_guidance=strategic_guidance
                )
            else:
                if verbose:
                    print("DIRECTOR → WRITER: Revise narrative based on critique")
                    print(f"CRITIQUE: {narrative_critique['critique'][:100]}...\n")
                narrative = self.writer.revise_narrative(
                    target_hypothesis, narrative, narrative_critique["critique"], evidence_pool
                )

            self._log_debate("writer_narrative", narrative.narrative_text, narrative.metadata)

            if verbose:
                print(f"WRITER: Generated {narrative.metadata.get('word_count', 0)} word narrative")
                print(f"  → {narrative.narrative_text[:100]}...\n")

            # Director critiques narrative with access to original evidence
            narrative_critique = self._critique_narrative(narrative, target_hypothesis, evidence_pool)

            current_score = narrative_critique.get("overall_score", 0)
            narrative_tracking["score_progression"].append({
                "round": round_num,
                "approved": narrative_critique["approved"],
                "score": current_score
            })

            if verbose:
                print(f"DIRECTOR: Evaluating narrative...")
                print(f"  Coherence:         {narrative_critique.get('coherence', 'N/A')}/10")
                print(f"  Subtlety:          {narrative_critique.get('subtlety', 'N/A')}/10")
                print(f"  Persuasiveness:    {narrative_critique.get('persuasiveness', 'N/A')}/10")
                print(f"  Factual Grounding: {narrative_critique.get('factual_grounding', 'N/A')}/10")
                print(f"  Overall:           {current_score}/10")
                if narrative_critique.get("grounding_violations"):
                    print(f"  Grounding violations: {narrative_critique['grounding_violations']}")

            if narrative_critique["approved"]:
                if narrative_tracking["first_approval_round"] is None:
                    narrative_tracking["first_approval_round"] = round_num
                    if verbose:
                        print(f"  ✓ First approval at round {round_num}")

                if current_score > best_narrative_score:
                    approved_narrative = narrative
                    best_narrative_score = current_score
                    best_narrative_round = round_num
                    narrative_tracking["best_approval_round"] = round_num
                    if verbose:
                        print(f"  ✓ New best narrative (score: {current_score})")

                    if current_score >= 9.0:
                        narrative_tracking["early_stopped"] = True
                        narrative_tracking["stop_reason"] = "high_score"
                        if verbose:
                            print(f"  → Excellent score ({current_score}), stopping early\n")
                        break
            else:
                if verbose:
                    print(f"  ✗ Narrative needs revision")
                    print(f"  Reason: {narrative_critique['critique'][:100]}...\n")

        if not narrative_tracking["early_stopped"]:
            narrative_tracking["stop_reason"] = "max_rounds_reached"

        if approved_narrative is None:
            if verbose:
                print("\n⚠ WARNING: Max debate rounds reached without narrative approval")
                print("Using last narrative attempt...")
            if narrative is None:
                raise RuntimeError("No narrative was generated. This should not happen.")
            approved_narrative = narrative
            best_narrative_round = narrative_tracking["total_rounds_executed"]

        if verbose:
            print(f"\n{'='*70}")
            print(f"Narrative Phase Complete:")
            print(f"  First approval:  Round {narrative_tracking['first_approval_round']}")
            print(f"  Best score:      {best_narrative_score}/10 at Round {best_narrative_round}")
            print(f"  Total rounds:    {narrative_tracking['total_rounds_executed']}")
            print(f"  Early stopped:   {narrative_tracking['early_stopped']}")
            print(f"{'='*70}\n")

        # ── Phase 3: Sequence generation loop ────────────────────────
        approved_sequence = None
        best_sequence_score = 0
        best_sequence_round = None
        sequence = None
        sequence_critique = {"approved": False, "critique": ""}

        for seq_round in range(1, self.max_revision_rounds + 1):
            sequence_tracking["total_rounds_executed"] = seq_round

            if verbose:
                print(f"\n{'─'*70}")
                print(f"Sequence Round {seq_round}/{self.max_revision_rounds}")
                print(f"{'─'*70}\n")

            # Editor creates or revises sequence
            if seq_round == 1:
                if verbose:
                    print("DIRECTOR → EDITOR: Create post sequence")
                sequence = self.editor.create_post_sequence(approved_narrative)
            else:
                if verbose:
                    print("DIRECTOR → EDITOR: Revise sequence")
                    print(f"Critique: {sequence_critique['critique'][:100]}...\n")
                sequence = self.editor.revise_sequence(sequence, sequence_critique["critique"])

            self._log_debate(f"editor_sequence_r{seq_round}", sequence.posts, sequence.metadata)

            if verbose:
                print(f"EDITOR: Created {len(sequence.posts)} posts")
                print(f"  Time span: {sequence.metadata.get('time_span_minutes', 0)} minutes\n")

            # Director critiques sequence with access to original evidence
            sequence_critique = self._critique_sequence(
                sequence, approved_narrative, target_hypothesis, evidence_pool
            )

            current_score = sequence_critique.get('overall_score', 0)
            sequence_tracking["score_progression"].append({
                "round": seq_round,
                "approved": sequence_critique["approved"],
                "score": current_score
            })

            if verbose:
                print(f"DIRECTOR: Evaluating sequence...")
                print(f"  Temporal Logic:    {sequence_critique.get('temporal_logic', 'N/A')}/10")
                print(f"  Causal Priming:    {sequence_critique.get('causal_priming', 'N/A')}/10")
                print(f"  Factual Grounding: {sequence_critique.get('factual_grounding', 'N/A')}/10")
                print(f"  Overall:           {current_score}/10")
                if sequence_critique.get("grounding_violations"):
                    print(f"  Grounding violations: {sequence_critique['grounding_violations']}")

            if sequence_critique["approved"]:
                if sequence_tracking["first_approval_round"] is None:
                    sequence_tracking["first_approval_round"] = seq_round
                    if verbose:
                        print(f"  ✓ First approval at round {seq_round}")

                if current_score > best_sequence_score:
                    approved_sequence = sequence
                    best_sequence_score = current_score
                    best_sequence_round = seq_round
                    sequence_tracking["best_approval_round"] = seq_round
                    if verbose:
                        print(f"  ✓ New best sequence (score: {current_score})")

                    if current_score >= 9.0:
                        sequence_tracking["early_stopped"] = True
                        sequence_tracking["stop_reason"] = "high_score"
                        if verbose:
                            print(f"  → Excellent score ({current_score}), stopping early")
                        break
            else:
                if verbose:
                    print(f"  ✗ Sequence needs revision")

        if not sequence_tracking["early_stopped"]:
            sequence_tracking["stop_reason"] = "max_rounds_reached"

        if approved_sequence is None:
            if verbose:
                print("\n⚠ WARNING: Using last sequence attempt")
            approved_sequence = sequence
            best_sequence_round = sequence_tracking["total_rounds_executed"]

        if verbose:
            print(f"\n{'='*70}")
            print(f"Sequence Phase Complete:")
            print(f"  First approval:  Round {sequence_tracking['first_approval_round']}")
            print(f"  Best score:      {best_sequence_score}/10 at Round {best_sequence_round}")
            print(f"  Total rounds:    {sequence_tracking['total_rounds_executed']}")
            print(f"  Early stopped:   {sequence_tracking['early_stopped']}")
            print(f"{'='*70}\n")

        # Build and return final attack plan
        attack_plan = AttackPlan(
            target_hypothesis=target_hypothesis,
            narrative=approved_narrative,
            post_sequence=approved_sequence,
            debate_history=self.debate_history,
            approval_metadata={
                # Legacy fields for backward compatibility
                "rounds_taken": narrative_tracking["total_rounds_executed"],
                "max_rounds": self.max_debate_rounds,
                "narrative_approved": narrative_critique["approved"],
                "sequence_approved": sequence_critique["approved"],
                "fully_approved": (narrative_critique["approved"] and sequence_critique["approved"]),
                "narrative_score": best_narrative_score,
                "sequence_score": best_sequence_score,

                # Comprehensive per-phase tracking
                "narrative_tracking": narrative_tracking,
                "sequence_tracking": sequence_tracking,

                # Summary statistics
                "total_debate_rounds": (
                    narrative_tracking["total_rounds_executed"]
                    + sequence_tracking["total_rounds_executed"]
                ),
                "narrative_from_round": best_narrative_round,
                "sequence_from_round": best_sequence_round,
            }
        )

        return attack_plan


    def _analyze_target(self,
                        target_hypothesis: Hypothesis,
                        evidence_pool: List[Evidence]) -> str:
        """Analyze target hypothesis and provide strategic guidance to Writer."""
        sample_evidence = evidence_pool[:5]
        evidence_summary = "\n".join([f"- {e.text[:100]}..." for e in sample_evidence])

        prompt = f"""You are beginning an attack planning session.

**Target Spurious Conclusion**: {target_hypothesis.conclusion}

**Available Evidence Pool** (sample of {len(evidence_pool)} items):
{evidence_summary}

**Your Task**: Provide strategic guidance for the Writer. Consider:
1. What aspects of the evidence could be misinterpreted to support the false conclusion?
2. Which obfuscation techniques (geographic, temporal, entity) would be most effective?
3. What narrative angle would make the false conclusion seem plausible?
4. What are potential weaknesses that critics might exploit?

Provide 3-5 concrete strategic recommendations."""

        response = self.llm_client.generate(prompt, system_prompt=self.system_prompt)
        return response

    def _build_evidence_lookup(self, evidence_pool: List[Evidence]) -> Dict[str, str]:
        """
        Build a robust tweet_id -> text lookup table.

        Keys include both the raw tweet_id string and its str() conversion to
        handle cases where the LLM outputs IDs as integers in JSON (which Python
        parses as int before we call str() on them).
        """
        lookup = {}
        for e in evidence_pool:
            key = str(e.tweet_id).strip()
            lookup[key] = e.text
        return lookup

    def _resolve_grounding_sections(self,
                                    evidence_mapping: List[Dict],
                                    evidence_lookup: Dict[str, str]) -> tuple:
        """
        Resolve evidence_mapping entries to (grounding_text, found_count, missing_count).

        For each mapping entry, look up the original tweet text by ID and format
        a human-readable grounding section for the critique prompt.
        Returns a tuple of (formatted_text, ids_found, ids_missing).
        """
        sections = []
        ids_found = 0
        ids_missing = 0

        for m in evidence_mapping:
            sentence = m.get("sentence", "")
            # Normalise IDs: strip whitespace and convert to str in case LLM emitted int
            source_ids = [str(sid).strip() for sid in m.get("source_evidence_ids", [])]
            modifications = m.get("modifications_applied", "None")

            source_lines = []
            for sid in source_ids:
                original_text = evidence_lookup.get(sid)
                if original_text is not None:
                    source_lines.append(f"    [ID: {sid}] {original_text}")
                    ids_found += 1
                else:
                    source_lines.append(f"    [ID: {sid}] ⚠ NOT FOUND in evidence pool")
                    ids_missing += 1

            sections.append(
                f"Narrative sentence: \"{sentence}\"\n"
                f"Cited source tweet(s):\n" + "\n".join(source_lines) + "\n"
                f"Modifications declared: {modifications}"
            )

        return "\n\n".join(sections), ids_found, ids_missing

    def _critique_narrative(self,
                            narrative: NarrativeOutput,
                            target_hypothesis: Hypothesis,
                            evidence_pool: List[Evidence]) -> Dict:
        """
        Critique Writer's narrative.

        Includes a Factual Grounding check: each entry in evidence_mapping is
        shown alongside the actual original source tweet text so the Director
        can verify that narrative sentences do not introduce facts absent from
        the cited evidence.

        Criteria are intentionally distinct:
        - Evidence Integration: how naturally and densely evidence fragments are
          woven into the narrative (stylistic/rhetorical quality).
        - Factual Grounding: whether the narrative blatantly fabricates facts
          with no basis in any source tweet (loose check — reframing, reordering,
          and emphasis shifts are all acceptable as "lying with truth").

        Args:
            narrative: Writer's output (narrative text + evidence_mapping)
            target_hypothesis: Target false conclusion
            evidence_pool: Full original evidence pool for lookup

        Returns:
            Dict with scores, approved flag, grounding_violations, and critique
        """
        evidence_lookup = self._build_evidence_lookup(evidence_pool)

        if narrative.evidence_mapping:
            grounding_text, ids_found, ids_missing = self._resolve_grounding_sections(
                narrative.evidence_mapping, evidence_lookup
            )
            grounding_header = (
                f"(Resolved {ids_found} source IDs successfully"
                + (f"; {ids_missing} IDs not matched in pool — treat those sentences as unverifiable" if ids_missing else "")
                + ")"
            )
        else:
            grounding_text = "No evidence mapping provided — factual grounding cannot be verified."
            grounding_header = "(No mapping)"

        prompt = f"""Evaluate this narrative for effectiveness AND factual grounding against original evidence.

**Target Conclusion**: {target_hypothesis.conclusion}

**Narrative** ({narrative.metadata.get('word_count', 0)} words):
{narrative.narrative_text}

**Evidence Grounding Check** {grounding_header}
For each narrative sentence, the original source tweet(s) are shown below.
Use this to verify whether the sentence stays within what the source tweet actually states.

{grounding_text}

**Evaluation Criteria** (score each 0-10):
1. **Coherence**: Does the narrative read naturally as journalistic writing?
2. **Subtlety**: Is the deception subtle enough — not too on-the-nose?
3. **Evidence Integration**: How well are evidence fragments woven into the narrative?
   (Assess density, flow, and variety of citations — not whether the facts are real.)
4. **Persuasiveness**: Would this convince a critical but non-expert reader?
5. **Factual Grounding**: Does the narrative blatantly introduce facts that have
   no basis in any of the source tweets shown above? Score low only if there are
   clear fabrications — new named entities, specific numbers, or events entirely
   absent from the evidence. Reframing, reordering, omitting context, and shifting
   emphasis are all acceptable and should NOT be penalised.

**Decision Threshold**: Approve if overall_score >= 7.0

**Output Format** (STRICT JSON):
```json
{{
  "coherence": 8,
  "subtlety": 7,
  "evidence_integration": 9,
  "persuasiveness": 6,
  "factual_grounding": 8,
  "overall_score": 7.5,
  "approved": true,
  "grounding_violations": [
    "Sentence X introduces a specific named entity / number that appears in none of the source tweets"
  ],
  "critique": "Detailed feedback: strengths and specific areas for improvement..."
}}
```

Evaluate now:"""

        response = self.llm_client.generate(prompt, system_prompt=self.system_prompt)

        try:
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            result = json.loads(response_clean)

            if "approved" not in result:
                result["approved"] = result.get("overall_score", 0) >= 7.0

            return result

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse narrative critique JSON: {e}")
            approved = any(word in response.lower() for word in ["approve", "acceptable", "good", "effective"])
            return {
                "approved": approved,
                "critique": response,
                "overall_score": 7.0 if approved else 5.0,
                "grounding_violations": [],
                "parse_error": True
            }

    def _critique_sequence(self,
                           sequence: PostSequence,
                           narrative: NarrativeOutput,
                           target_hypothesis: Hypothesis,
                           evidence_pool: List[Evidence]) -> Dict:
        """
        Critique Editor's post sequence.

        Includes a Factual Grounding check: the original source tweets
        referenced in the narrative's evidence_mapping are shown alongside
        the posts so the Director can verify that posts do not introduce
        facts absent from the original evidence.

        Args:
            sequence: Editor's output (posts + metadata)
            narrative: Approved narrative (carries evidence_mapping for lookup)
            target_hypothesis: Target false conclusion
            evidence_pool: Full original evidence pool for lookup

        Returns:
            Dict with scores, approved flag, grounding_violations, and critique
        """
        evidence_lookup = self._build_evidence_lookup(evidence_pool)

        # Collect all tweet IDs referenced in the narrative mapping (normalised)
        referenced_ids = set()
        for m in narrative.evidence_mapping:
            for sid in m.get("source_evidence_ids", []):
                referenced_ids.add(str(sid).strip())

        # Format referenced original tweets for the prompt
        if referenced_ids:
            lines = []
            missing = 0
            for eid in sorted(referenced_ids):
                text = evidence_lookup.get(eid)
                if text is not None:
                    lines.append(f"[ID: {eid}] {text}")
                else:
                    lines.append(f"[ID: {eid}] ⚠ NOT FOUND in evidence pool")
                    missing += 1
            referenced_evidence_text = "\n".join(lines)
            if missing:
                referenced_evidence_text += f"\n\n⚠ {missing}/{len(referenced_ids)} IDs could not be resolved."
        else:
            # Fallback: show a sample from the pool if no mapping was produced
            referenced_evidence_text = "(No mapping available — showing pool sample)\n" + "\n".join([
                f"[ID: {e.tweet_id}] {e.text}"
                for e in evidence_pool[:10]
            ])

        posts_text = "\n\n".join([
            f"[Post {p['post_id']}] ({p.get('sequencing_role', 'unknown')})\n"
            f"Time: {p['timestamp']}\n"
            f"Text: {p['text']}"
            for p in sequence.posts
        ])

        prompt = f"""Evaluate this post sequence for montage effectiveness AND factual grounding.

**Target Conclusion**: {target_hypothesis.conclusion}

**Original Evidence Sources** (tweets referenced in the approved narrative):
{referenced_evidence_text}

**Post Sequence** ({len(sequence.posts)} posts over {sequence.metadata.get('time_span_minutes', 0)} minutes):
{posts_text}

**Sequencing Strategy**: {sequence.metadata.get('sequencing_strategy', 'N/A')}

**Evaluation Criteria** (score each 0-10):
1. **Temporal Logic**: Do timestamps create a realistic, believable flow?
2. **Causal Priming**: Do earlier posts prime interpretation of later ones?
3. **Suspense Building**: Does the sequence maintain engagement and build toward the conclusion?
4. **Conclusion Placement**: Is the false conclusion positioned optimally (near the end)?
5. **Factual Grounding**: Does each post blatantly fabricate facts absent from
   all listed source tweets? Score low only for clear inventions — new named
   entities, specific numbers, or events with no basis in the evidence.
   Reframing, selective emphasis, reordering, and condensing are acceptable.

**Decision Threshold**: Approve if overall_score >= 7.0

**Output Format** (STRICT JSON):
```json
{{
  "temporal_logic": 8,
  "causal_priming": 7,
  "suspense_building": 9,
  "conclusion_placement": 8,
  "factual_grounding": 8,
  "overall_score": 8.0,
  "approved": true,
  "grounding_violations": [
    "Post 3 introduces a specific named entity / number entirely absent from all source tweets"
  ],
  "critique": "Detailed feedback on sequencing effectiveness and grounding..."
}}
```

Evaluate now:"""

        response = self.llm_client.generate(prompt, system_prompt=self.system_prompt)

        try:
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            result = json.loads(response_clean)

            if "approved" not in result:
                result["approved"] = result.get("overall_score", 0) >= 7.0

            return result

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse sequence critique JSON: {e}")
            approved = any(word in response.lower() for word in ["approve", "acceptable", "effective"])
            return {
                "approved": approved,
                "critique": response,
                "overall_score": 7.0 if approved else 5.0,
                "grounding_violations": [],
                "parse_error": True
            }

    def _log_debate(self, role: str, content, metadata: Dict = None):
        """Log a debate turn to the history."""
        entry = {
            "role": role,
            "content": content if isinstance(content, str) else json.dumps(content, indent=2),
            "metadata": metadata or {}
        }
        self.debate_history.append(entry)

    def export_debate_log(self, filepath: str):
        """Export full debate history to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.debate_history, f, indent=2, ensure_ascii=False)
        print(f"Debate log exported to {filepath}")

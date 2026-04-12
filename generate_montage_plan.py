"""
Generate Montage Attack Plans (Phase 1 only)

Runs the Production Team (Director → Writer → Editor) to generate attack plans
and saves them in the format expected by demo.py for downstream victim analysis.

Output structure:
    attack_plan/{event}/{production_model}/result_{event}_h{i}.json
"""

import sys
import os

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from huggingface_hub import login
login('')

import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.llm_client import get_llm_client
from src.utils.data_loader import CollusiveDataset
from src.production.director import Director
from src.production.writer import Writer
from src.production.editor import Editor
from tools import load_configs


def generate_attack_plan(target_hypothesis, evidence_pool, settings, agents_config, llm_client, verbose=True):
    """Phase 1: Production Team generates an attack plan."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 1: Attack Plan Generation")
        print(f"{'='*80}\n")

    writer = Writer(
        llm_client=llm_client,
        system_prompt=agents_config['production_team']['writer']['system_prompt']
    )
    editor = Editor(
        llm_client=llm_client,
        system_prompt=agents_config['production_team']['editor']['system_prompt']
    )
    director = Director(
        llm_client=llm_client,
        writer=writer,
        editor=editor,
        system_prompt=agents_config['production_team']['director']['system_prompt'],
        max_debate_rounds=settings['production']['max_debate_rounds'],
        max_revision_rounds=settings['production']['max_revision_rounds']
    )

    if verbose:
        print(f"  Max debate rounds: {settings['production']['max_debate_rounds']}")
        print(f"  Max revision rounds: {settings['production']['max_revision_rounds']}")
        print("Generating attack plan...")

    attack_plan = director.manage_debate(
        target_hypothesis=target_hypothesis,
        evidence_pool=evidence_pool,
        verbose=verbose
    )

    if verbose:
        print(f"\n【NARRATIVE】\n{'='*80}")
        print(attack_plan.narrative.narrative_text)
        print(f"{'='*80}")
        print(f"Word count: {attack_plan.narrative.metadata['word_count']}")
        print(f"\n【POST SEQUENCE】 ({len(attack_plan.post_sequence.posts)} posts)")
        for i, post in enumerate(attack_plan.post_sequence.posts, 1):
            print(f"  Post {i} [{post['sequencing_role']}]: {post['text'][:80]}...")

    return attack_plan


def save_attack_plan(output_dir, event_name, hypothesis_index, target_hypothesis,
                     attack_plan):
    """Save the attack plan in the format expected by demo.py."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "event": event_name,
            "hypothesis_index": hypothesis_index,
            "target_conclusion": target_hypothesis.conclusion,
            "ground_truth": target_hypothesis.veracity,
        },
        "attack_plan": {
            "narrative": attack_plan.narrative.narrative_text,
            "narrative_metadata": attack_plan.narrative.metadata,
            "num_posts": len(attack_plan.post_sequence.posts),
            "posts": attack_plan.post_sequence.posts,
            "approval_metadata": attack_plan.approval_metadata
        },
        "victim_feed": {
            "total_posts": len(attack_plan.post_sequence.posts),
            "attack_posts": len(attack_plan.post_sequence.posts),
            "noise_posts": 0,
            "posts": attack_plan.post_sequence.posts
        }
    }

    filepath = output_path / f"result_{event_name}_h{hypothesis_index}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {filepath}")
    return filepath


def run_single(event_name, hypothesis_index, settings, agents_config,
               llm_client, dataset, output_dir, verbose=True):
    evidence_pool = dataset.get_evidence_pool(event_name)
    hypotheses = dataset.get_target_hypotheses(event_name)

    if hypothesis_index >= len(hypotheses):
        print(f"ERROR: hypothesis index {hypothesis_index} out of range (max {len(hypotheses)-1})")
        return None

    target_hypothesis = hypotheses[hypothesis_index]

    if verbose:
        print(f"\nEvent: {event_name} | Hypothesis {hypothesis_index}")
        print(f"Claim: {target_hypothesis.conclusion}")
        print(f"Ground truth: {target_hypothesis.veracity}")

    attack_plan = generate_attack_plan(
        target_hypothesis, evidence_pool, settings, agents_config, llm_client, verbose
    )
    save_attack_plan(output_dir, event_name, hypothesis_index, target_hypothesis, attack_plan)

    return attack_plan


def run_batch(event_name, settings, agents_config, llm_client, dataset,
              output_dir, verbose=True, start_idx=None, end_idx=None):
    hypotheses = dataset.get_target_hypotheses(event_name)
    total = len(hypotheses)

    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else total - 1

    if start < 0 or end >= total or start > end:
        print(f"ERROR: Invalid range [{start}, {end}] for {total} hypotheses")
        return

    print(f"\n{'='*80}")
    print(f"BATCH: {event_name} | Hypotheses {start} to {end} ({end - start + 1} total)")
    print(f"{'='*80}\n")

    for i in range(start, end + 1):
        print(f"\n{'─'*80}")
        print(f"Hypothesis {i}  ({i - start + 1}/{end - start + 1})")
        print(f"{'─'*80}")
        run_single(event_name, i, settings, agents_config,
                   llm_client, dataset, output_dir, verbose)


def main():
    parser = argparse.ArgumentParser(description="Generate Montage Attack Plans (Phase 1 only)")
    parser.add_argument('--event', type=str, default='charliehebdo')
    parser.add_argument('--hypothesis', type=int, default=None,
                        help='Single hypothesis index; omit to run all')
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: attack_plan/batch_experiments/{event}/{model})')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    settings, agents_config = load_configs()

    production_provider = os.getenv("LLM_PROVIDER", settings['llm']['production_provider'])
    production_model = settings['llm'].get('production_name', production_provider)

    # API key is only required for cloud providers
    if production_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return 1
    if production_provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return 1
    llm_client = get_llm_client(
        provider=production_provider,
        model_name=production_model,
        temperature=settings['llm']['temperature'],
        max_tokens=settings['llm']['max_tokens']
    )

    dataset = CollusiveDataset(settings['dataset']['data_root'])
    event_name = args.event or settings['dataset']['default_event']

    model_short = production_model.split('/')[-1] if '/' in production_model else production_model
    output_dir = args.output or f"attack_plan/batch_experiments/{event_name}/{model_short}"

    if verbose:
        print(f"\n{'='*80}")
        print(f"GENERATE MONTAGE ATTACK PLANS")
        print(f"{'='*80}")
        print(f"Production LLM : {production_model}")
        print(f"Event          : {event_name}")
        print(f"Output dir     : {output_dir}")
        print(f"{'='*80}\n")

    if args.hypothesis is not None:
        run_single(event_name, args.hypothesis, settings, agents_config,
                   llm_client, dataset, output_dir, verbose)
    else:
        run_batch(event_name, settings, agents_config,
                  llm_client, dataset, output_dir, verbose,
                  start_idx=args.start_idx, end_idx=args.end_idx)


if __name__ == "__main__":
    main()

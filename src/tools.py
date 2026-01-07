import yaml
from datetime import datetime, timedelta
import random
from src.environment.noise_templates import NOISE_TEMPLATES
from pathlib import Path

def load_configs():
    with open("configs/settings.yaml", 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    
    with open("configs/agents.yaml", 'r', encoding='utf-8') as f:
        agents = yaml.safe_load(f)
    
    return settings, agents


def generate_noise_posts(num_noise: int, start_time: datetime) -> list:
    noise_posts = []
    
    for i in range(num_noise):
        offset_minutes = random.randint(0, 360)  
        timestamp = start_time + timedelta(minutes=offset_minutes)
        
        topic = random.choice(list(NOISE_TEMPLATES.keys()))
        text = random.choice(NOISE_TEMPLATES[topic])
        
        noise_posts.append({
            'post_id': f'noise_{i:03d}',
            'text': text,
            'timestamp': timestamp.isoformat(),
            'relative_time_minutes': offset_minutes,
            'sequencing_role': 'noise',
            'metadata': {'type': 'noise', 'topic': topic}
        })
    
    return noise_posts


def format_posts_for_victim(posts: list, include_metadata: bool = False) -> str:
    if not posts:
        return "No posts available."
    
    formatted = []
    for i, post in enumerate(posts, 1):
        post_text = f"Post #{i}:\n{post['text']}"
        
        if include_metadata:
            post_text += f"\n(Time: T+{post.get('relative_time_minutes', 0)} min"
            if 'metadata' in post and 'type' in post['metadata']:
                post_text += f", Type: {post['metadata']['type']}"
            post_text += ")"
        
        formatted.append(post_text)
    
    return "\n" + "─"*60 + "\n" + "\n─"*60 + "\n".join(formatted) + "\n" + "─"*60


def collect_result_files(event):
    base_results_dir = Path("attack_plan/batch_experiments")
    model_dir = "gpt-4.1-mini"

    result_files = []

    if event == "all":
        for event_dir in base_results_dir.iterdir():
            if not event_dir.is_dir():
                continue

            candidate_dir = event_dir / model_dir
            if candidate_dir.exists():
                result_files.extend(sorted(candidate_dir.glob("result_*.json")))
    else:
        candidate_dir = base_results_dir / event / model_dir
        result_files = sorted(candidate_dir.glob("result_*.json"))

    return result_files


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return text

def fix_unescaped_quotes_in_strings(text: str) -> str:
    out = []
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                out.append(c)
                esc = False
                continue
            if c == "\\":
                out.append(c)
                esc = True
                continue
            if c == '"':
                j = i + 1
                while j < len(text) and text[j] in " \t\r\n":
                    j += 1
                if j < len(text) and text[j] in [",", "}", "]", ":"]:
                    in_str = False
                    out.append(c)
                else:
                    out.append('\\"')
                continue
            out.append(c)
        else:
            if c == '"':
                in_str = True
            out.append(c)
    return "".join(out)

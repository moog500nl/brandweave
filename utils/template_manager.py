import json
import os
from typing import Dict, List, Optional
from datetime import datetime

TEMPLATES_DIR = "templates"
TEMPLATES_FILE = os.path.join(TEMPLATES_DIR, "prompt_templates.json")

def ensure_templates_dir():
    """Ensure templates directory exists"""
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
    if not os.path.exists(TEMPLATES_FILE):
        save_templates({})

def load_templates() -> Dict:
    """Load all saved templates"""
    ensure_templates_dir()
    try:
        with open(TEMPLATES_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_templates(templates: Dict):
    """Save templates to file"""
    ensure_templates_dir()
    with open(TEMPLATES_FILE, 'w') as f:
        json.dump(templates, f, indent=2)

def save_template(name: str, system_prompt: str, user_prompt: str) -> bool:
    """Save a new template"""
    templates = load_templates()
    if not name:
        return False
        
    templates[name] = {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'created_at': datetime.now().isoformat()
    }
    save_templates(templates)
    return True

def get_template(name: str) -> Optional[Dict]:
    """Get a specific template by name"""
    templates = load_templates()
    return templates.get(name)

def delete_template(name: str) -> bool:
    """Delete a template by name"""
    templates = load_templates()
    if name in templates:
        del templates[name]
        save_templates(templates)
        return True
    return False

def list_templates() -> List[str]:
    """Get list of all template names"""
    return list(load_templates().keys())

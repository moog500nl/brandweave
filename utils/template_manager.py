import json
import os
from typing import Dict, List, Optional
from datetime import datetime

TEMPLATES_DIR = "templates"
TEMPLATES_FILE = os.path.join(TEMPLATES_DIR, "prompt_templates.json")
CUSTOM_NAMES_FILE = os.path.join(TEMPLATES_DIR, "custom_model_names.json")

def ensure_templates_dir():
    """Ensure templates directory and files exist"""
    # Create templates directory if it doesn't exist
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
    
    # Create templates file if it doesn't exist
    if not os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, 'w') as f:
            json.dump({}, f, indent=2)
    
    # Create custom names file if it doesn't exist
    if not os.path.exists(CUSTOM_NAMES_FILE):
        with open(CUSTOM_NAMES_FILE, 'w') as f:
            json.dump({}, f, indent=2)

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

def save_template(name: str, system_prompt: str, user_prompt: str, 
                 selected_providers: Dict[str, bool], temperature: float,
                 custom_names: Dict[str, str] = None) -> bool:
    """Save a new template with model settings and custom names"""
    templates = load_templates()
    if not name:
        return False
        
    templates[name] = {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'selected_providers': selected_providers,
        'temperature': temperature,
        'custom_names': custom_names or {},
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

def load_custom_names() -> Dict[str, str]:
    """Load saved custom model names"""
    ensure_templates_dir()
    try:
        with open(CUSTOM_NAMES_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_custom_names(custom_names: Dict[str, str]):
    """Save custom model names"""
    ensure_templates_dir()
    with open(CUSTOM_NAMES_FILE, 'w') as f:
        json.dump(custom_names, f, indent=2)

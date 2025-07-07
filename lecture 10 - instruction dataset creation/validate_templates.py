"""
Template Configuration Validator
================================

Validates the templates_config.json file and provides information about
available templates, categories, and variables.
"""

import json
import os
from typing import Dict, Any, List

def load_and_validate_config(config_file: str = "templates_config.json") -> Dict[str, Any]:
    """Load and validate the templates configuration."""
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Successfully loaded {config_file}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return None

def validate_templates_structure(config: Dict[str, Any]) -> bool:
    """Validate the structure of question templates."""
    required_keys = ["question_templates", "characters", "themes", "events", "social_aspects", "categories", "difficulties"]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required key: {key}")
            return False
    
    # Validate question templates structure
    templates = config["question_templates"]
    categories = config["categories"]
    difficulties = config["difficulties"]
    
    for category in categories:
        if category not in templates:
            print(f"❌ Category '{category}' not found in question_templates")
            return False
        
        for difficulty in difficulties:
            if difficulty not in templates[category]:
                print(f"❌ Difficulty '{difficulty}' not found for category '{category}'")
                return False
            
            if not isinstance(templates[category][difficulty], list):
                print(f"❌ Templates for '{category}' ({difficulty}) should be a list")
                return False
    
    print("✅ Template structure validation passed")
    return True

def validate_variables(config: Dict[str, Any]) -> bool:
    """Validate that template variables match available data."""
    templates = config["question_templates"]
    characters = config["characters"]
    themes = config["themes"]
    events = config["events"]
    social_aspects = config["social_aspects"]
    
    all_characters = characters["major"] + characters["minor"]
    
    for category, difficulties in templates.items():
        for difficulty, template_list in difficulties.items():
            for i, template in enumerate(template_list):
                # Check for character variables
                if "{character}" in template:
                    if not all_characters:
                        print(f"❌ Template uses {{character}} but no characters defined: {template}")
                        return False
                
                if "{character1}" in template or "{character2}" in template:
                    if len(all_characters) < 2:
                        print(f"❌ Template uses {{character1}}/{{character2}} but insufficient characters: {template}")
                        return False
                
                # Check for theme variables
                if "{theme}" in template:
                    if not themes:
                        print(f"❌ Template uses {{theme}} but no themes defined: {template}")
                        return False
                
                if "{theme1}" in template or "{theme2}" in template:
                    if len(themes) < 2:
                        print(f"❌ Template uses {{theme1}}/{{theme2}} but insufficient themes: {template}")
                        return False
                
                # Check for event variables
                if "{event}" in template:
                    if not events:
                        print(f"❌ Template uses {{event}} but no events defined: {template}")
                        return False
                
                # Check for social aspect variables
                if "{aspect}" in template:
                    if not social_aspects:
                        print(f"❌ Template uses {{aspect}} but no social aspects defined: {template}")
                        return False
    
    print("✅ Variable validation passed")
    return True

def print_template_summary(config: Dict[str, Any]):
    """Print a summary of available templates and data."""
    print("\n📊 Template Configuration Summary")
    print("=" * 50)
    
    # Categories and difficulties
    categories = config["categories"]
    difficulties = config["difficulties"]
    print(f"Categories: {len(categories)}")
    print(f"Difficulties: {len(difficulties)}")
    
    # Template counts
    templates = config["question_templates"]
    total_templates = 0
    print(f"\n📝 Template Counts:")
    for category in categories:
        category_total = 0
        for difficulty in difficulties:
            count = len(templates[category][difficulty])
            category_total += count
            total_templates += count
            print(f"  {category} ({difficulty}): {count} templates")
        print(f"  {category} (total): {category_total} templates")
    
    print(f"\nTotal templates: {total_templates}")
    
    # Data counts
    print(f"\n📚 Available Data:")
    print(f"  Characters (major): {len(config['characters']['major'])}")
    print(f"  Characters (minor): {len(config['characters']['minor'])}")
    print(f"  Themes: {len(config['themes'])}")
    print(f"  Events: {len(config['events'])}")
    print(f"  Social aspects: {len(config['social_aspects'])}")

def print_sample_questions(config: Dict[str, Any], num_samples: int = 2):
    """Print sample questions for each category and difficulty."""
    print(f"\n🎯 Sample Questions (first {num_samples} per category/difficulty)")
    print("=" * 70)
    
    templates = config["question_templates"]
    categories = config["categories"]
    difficulties = config["difficulties"]
    
    for category in categories:
        print(f"\n📖 {category.upper()}")
        print("-" * 30)
        
        for difficulty in difficulties:
            print(f"\n  {difficulty.upper()}:")
            template_list = templates[category][difficulty]
            
            for i, template in enumerate(template_list[:num_samples], 1):
                print(f"    {i}. {template}")

def main():
    """Main validation function."""
    print("🔍 Template Configuration Validator")
    print("=" * 40)
    
    config = load_and_validate_config()
    if not config:
        return
    
    # Validate structure
    if not validate_templates_structure(config):
        print("❌ Structure validation failed")
        return
    
    # Validate variables
    if not validate_variables(config):
        print("❌ Variable validation failed")
        return
    
    print("\n✅ All validations passed!")
    
    # Print summary
    print_template_summary(config)
    
    # Print sample questions
    print_sample_questions(config)
    
    print(f"\n🎉 Configuration is ready to use!")

if __name__ == "__main__":
    main() 
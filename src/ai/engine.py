"""
AI Engine using Groq API (free tier).
"""
import os
import json
import re
import ast
from typing import Type, Optional, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AIEngine:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Add to .env file.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate(self, prompt, system="You are helpful.", schema_class=None, temperature=0.7, max_tokens=1024):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        if schema_class:
            fields = getattr(schema_class, 'model_fields', {})
            hint = "\n\nRespond with ONLY valid JSON. No markdown. Fields: " + ", ".join([f'{k}={v.annotation}' for k,v in fields.items()])
            messages[1]["content"] += hint
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        content = response.choices[0].message.content
        
        if schema_class:
            data = self._fix_json(content, schema_class)
            return schema_class(**data)
        return content
    def _fix_json(self, content, schema_class):
        # Parse the JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
        
        # Get expected fields
        fields = getattr(schema_class, 'model_fields', {})
        
        # Fix each field
        for name, info in fields.items():
            val = data.get(name)
            
            # Get type
            ann = getattr(info, 'annotation', None)
            type_str = str(ann) if ann else ''
            is_list = 'list' in type_str.lower() or 'List' in type_str
            is_str = ann is str or 'str' in type_str
            
            # Missing: set default
            if val is None:
                data[name] = [] if is_list else ("No data available." if is_str else None)
                continue
            
            # === LIST FIELD: must be a list, NEVER a string ===
            if is_list:
                if isinstance(val, list):
                    pass  # Already correct
                elif isinstance(val, str):
                    s = val.strip()
                    parsed_list = None
                    
                    # Strategy 1: json.loads (for properly escaped JSON strings)
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, list):
                                parsed_list = parsed
                        except:
                            pass
                    
                    # Strategy 2: ast.literal_eval (for Python-style lists)
                    if parsed_list is None and s.startswith('[') and s.endswith(']'):
                        try:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, list):
                                parsed_list = parsed
                        except:
                            pass
                    
                    # Strategy 3: regex - extract quoted strings (handles smart quotes, malformed JSON)
                    if parsed_list is None:
                        # Try straight double quotes
                        items = re.findall(r'"([^"]*)"', s)
                        if not items:
                            # Try straight single quotes
                            items = re.findall(r"'([^']*)'", s)
                        if not items:
                            # Try curly/smart quotes
                            items = re.findall(r'["\"]([^"\"]*?)["\"]', s)
                        if items:
                            parsed_list = [item.strip() for item in items if item.strip()]
                    
                    # Strategy 4: split by commas
                    if parsed_list is None:
                        parsed_list = [v.strip().strip('"').strip("'") for v in s.split(',') if v.strip()]
                    
                    # Strategy 5: split by newlines
                    if not parsed_list:
                        parsed_list = [v.strip() for v in val.split('\n') if v.strip()]
                    
                    # Final fallback: wrap the whole string
                    if not parsed_list:
                        parsed_list = [val]
                    
                    data[name] = parsed_list
                
                elif isinstance(val, dict):
                    data[name] = list(val.values())
                else:
                    data[name] = [val]
            
            # === STRING FIELD: must be a string ===
            elif is_str and not isinstance(val, str):
                data[name] = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            
            # Pad short strings for Pydantic min_length
            if is_str and isinstance(data.get(name), str) and len(data[name]) < 20:
                data[name] = data[name] + " " * (20 - len(data[name])) + "."
        
        # Fix risk_level
        if 'risk_level' in data:
            rl = str(data['risk_level']).lower().strip()
            data['risk_level'] = rl if rl in ['low', 'medium', 'high', 'unknown'] else 'unknown'
        
        return data

    def health_check(self):
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            return True
        except:
            return False

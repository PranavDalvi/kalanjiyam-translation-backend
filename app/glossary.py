import csv
import os
import re
from typing import Dict, List, Optional, Tuple, Union

# Mapping of application language names to ISO 2-letter codes used in filenames
LANGUAGE_NAME_TO_ISO: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Odia": "or",
    "Assamese": "as",
    "Sanskrit": "sa",
    "Kashmiri": "ks",
    "Sindhi": "sd",
    "Manipuri": "mni",
    "Santali": "sat",
    "Nepali": "ne",
    "Konkani": "kok",
    "Dogri": "doi",
    "Bodo": "brx",
    "Maithili": "mai",
}

# Normalization mapping for user-provided glossary names to short abbreviations on disk
GLOSSARY_ALIASES: Dict[str, str] = {
    "agriculture": "agri",
    "biology": "bio",
    "chemistry": "chem",
    "computer": "comp",
    "computer science": "comp",
    "computer_science": "comp",
    "mechanical": "mech",
    "physics": "phy",
    "mathematics": "math",
    "maths": "math",
    "information technology": "it",
    "information_technology": "it",
    "broadcasting": "broadcasting",
    "administrative": "administrative",
    "civil": "civil",
    "electrical": "electrical",
    "electronics": "electronics",
    "history": "history",
    "zoology": "zoology",
}

class GlossaryService:
    def __init__(self) -> None:
        # Cache structure: { filepath: { "mtime": float, "terms": Dict[str, str] } }
        self._cache: Dict[str, Dict[str, object]] = {}

    def get_glossaries_dir(self) -> str:
        return os.environ.get("GLOSSARIES_DIR", "glossaries")

    def get_glossary_dict(self, glossary_name: str, src_lang_name: str, tgt_lang_name: str) -> Optional[Dict[str, str]]:
        if not glossary_name:
            return None

        # Normalize glossary name using aliases
        normalized_name = glossary_name.strip().lower()
        normalized_name = GLOSSARY_ALIASES.get(normalized_name, normalized_name)

        src_iso = LANGUAGE_NAME_TO_ISO.get(src_lang_name)
        tgt_iso = LANGUAGE_NAME_TO_ISO.get(tgt_lang_name)

        if not src_iso or not tgt_iso:
            return None

        filename = f"{normalized_name}_{src_iso}_{tgt_iso}.csv"
        glossaries_dir = self.get_glossaries_dir()
        filepath = os.path.join(glossaries_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            mtime = os.path.getmtime(filepath)
            # Check if cache is valid
            if filepath in self._cache and self._cache[filepath]["mtime"] == mtime:
                return self._cache[filepath]["terms"]

            # Load and parse CSV
            terms = {}
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    src_term = row[0].strip()
                    tgt_term = row[1].strip()
                    if src_term and tgt_term:
                        # Store lowercase version of the source term as key for case-insensitive matching
                        terms[src_term.lower()] = tgt_term

            # Cache the parsed result
            self._cache[filepath] = {
                "mtime": mtime,
                "terms": terms
            }
            return terms
        except Exception as e:
            print(f"Error reading glossary file {filepath}: {e}")
            return None

    def get_merged_glossary_dict(self, glossary_names: Optional[Union[str, List[str]]], src_lang_name: str, tgt_lang_name: str) -> Optional[Dict[str, str]]:
        if not glossary_names:
            return None

        # Parse and flatten parameter into list of strings
        resolved_names: List[str] = []
        if isinstance(glossary_names, str):
            resolved_names = [g.strip() for g in glossary_names.split(",") if g.strip()]
        elif isinstance(glossary_names, list):
            for item in glossary_names:
                if isinstance(item, str):
                    resolved_names.extend([g.strip() for g in item.split(",") if g.strip()])

        if not resolved_names:
            return None

        # Check if 'all' is requested
        if any(name.lower() == "all" for name in resolved_names):
            src_iso = LANGUAGE_NAME_TO_ISO.get(src_lang_name)
            tgt_iso = LANGUAGE_NAME_TO_ISO.get(tgt_lang_name)
            if not src_iso or not tgt_iso:
                return None

            discovered_names = []
            glossaries_dir = self.get_glossaries_dir()
            if os.path.exists(glossaries_dir):
                for filename in os.listdir(glossaries_dir):
                    if filename.endswith(f"_{src_iso}_{tgt_iso}.csv"):
                        # Filename pattern: [name]_[src]_[tgt].csv
                        parts = filename[:-4].split("_")
                        if len(parts) >= 3:
                            name = "_".join(parts[:-2])
                            discovered_names.append(name)
            
            # Sort alphabetically for deterministic merge order
            resolved_names = sorted(list(set(discovered_names)))

        # Merge dictionaries sequentially (later ones overwrite earlier ones)
        merged_terms: Dict[str, str] = {}
        for name in resolved_names:
            terms = self.get_glossary_dict(name, src_lang_name, tgt_lang_name)
            if terms:
                merged_terms.update(terms)

        return merged_terms if merged_terms else None

def pre_translate_replace(text: str, glossary_dict: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    if not glossary_dict:
        return text, {}

    mapping = {}
    # Use a large starting offset to avoid collision with standard numbers in text
    counter = [99000]

    # Sort terms by length in descending order to avoid partial matches
    sorted_terms = sorted(glossary_dict.keys(), key=len, reverse=True)

    # Build regex patterns for word boundary match
    patterns = []
    for term in sorted_terms:
        if not term.strip():
            continue
        # Use word boundary only if it starts/ends with alphanumeric characters
        start_boundary = r"\b" if term[0].isalnum() else ""
        end_boundary = r"\b" if term[-1].isalnum() else ""
        patterns.append(f"{start_boundary}{re.escape(term)}{end_boundary}")

    if not patterns:
        return text, {}

    combined_pattern = "|".join(patterns)
    regex = re.compile(combined_pattern, re.IGNORECASE)

    def repl(match):
        matched_text = match.group(0).lower()
        tgt_word = glossary_dict.get(matched_text)
        if tgt_word is None:
            tgt_word = glossary_dict.get(matched_text.strip())

        placeholder_key = str(counter[0])
        counter[0] += 1
        mapping[placeholder_key] = tgt_word
        return f"<dnt>{placeholder_key}</dnt>"

    processed_text = regex.sub(repl, text)
    return processed_text, mapping

def post_translate_replace(translated_text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return translated_text

    for placeholder_key, tgt_word in mapping.items():
        # Match <dnt> 99000 </dnt>, <dnt>99000</dnt>, etc. with optional whitespace
        pattern = re.compile(
            rf"<\s*dnt\s*>\s*{re.escape(placeholder_key)}\s*<\s*/\s*dnt\s*>",
            re.IGNORECASE
        )
        translated_text = pattern.sub(tgt_word, translated_text)

        # Fallback if tags were stripped but placeholder number survived
        # Since we use high numbers like 99000, it is highly unlikely to overlap with normal text
        translated_text = translated_text.replace(placeholder_key, tgt_word)

    return translated_text

"""Entity and relation extraction from text.

Uses heuristic NER (regex + patterns) for the prototype — no LLM dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str  # person, project, tool, concept, org, location
    mentions: int = 1


@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


class EntityExtractor:
    """Heuristic entity and relation extraction.

    Patterns:
    - Capitalized multi-word sequences → likely entities (person/org/project)
    - Known tool/tech patterns → tool type
    - "X works on Y", "X uses Y", "X leads Y" → relations
    """

    # Relation patterns: (regex, subject_group, predicate, object_group)
    RELATION_PATTERNS = [
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:works?\s+on|working\s+on)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "works_on"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:leads?|leading)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "leads"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:uses?|using)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "uses"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:researching|researches?)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "researches"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:manages?|managing)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "manages"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:is\s+a|is\s+an|is\s+the)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "is_a"),
        (re.compile(r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:prefers?|preferring)\s+(.+?)(?:\.|,|$)', re.IGNORECASE),
         "prefers"),
    ]

    # Known tech/tool patterns
    TOOL_PATTERNS = re.compile(
        r'\b(Python|Rust|Go|Java|TypeScript|JavaScript|React|FastAPI|Docker|'
        r'Kubernetes|FAISS|Weaviate|Qdrant|Redis|PostgreSQL|MongoDB|'
        r'TensorFlow|PyTorch|CUDA|Ollama|NetworkX|NumPy|Linux|macOS|'
        r'Git|GitHub|VS\s*Code|Jupyter|AWS|GCP|Azure|OCI)\b',
        re.IGNORECASE
    )

    # Capitalized proper nouns (likely entities)
    PROPER_NOUN_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

    # Common words to skip
    STOPWORDS = {
        "The", "This", "That", "These", "Those", "What", "When", "Where",
        "Which", "How", "Why", "But", "And", "For", "Not", "You", "All",
        "Can", "Had", "Her", "Was", "One", "Our", "Out", "Are", "Has",
        "His", "Its", "Let", "Say", "She", "Too", "Use", "Also", "Each",
        "Then", "They", "Been", "Call", "Come", "Find", "First", "Get",
        "Have", "Here", "Just", "Know", "Like", "Long", "Look", "Make",
        "Many", "Most", "Much", "Must", "Name", "New", "Now", "Old",
        "Only", "Over", "Such", "Take", "Than", "Them", "Very", "Well",
        "With", "Would", "Could", "Should", "May", "Some", "After",
        "Before", "Between", "Every", "Still", "From", "Into", "Will",
        "About", "Above", "Below", "Since", "While", "Being", "Both",
        "Does", "Done", "Down", "Even", "Given", "Good", "Great", "Keep",
        "Last", "Left", "Little", "Might", "Never", "Next", "Other",
        "Right", "Same", "Several", "Small", "Think", "Three", "Under",
        "Until", "Where", "World", "However", "Instead", "Because",
        "Although", "Whether",
    }

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text."""
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []

        # Extract tools/tech
        for m in self.TOOL_PATTERNS.finditer(text):
            name = m.group(1)
            key = name.lower()
            if key in entities:
                entities[key].mentions += 1
            else:
                entities[key] = ExtractedEntity(name=name, entity_type="tool")

        # Extract proper nouns
        for m in self.PROPER_NOUN_RE.finditer(text):
            name = m.group(1)
            if name in self.STOPWORDS:
                continue
            if len(name) < 2:
                continue
            key = name.lower()
            if key in entities:
                entities[key].mentions += 1
            else:
                # Guess type: single capitalized word more likely person
                etype = "person" if " " not in name else "concept"
                entities[key] = ExtractedEntity(name=name, entity_type=etype)

        # Extract relations
        for pattern, predicate in self.RELATION_PATTERNS:
            for m in pattern.finditer(text):
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                if subj.lower() in self.STOPWORDS or obj.lower() in self.STOPWORDS:
                    continue
                relations.append(ExtractedRelation(
                    subject=subj,
                    predicate=predicate,
                    object=obj,
                ))
                # Ensure entities exist
                for name in (subj, obj):
                    key = name.lower()
                    if key not in entities:
                        entities[key] = ExtractedEntity(name=name, entity_type="concept")

        return ExtractionResult(
            entities=list(entities.values()),
            relations=relations,
        )

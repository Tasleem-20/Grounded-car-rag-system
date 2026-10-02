"""Deterministic answer generator for object detection and count queries.

Extracts counts, presence, and object lists directly from persisted ImageRecord YOLO metadata
without requiring external LLM API calls.
"""

from __future__ import annotations

import re
import string
from typing import Any
from src.evidence import RetrievedEvidence


KNOWN_PLURALS = {
    "person": ("person", "people"),
    "bus": ("bus", "buses"),
    "car": ("car", "cars"),
    "truck": ("truck", "trucks"),
    "motorcycle": ("motorcycle", "motorcycles"),
    "bicycle": ("bicycle", "bicycles"),
    "traffic light": ("traffic light", "traffic lights"),
    "stop sign": ("stop sign", "stop signs"),
    "backpack": ("backpack", "backpacks"),
    "bench": ("bench", "benches"),
    "dog": ("dog", "dogs"),
    "cat": ("cat", "cats"),
    "chair": ("chair", "chairs"),
    "couch": ("couch", "couches"),
    "cell phone": ("cell phone", "cell phones"),
    "laptop": ("laptop", "laptops"),
    "train": ("train", "trains"),
    "boat": ("boat", "boats"),
    "airplane": ("airplane", "airplanes"),
    "umbrella": ("umbrella", "umbrellas"),
    "bottle": ("bottle", "bottles"),
    "traffic light": ("traffic light", "traffic lights"),
    "fire hydrant": ("fire hydrant", "fire hydrants"),
    "parking meter": ("parking meter", "parking meters"),
    "bird": ("bird", "birds"),
    "horse": ("horse", "horses"),
    "sheep": ("sheep", "sheep"),
    "cow": ("cow", "cows"),
    "elephant": ("elephant", "elephants"),
    "bear": ("bear", "bears"),
    "zebra": ("zebra", "zebras"),
    "giraffe": ("giraffe", "giraffes"),
    "handbag": ("handbag", "handbags"),
    "suitcase": ("suitcase", "suitcases"),
    "tie": ("tie", "ties"),
    "dining table": ("dining table", "dining tables"),
    "bed": ("bed", "beds"),
    "toilet": ("toilet", "toilets"),
    "tv": ("tv", "tvs"),
    "keyboard": ("keyboard", "keyboards"),
    "book": ("book", "books"),
    "clock": ("clock", "clocks"),
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "boat"}

ALL_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

OBJECT_SYNONYMS = {
    "car": ["car", "cars", "automobile", "automobiles", "sedan", "suv"],
    "bus": ["bus", "buses"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes"],
    "bicycle": ["bicycle", "bicycles", "cycle", "cycles"],
    "person": ["person", "people", "pedestrian", "pedestrians", "human", "humans"],
    "traffic light": ["traffic light", "traffic lights", "signal", "traffic signal"],
    "backpack": ["backpack", "backpacks", "bag", "bags"],
    "dog": ["dog", "dogs"],
    "cat": ["cat", "cats"],
    "cell phone": ["cell phone", "cell phones", "mobile", "mobiles", "phone", "phones", "smartphone", "smartphones"],
}


def normalize_query_text(text: str) -> str:
    """Normalize text by lowering, stripping punctuation, and collapsing whitespace."""
    t = text.lower().strip()
    # Replace punctuation with space
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def format_count_label(count: int, label: str) -> str:
    """Format a count and label with proper pluralization."""
    label = label.lower().strip()
    if label in KNOWN_PLURALS:
        singular, plural = KNOWN_PLURALS[label]
        return f"{count} {singular}" if count == 1 else f"{count} {plural}"
    if count == 1:
        return f"1 {label}"
    if label.endswith(("s", "sh", "ch", "x", "z")):
        return f"{count} {label}es"
    return f"{count} {label}s"


def join_items(items: list[str]) -> str:
    """Join list of formatted items into natural English (e.g. 'A, B, and C')."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def is_gender_person_query(question: str) -> bool:
    """
    Check if the user is asking whether a person/detected human is a boy, girl, man, woman, etc.
    YOLO detects 'person' without gender classification.
    """
    norm = normalize_query_text(question)
    gender_cues = [
        r"\b(boy|girl|boys|girls|man|woman|men|women|male|female|gender)\b"
    ]
    if not any(re.search(pat, norm) for pat in gender_cues):
        return False

    # Exclude document/certificate questions
    if any(w in norm for w in ["certificate", "pdf", "document", "recipient", "marks", "score", "grade", "roll number"]):
        return False

    return True


def answer_gender_person_query(question: str, evidence: list[RetrievedEvidence]) -> tuple[str, bool]:
    """
    Generate standardized response for gender questions regarding detected persons.
    Does not hallucinate or guess gender.
    """
    counts = extract_detection_counts_from_evidence(evidence)
    person_count = counts.get("person", 0)

    if person_count > 0:
        return "A person is detected in the image, but the system does not determine whether the person is a boy or girl.", True
    else:
        return "No person was detected in the image.", True


def is_direct_object_detection_query(question: str) -> bool:
    """
    Check if query can and must be answered deterministically from YOLO metadata.
    Excludes multimodal questions, text questions, and complex semantic vision queries.
    """
    norm = normalize_query_text(question)

    # 0. Check gender person questions (e.g. "Is there a boy or girl in the image?")
    if is_gender_person_query(question):
        return True
    
    # 1. Explicit exclusions for document/text/multimodal questions
    multimodal_or_text_exclusions = [
        "document", "pdf", "text", "file", "policy", "rules", "certificate",
        "manual", "report", "statement", "terms", "article", "section",
        "paragraph", "say about", "says about", "according to", "written in",
        "information", "provide about", "who is", "what is abs", "topic",
        "grade", "marks", "roll number", "summary", "summarize",
    ]
    for exc in multimodal_or_text_exclusions:
        if exc in norm:
            return False

    # 2. Explicit exclusions that require semantic vision/language reasoning
    semantic_exclusions = [
        "describe", "what is happening", "what is the person doing",
        "what color", "what is written", "read text", "ocr",
        "why", "scene", "street scene", "wearing", "background", "view",
    ]
    for exc in semantic_exclusions:
        if re.search(rf"\b{re.escape(exc)}\b", norm):
            return False

    # 3. Direct Detection & Count Patterns
    count_patterns = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\btotal (number|count|quantity)?\b",
        r"\bare there (any\s+)?",
        r"\bis there (a\s+|an\s+|any\s+)?",
        r"\bdo we have (any\s+)?",
        r"\bcan you find (a\s+|an\s+|any\s+)?",
        r"\bwhat (objects?|vehicles?|items?|entities?)\b",
        r"\bwhich (vehicles?|objects?|items?|entities?)\b",
        r"\blist (all\s+|the\s+)?(detected\s+|present\s+)?(objects?|vehicles?|items?|entities?)\b",
        r"\bdetected (objects?|vehicles?|items?|entities?)\b",
        r"\bwhat is detected\b",
        r"\bwhat was detected\b",
    ]

    has_pattern = any(re.search(pat, norm) for pat in count_patterns)

    # 4. Check for any recognized object, vehicle, or COCO entity
    has_obj = any(
        any(re.search(rf"\b{re.escape(syn)}\b", norm) for syn in syn_list)
        for syn_list in OBJECT_SYNONYMS.values()
    ) or any(
        re.search(rf"\b{re.escape(coco_cls)}\b", norm)
        for coco_cls in ALL_COCO_CLASSES
    ) or any(w in norm.split() for w in ["vehicle", "vehicles", "object", "objects", "item", "items", "entity", "entities"])
    
    has_count_cue = any(
        w in norm.split()
        for w in ["how", "many", "count", "number", "total", "detected", "detected?", "present", "are", "is", "there", "any"]
    )
    
    if has_pattern and (has_obj or any(w in norm.split() for w in ["detected", "present", "visible", "image", "photo", "picture"])):
        return True

    if has_obj and has_count_cue:
        return True

    return False


def extract_detection_counts_from_evidence(evidence: list[RetrievedEvidence]) -> dict[str, int]:
    """Aggregate detection counts from all object_detection and image evidence records."""
    aggregated: dict[str, int] = {}
    for ev in evidence:
        extra = getattr(ev, "extra", {}) or {}
        counts = extra.get("detection_counts")
        if counts and isinstance(counts, dict):
            for k, v in counts.items():
                aggregated[k.lower()] = max(aggregated.get(k.lower(), 0), int(v))
        elif extra.get("detections"):
            from collections import Counter
            local = Counter(d.get("label", "").lower() for d in extra.get("detections", []) if d.get("label"))
            for k, v in local.items():
                aggregated[k] = max(aggregated.get(k, 0), v)
    return aggregated


def answer_detection_query(question: str, evidence: list[RetrievedEvidence]) -> tuple[str, bool]:
    """
    Generate a deterministic answer for an object detection query.
    Returns (answer_text, is_success).
    """
    # 0. Check gender person questions
    if is_gender_person_query(question):
        return answer_gender_person_query(question, evidence)

    counts = extract_detection_counts_from_evidence(evidence)
    norm = normalize_query_text(question)

    if not evidence and not counts:
        return "No indexed image with object detection evidence was found.", False

    # 1. Check if user is asking for "all detected objects" or "which objects"
    all_objects_patterns = [
        r"\bwhat (objects?|items?) (were|are|is|have been)?\s*(detected|present|found)\b",
        r"\blist (all |the )?(detected |present )?(objects?|items?)\b",
        r"\bwhat is detected\b",
        r"\bwhat was detected\b",
        r"\bdetected objects\b",
    ]
    if any(re.search(pat, norm) for pat in all_objects_patterns):
        if not counts:
            return "No objects were detected in the indexed image.", True
        items = [format_count_label(c, lbl) for lbl, c in sorted(counts.items()) if c > 0]
        if not items:
            return "No objects were detected above the confidence threshold.", True
        return f"Detected objects: {join_items(items)}.", True

    # 2. Check if user is asking for "vehicles" (aggregate: car, bus, truck, motorcycle, bicycle)
    vehicles_patterns = [
        r"\b(how many|count|number of) vehicles\b",
        r"\bwhich vehicles (are|were|is)?\s*(present|detected)\b",
        r"\bwhat vehicles (are|were|is)?\s*(present|detected)\b",
        r"\bany vehicles\b",
        r"\bare there (any )?vehicles\b",
        r"\blist (all |the )?(detected |present )?vehicles\b",
        r"\bvehicles (detected|count|present|here)\b",
    ]
    if any(re.search(pat, norm) for pat in vehicles_patterns):
        veh_counts = {lbl: c for lbl, c in counts.items() if lbl in VEHICLE_CLASSES and c > 0}
        total_veh = sum(veh_counts.values())
        if total_veh == 0:
            return "No vehicles were detected in the indexed image.", True
        items = [format_count_label(c, lbl) for lbl, c in sorted(veh_counts.items())]
        if "how many" in norm or "number of" in norm or "count" in norm:
            return f"{total_veh} vehicles were detected in the indexed image ({join_items(items)}).", True
        return f"The vehicles detected are: {join_items(items)} (total: {total_veh}).", True

    # 3. Detect specific object classes mentioned in the question
    matched_classes: list[str] = []
    for std_class, synonyms in OBJECT_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(syn)}\b", norm) for syn in synonyms):
            matched_classes.append(std_class)

    # If no specific recognized class from dictionary, check all keys in counts and all COCO classes (singular/plural)
    if not matched_classes:
        for k in counts.keys():
            if re.search(rf"\b{re.escape(k)}(?:s|es)?\b", norm):
                matched_classes.append(k)
        if not matched_classes:
            for coco_cls in ALL_COCO_CLASSES:
                if re.search(rf"\b{re.escape(coco_cls)}(?:s|es)?\b", norm):
                    matched_classes.append(coco_cls)

    # 4. If specific objects were matched
    if matched_classes:
        # Check if question is a presence check ("is there", "are there", "any", "do we have")
        is_presence_check = bool(re.search(r"\b(is there|are there|any|do we have)\b", norm)) and not bool(re.search(r"\bhow many\b", norm))

        if len(matched_classes) == 1:
            cls_name = matched_classes[0]
            count = counts.get(cls_name, 0)
            singular, plural = KNOWN_PLURALS.get(cls_name, (cls_name, f"{cls_name}s"))
            
            if is_presence_check:
                if count > 0:
                    formatted = format_count_label(count, cls_name)
                    return f"Yes. {formatted} {'was' if count == 1 else 'were'} detected in the indexed image.", True
                else:
                    return f"No {singular} was detected in the indexed image.", True
            else:
                # Count question ("how many cars here", "How many cars are there?", "cars count")
                if count > 0:
                    formatted = format_count_label(count, cls_name)
                    return f"{formatted} {'was' if count == 1 else 'were'} detected in the indexed image.", True
                else:
                    return f"0 {plural} were detected in the indexed image.", True
        else:
            # Multiple specific classes ("How many cars and buses are there?")
            parts = []
            for cls_name in matched_classes:
                c = counts.get(cls_name, 0)
                parts.append(format_count_label(c, cls_name))
            return f"{join_items(parts)} were detected in the indexed image.", True

    # Fallback for generic count questions if any objects exist
    if "how many" in norm or "count" in norm or "objects" in norm:
        items = [format_count_label(c, lbl) for lbl, c in sorted(counts.items()) if c > 0]
        if items:
            return f"Detected objects: {join_items(items)}.", True
        return "No objects were detected in the indexed image.", True

    return "", False

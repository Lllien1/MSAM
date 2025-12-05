"""Short-word defect dictionary for MVTec-AD.

This follows the PEFT spec in `sam3_peft.md` and keeps prompts concise
to fit SAM3's text encoder context length.
"""

mvtec_short_keywords = {
    "bottle": ["crack", "dent", "leak", "contamination", "broken"],
    "cable": ["twist", "bent", "wire", "cut", "poke"],
    "capsule": ["crack", "imprint", "scratch", "dent"],
    "carpet": ["hole", "metal", "thread", "color", "cut"],
    "grid": ["bent", "broken", "glue", "metal", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined", "contamination", "crack", "imprint", "scratch"],
    "screw": ["manipulated", "scratch", "thread"],
    "tile": ["crack", "glue", "gray", "oil", "rough"],
    "toothbrush": ["defective", "missing"],
    "transistor": ["bent", "cut", "damaged", "misplaced"],
    "wood": ["color", "hole", "liquid", "scratch"],
    "zipper": ["broken", "fabric", "rough", "split"],
}

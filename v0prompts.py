SCENE_SCHEMA = {
    "name": "scene_description",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["description", "setting", "people", "objects", "actions", "tags"],
        "properties": {
            "description": {
                "type": "string",
                "description": "Prose description of the segment; the main text users search against.",
            },
            "setting": {
                "type": "string",
                "description": "Where this takes place, e.g. 'supermarket checkout area'.",
            },
            "people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Each person visible, described by role/appearance, never by name.",
            },
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Salient objects visible in the segment.",
            },
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Discrete actions occurring across the frames.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short keywords for filtering.",
            },
        },
    },
}


OCR_SCHEMA = {
    "name": "ocr_reading",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["texts", "summary"],
        "properties": {
            "texts": {
                "type": "array",
                "description": "Each distinct piece of text visible across the frames.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "context"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text exactly as written, including case and punctuation.",
                        },
                        "context": {
                            "type": "string",
                            "description": "What this text is and where it appears, e.g. 'a label on a cupboard'.",
                        },
                    },
                },
            },
            "summary": {
                "type": "string",
                "description": "One sentence on what kind of text this segment contains.",
            },
        },
    },
}


OBJECT_SCHEMA = {
    "name": "object_reading",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["detections", "summary"],
        "properties": {
            "detections": {
                "type": "array",
                "description": "Each distinct object worth indexing.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object", "description", "context"],
                    "properties": {
                        "object": {
                            "type": "string",
                            "description": "Short name for the object, e.g. 'shopping cart'.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What it looks like: colour, material, size, state.",
                        },
                        "context": {
                            "type": "string",
                            "description": "What it is doing or being used for in this scene, and by whom.",
                        },
                    },
                },
            },
            "summary": {
                "type": "string",
                "description": "One sentence on what objects this segment contains.",
            },
        },
    },
}


PEOPLE_SCHEMA = {
    "name": "people_reading",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["people", "summary"],
        "properties": {
            "people": {
                "type": "array",
                "description": "One entry per distinct person visible.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["box_id", "appearance", "clothing", "role", "action"],
                    "properties": {
                        "box_id": {
                            "type": "integer",
                            "description": "The number drawn on this person's box. Use 0 if they had no box.",
                        },
                        "appearance": {
                            "type": "string",
                            "description": "Build, hair, and other lasting physical features. No names, no guesses at identity.",
                        },
                        "clothing": {
                            "type": "string",
                            "description": "Every visible garment with its colour - the most reliable way to recognise them again.",
                        },
                        "role": {
                            "type": "string",
                            "description": "Their apparent role, e.g. 'cashier', 'customer', 'passer-by'.",
                        },
                        "action": {
                            "type": "string",
                            "description": "What they do across these frames, including any movement.",
                        },
                    },
                },
            },
            "summary": {
                "type": "string",
                "description": "One sentence on who is present and what they are doing.",
            },
        },
    },
}



TEXT_PROMPT = """\
Read the text visible in these images. Red boxes mark regions where text was \
detected - treat them as hints, and also read any text they missed. Ignore the \
boxes themselves; they are an overlay, not part of the scene.

For every distinct piece of text, give it exactly as written, and say what it is \
and where it appears - for example "a label on a cupboard", "a street sign", "a \
timestamp burned into the footage", "a readout on a targeting display".

The images are consecutive frames from one segment of a video, so the same text \
may appear in several of them. Report each distinct piece of text once, not once \
per frame. If a piece of text changes between frames, report the versions \
separately and say so.

Transcribe only what you can actually read. Do not guess at text that is blurred \
or cut off, and do not describe the scene beyond what is needed to place the text."""


OBJECT_PROMPT = """\
Identify the objects in these images. Cyan boxes mark regions where something was \
detected - treat them as hints about where to look, and also include anything \
important they missed. The boxes are an overlay, not part of the scene, and they \
are deliberately unlabelled: identify each object yourself rather than assuming \
what the box was drawn around.

For each distinct object give:
- what it is, as a short name
- what it looks like: colour, material, size, condition
- what it is doing or being used for here, and by whom

The images are consecutive frames from one segment of a video, so the same object \
may appear in several of them. Report each object once, not once per frame, and if \
its use changes across the frames say so.

Report objects that matter to understanding the scene. Skip incidental background \
clutter, and do not guess at things you cannot actually make out."""

PERSON_PROMPT = """\
Describe each person in these images. Yellow boxes are drawn around detected \
people and numbered - use those numbers to say which person you are describing. \
The boxes are an overlay, not part of the scene. If someone is clearly visible but \
has no box, describe them with box_id 0.

For each person give:
- appearance: build, hair, and other features that persist
- clothing: every visible garment and its colour, in as much detail as you can
- role: what they appear to be doing there, e.g. cashier, customer, passer-by
- action: what they do across these frames, including any movement

Clothing and appearance are what will be used to recognise the same person later \
in the video, so be specific and consistent: prefer "man in a white t-shirt and \
yellow shorts" over "customer". Never name anyone or guess who they are.

The images are consecutive frames from one segment, so the same person appears in \
several. Describe each person once, not once per frame."""

SCENE_PROMPT = """\
Describe this scene for a video search index. Cover: who and what is visible, the \
setting, and what is happening. Be specific and factual - this description is what \
users will search against.

The images are frames sampled in chronological order from a single continuous segment \
of one video. They are not separate scenes - treat them as one moment unfolding over \
time, and describe any movement or change across them (for example, someone moving \
from one place to another, or something appearing or disappearing).

Describe only what is actually visible. Do not identify individuals by name, and do \
not infer motion you cannot see across the frames."""

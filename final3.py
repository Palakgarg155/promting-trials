import torch
torch.cuda.empty_cache()

import re
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic_structure import FinalResponse, SymptomCodeEnum, ProblemCodeEnum, CauseCodeEnum, ResolutionCodeEnum

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\^\^\^.*?\^\^\^', '', text, flags=re.DOTALL)
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r'\bCHU\b|\bCHUDATA\b|\bPICHUDATA\b', '', text, flags=re.IGNORECASE)
    date_pattern = r'\b(?:(?:\d{1,2}[-/thstndrd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|...|Dec(?:ember)?)?[-/\s]*(?:\d{1,2})?[-/\s]*\d{2,4})\b'
    text = re.sub(date_pattern, '', text)

    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

def extract_information(llm, sampling_params, symptoms: str, chunk: str):
    prompt = f"""
You are a medical device diagnostic assistant. Extract the following from the inputs:

- Symptom: What is the observable symptom?
- Problem: What is the technical issue?
- Cause: What caused the issue?
- Resolution: How was it resolved?

Also include the relevant text from chunk (label it as "Retrieved Chunk").

Cleaned Symptom History:
{symptoms}

Cleaned Retrieved Chunk:
{chunk}

Respond in the following format exactly:
Symptom: ...
Problem: ...
Cause: ...
Resolution: ...
Retrieved Chunk: ...
"""

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    print(f"[LLM Extracted Info]\n{response}")
    return response

import json
import logging
from typing import Dict

def classify_information_direct(llm, sampling_params, extracted_info: str):
    """Use a direct approach without guided decoding to get classification."""
    
    # Create bullet point strings from Enum values
    def format_bullets(enum_cls):
        return "\n".join([f'- "{e.value}"' for e in enum_cls])

    symptom_bullets = format_bullets(SymptomCodeEnum)
    problem_bullets = format_bullets(ProblemCodeEnum)
    cause_bullets = format_bullets(CauseCodeEnum)
    resolution_bullets = format_bullets(ResolutionCodeEnum)

    prompt = f"""
You are a classification expert for SPCR codes used in medical equipment diagnostics.

Given the following extracted fields:
{extracted_info}

Classify each element using ONLY the EXACT strings from the predefined code values below.
You must use the EXACT string values within quotes - do not use code identifiers like S3023.

### Symptom Codes (use EXACTLY one of these strings):
{symptom_bullets}

### Problem Codes (use EXACTLY one of these strings):
{problem_bullets}

### Cause Codes (use EXACTLY one of these strings):
{cause_bullets}

### Resolution Codes (use EXACTLY one of these strings):
{resolution_bullets}

You MUST respond with COMPLETE and VALID JSON using EXACTLY this format:

{{
  "symptom_code": "<copy exact string from symptom codes list>",
  "problem_code": "<copy exact string from problem codes list>",
  "cause_code": "<copy exact string from cause codes list>",
  "resolution_code": "<copy exact string from resolution codes list>",
  "justification": {{
    "symptom_code": "<your reason for this choice>",
    "problem_code": "<your reason for this choice>",
    "cause_code": "<your reason for this choice>",
    "resolution_code": "<your reason for this choice>"
  }}
}}

IMPORTANT: USE THE EXACT STRING VALUES FROM THE LISTS ABOVE. DO NOT USE CODES LIKE "S3023" or "P3007".
It is critical that your entire response is properly formatted JSON with no truncation.
"""

    try:
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        response = response.replace("```json", "").replace("```", "").strip()
        logging.info(f"[LLM Classification Output]\n{response}")

        parsed = json.loads(response)

        # Map field names to their corresponding Enum classes
        enum_map: Dict[str, type] = {
            "symptom_code": SymptomCodeEnum,
            "problem_code": ProblemCodeEnum,
            "cause_code": CauseCodeEnum,
            "resolution_code": ResolutionCodeEnum,
        }

        # Convert string values to Enums
        for field, enum_cls in enum_map.items():
            parsed[field] = enum_cls(parsed[field])

        # Validate via Pydantic model
        validated = FinalResponse(**parsed)

        return {
            "validated_output": {
                field: {
                    "name": getattr(validated, field).name,
                    "value": getattr(validated, field).value
                }
                for field in enum_map
            },
            "justification": parsed.get("justification", {})
        }

    except Exception as e:
        return {
            "error": f"Validation failed: {str(e)}",
            "raw_output": response if 'response' in locals() else None
        }

def main():
    # Initialize LLM
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    # Sampling parameters - using even lower temperature for more deterministic outputs
    sampling_params = SamplingParams(
        temperature=0.1,  # Very low temperature for highly deterministic outputs
        top_p=0.95,
        max_tokens=1024,  # Ensure enough tokens for complete responses
        stop=None         # Don't use any stop tokens to ensure complete generation
    )
    
    sample_symptoms = """
        : "^^^ 1-6KFOUS |Lead Medical Dealer |Field Support | 22-Jan-2024 ^^^
Ventilator power supply issue.
***************************************************************************************************"	"^^^ 1-FSARAS | |Field Support  | 04-Dec-2024 ^^^
Customer reported problem "Ventilator power supply issue" cannot be reproduced and the problem observed was blank display. Unable to use the ventilator. Problem occurred during pre-use checkout and no patient was involved."	"^^^ 1-6KFOUS |Lead Medical Dealer |Field Support | 22-Jan-2024 ^^^
Problem Solution:-; Patient Impact:-Quality Contact:Tom; Failure Date:;  Patient Impact/Outcome:; Patient Consequence:; Intervention Required:; Additional Details:;
***************************************************************************************************
***************************************************************************************************
^^^ 1-SHIDJKA | |Field Support | 04-Feb-2024 ^^^
Problem Solution:-Have contacted the customer via phone we understand display was blank. We advised the customer to check cable which connect motherboard to display. Checked and found the display cameup. The machine ventilates in all modes. The machine is working in good condition. The machine model is CASDPASFASWF and the serial no is OJASD08986966.; Patient Impact:-Quality Contact:Pradeep; Failure Date:12/01/2024 06:14:07;  Patient Impact/Outcome:No Patient Involved; Patient Consequence:Problem occurred during pre-use checkout and no patient was involved.; Intervention Required:NA; Additional Details:NA;"	"^^^ 1-9809328102398 | |Field Support  | 04-FEB-2024 ^^^
The service Manual part no is 09834027831-001 Rev F. We advised the customer to check cable which connect motherboard to display. Checked and found the display cameup. The machine works in all modes. The machine is working in good condition.  Customer also checked and confirmed that machine is working successfully.; Test Passed:Y	SPCR Task: 11-FEB-2024 - Vijay Gore forwarded case to Engineering for Review."

    """

    sample_chunk = "Display is showing artifact lines and freezing intermittently"

    # Step 1: Clean both
    clean_symptoms = clean_text(sample_symptoms)
    clean_chunk = clean_text(sample_chunk)

    # Step 2: Extraction
    extracted = extract_information(llm, sampling_params, clean_symptoms, clean_chunk)

    # Step 3: Classification - using direct approach instead of guided decoding
    classified = classify_information_direct(llm, sampling_params, extracted)

    # Final Output
    print("\nFinal Structured Classification Result:")
    print(json.dumps(classified, indent=4))


if __name__ == "__main__":
    main()
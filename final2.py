import torch
torch.cuda.empty_cache()

import re
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic_structure import FinalResponse, SymptomCodeEnum, ProblemCodeEnum, CauseCodeEnum, ResolutionCodeEnum

# Initialize guided decoding
guided_decoding_params = GuidedDecodingParams(json=FinalResponse.model_json_schema())

# Initialize LLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    guided_decoding=None  # Used only in the second call
)

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

def extract_information(symptoms: str, chunk: str):
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

def classify_information(extracted_info: str):
    symptom_codes = "\n".join([f"- {code.name}: {code.value}" for code in SymptomCodeEnum])
    problem_codes = "\n".join([f"- {code.name}: {code.value}" for code in ProblemCodeEnum])
    cause_codes = "\n".join([f"- {code.name}: {code.value}" for code in CauseCodeEnum])
    resolution_codes = "\n".join([f"- {code.name}: {code.value}" for code in ResolutionCodeEnum])

    prompt = f"""
You are a classification expert for SPCR codes used in medical equipment diagnostics.

Given the following extracted fields:
{extracted_info}

Classify each element using ONLY the predefined code values below:

### Symptom Codes:
{symptom_codes}

### Problem Codes:
{problem_codes}

### Cause Codes:
{cause_codes}

### Resolution Codes:
{resolution_codes}

Return your answer in **valid JSON** using this format:
{{
  "symptom_code": "code_value",
  "problem_code": "code_value",
  "cause_code": "code_value",
  "resolution_code": "code_value",
  "justification": {{
    "symptom_code": "reason for choice",
    "problem_code": "reason for choice",
    "cause_code": "reason for choice",
    "resolution_code": "reason for choice"
  }}
}}
"""

    # Classification uses guided decoding to ensure proper JSON schema
    guided_sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=guided_decoding_params
    )

    outputs = llm.generate([prompt], guided_sampling_params)
    json_response = outputs[0].outputs[0].text.strip()
    print(f"[LLM Classification Output]\n{json_response}")

    try:
        parsed = json.loads(json_response)
        validated = FinalResponse(**parsed)
        return {
            "validated_output": {
                "symptom_code": {"name": validated.symptom_code.name, "value": validated.symptom_code.value},
                "problem_code": {"name": validated.problem_code.name, "value": validated.problem_code.value},
                "cause_code": {"name": validated.cause_code.name, "value": validated.cause_code.value},
                "resolution_code": {"name": validated.resolution_code.name, "value": validated.resolution_code.value}
            },
            "justification": parsed.get("justification", {})
        }
    except Exception as e:
        return {
            "error": f"Validation failed: {str(e)}",
            "raw_output": json_response
        }

def main():
    sample_symptoms = """^^^ 1-6KFOUS |Lead Medical Dealer |Field Support | 22-Jan-2024 ^^^
Ventilator power supply issue.^^^ 1-FSARAS | |Field Support  | 04-Dec-2024 ^^^
Customer reported problem 'Ventilator power supply issue' cannot be reproduced and the problem observed was blank display. Unable to use the ventilator..."""

    sample_chunk = "Display is showing artifact lines and freezing intermittently"

    # Step 1: Clean both
    clean_symptoms = clean_text(sample_symptoms)
    clean_chunk = clean_text(sample_chunk)

    # Step 2: Extraction
    extracted = extract_information(clean_symptoms, clean_chunk)

    # Step 3: Classification
    classified = classify_information(extracted)

    # Final Output
    print("\nFinal Structured Classification Result:")
    print(json.dumps(classified, indent=4))


if __name__ == "__main__":
    main()

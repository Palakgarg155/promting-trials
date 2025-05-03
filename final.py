import torch
torch.cuda.empty_cache()  # Uncomment if needed to clear unused memory
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
from pydantic_structure import FinalResponse, SymptomCodeEnum, ProblemCodeEnum, CauseCodeEnum, ResolutionCodeEnum

# Initialize guided decoding with the schema
guided_decoding_params = GuidedDecodingParams(json=FinalResponse.model_json_schema())

# Initialize LLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    guided_decoding=guided_decoding_params
)

def clean_text(text: str) -> str:
    """Clean and preprocess text by removing irrelevant patterns and duplicates."""
    if not text:
        return ""
        
    # Remove text enclosed in ^^^ ... ^^^
    text = re.sub(r'\^\^\^.*?\^\^\^', '', text, flags=re.DOTALL)
    
    # Remove asterisks and specific terms
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r'\bCHU\b|\bCHUDATA\b|\bPICHUDATA\b', '', text, flags=re.IGNORECASE)
    
    # Remove dates
    date_pattern = r'\b(?:(?:\d{1,2}[-/thstndrd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?[-/\s]*(?:\d{1,2})?[-/\s]*\d{2,4})\b'
    text = re.sub(date_pattern, '', text)
    
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_sentences.append(sentence)
            
    # Rejoin unique sentences
    return ' '.join(unique_sentences)

def process_with_single_llm_call(symptoms, chunk):
    """Process text and classify it with a single LLM call."""
    # Clean the text first
    clean_text_content = clean_text(chunk)
    
    # Create formatted strings with all available enum options
    symptom_codes = "\n".join([f"- {code.name}: {code.value}" for code in SymptomCodeEnum])
    problem_codes = "\n".join([f"- {code.name}: {code.value}" for code in ProblemCodeEnum])
    cause_codes = "\n".join([f"- {code.name}: {code.value}" for code in CauseCodeEnum])
    resolution_codes = "\n".join([f"- {code.name}: {code.value}" for code in ResolutionCodeEnum])

    # Combined prompt that handles both extraction and classification
    combined_prompt = f"""
    You are an expert diagnostic AI assisting in troubleshooting medical equipment issues.
    
    ### Step 1: Extract key elements from the raw text below:
    - Extract the Symptom (observable manifestation or error message)
    - Extract the Problem (underlying technical issue)
    - Extract the Cause (root cause of the issue)
    - Extract the Resolution (solution implemented)
    
    Raw Text: "{clean_text_content}"
    
    ### Step 2: Based on your extraction, classify each element using ONLY these predefined codes:
    
    ### Available Symptom Codes (CHOOSE ONE ONLY):
    {symptom_codes}
    
    ### Available Problem Codes (CHOOSE ONE ONLY):
    {problem_codes}
    
    ### Available Cause Codes (CHOOSE ONE ONLY):
    {cause_codes}
    
    ### Available Resolution Codes (CHOOSE ONE ONLY):
    {resolution_codes}
    
    ### Additional Context:
    - **Provided Symptoms:** "{symptoms}"
    
    Respond ONLY in a structured JSON format using the exact code values from the lists above. Your response must match the schema exactly.
    """
    
    try:
        # Generate response using guided decoding
        outputs = llm.generate([combined_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Parse the JSON response
        parsed_response = json.loads(generated_text)
        print(f"Parsed Response: {parsed_response}")
        
        # Validate with Pydantic model
        validated_response = FinalResponse(**parsed_response)
        print(f"Validated Response: {validated_response}")
        
        return {
            "symptom_code": {
                "name": validated_response.symptom_code.name, 
                "value": validated_response.symptom_code.value
            },
            "problem_code": {
                "name": validated_response.problem_code.name, 
                "value": validated_response.problem_code.value
            },
            "cause_code": {
                "name": validated_response.cause_code.name, 
                "value": validated_response.cause_code.value
            }, 
            "resolution_code": {
                "name": validated_response.resolution_code.name,
                "value": validated_response.resolution_code.value
            }
        }
    
    except Exception as e:
        return {
            "error": f"Parsing failed: {str(e)}", 
            "raw_output": generated_text if 'generated_text' in locals() else "No output generated"
        }


def main():
    """Test function to demonstrate the classification pipeline with sample data."""
    # Sample test data
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
    
    print("Testing process_with_single_llm_call function...")
    print("=" * 80)
    print(f"Sample symptoms: {sample_symptoms}")
    print("-" * 80)
    print(f"Sample chunk: {sample_chunk}")
    print("-" * 80)
    
    # Test the clean_text function separately
    cleaned_text = clean_text(sample_chunk)
    print(f"Cleaned text: {cleaned_text}")
    print("-" * 80)
    
    # Run the optimized single-call pipeline
    try:
        results = process_with_single_llm_call(sample_symptoms, sample_chunk)
        print("Classification Results:")
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
# import torch
# torch.cuda.empty_cache()  # Uncomment if needed to clear unused memory
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
from generation_component.pydantic_structure import FinalResponse, SymptomCodeEnum, ProblemCodeEnum, CauseCodeEnum, ResolutionCodeEnum

# Initialize guided decoding with the schema
guided_decoding_params = GuidedDecodingParams(json=FinalResponse.model_json_schema())

# Initialize LLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    max_tokens=100,
    guided_decoding=guided_decoding_params
)

def classify_with_llm(symptoms, chunk):
    """Use LLM to classify the root cause and resolution based on provided symptoms and manual section."""
    
    # Create formatted strings with all available enum options
    symptom_codes = "\n".join([f"- {code.name}: {code.value}" for code in SymptomCodeEnum])
    problem_codes = "\n".join([f"- {code.name}: {code.value}" for code in ProblemCodeEnum])
    cause_codes = "\n".join([f"- {code.name}: {code.value}" for code in CauseCodeEnum])
    resolution_codes = "\n".join([f"- {code.name}: {code.value}" for code in ResolutionCodeEnum])
    
    prompt = f"""
    You are an expert diagnostic AI assisting in troubleshooting medical equipment issues. 
    Given a reported symptom and relevant troubleshooting manual section, determine:
    
    - **Symptom Code:** The code representing the symptoms observed by the customer or by the service engineer initially. These are the externally visible issues or error conditions that prompted the service request. YOU MUST SELECT FROM THE PREDEFINED SYMPTOM CODES LISTED BELOW.
    
    - **Problem Code:** The code representing the underlying technical issue that is causing the observed symptoms. This identifies what is actually malfunctioning in the system. YOU MUST SELECT FROM THE PREDEFINED PROBLEM CODES LISTED BELOW.
    
    - **Cause Code:** The most appropriate predefined cause code that identifies the specific component or condition responsible for the problem. YOU MUST SELECT FROM THE PREDEFINED CAUSE CODES LISTED BELOW.
    
    - **Resolution Code:** The best matching predefined resolution code that describes the action taken to resolve the issue. YOU MUST SELECT FROM THE PREDEFINED RESOLUTION CODES LISTED BELOW.
    
    IMPORTANT: For all classifications, you MUST use ONLY the exact codes provided below. Do not create new codes or modify the existing ones.
    
    ### Available Symptom Codes (CHOOSE ONE ONLY):
    {symptom_codes}
    
    ### Available Problem Codes (CHOOSE ONE ONLY):
    {problem_codes}
    
    ### Available Cause Codes (CHOOSE ONE ONLY):
    {cause_codes}
    
    ### Available Resolution Codes (CHOOSE ONE ONLY):
    {resolution_codes}
    
    ### Provided Data:
    - **Symptom:** "{symptoms}"
    - **Manual Section:** "{chunk}"
    
    Respond ONLY in a structured JSON format using the exact code values from the lists above. Your response must match the schema exactly.
    """
    
    try:
        # Generate response using guided decoding
        outputs = llm.generate([prompt], sampling_params)
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


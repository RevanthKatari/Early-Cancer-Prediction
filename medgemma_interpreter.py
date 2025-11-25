from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import streamlit as st
from functools import lru_cache

# Try to import transformers, but make it optional
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

# MedGemma model identifier - using Gemma 2B as base (MedGemma 4B may need specific model ID)
# You can replace this with the actual MedGemma 4B model ID when available
MEDGEMMA_MODEL_ID = "google/gemma-2-2b-it"  # Fallback to Gemma 2B IT model
# Alternative options:
# - "google/gemma-2-9b-it" for larger model
# - "google/medgemma-4b" if available on HuggingFace


@lru_cache(maxsize=1)
def _load_medgemma():
    """Load MedGemma model and tokenizer with caching."""
    if not TRANSFORMERS_AVAILABLE or torch is None:
        return None, None
    
    try:
        # Use device_map="auto" for automatic GPU/CPU allocation
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        torch_dtype = torch.float16 if has_cuda else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(MEDGEMMA_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MEDGEMMA_MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto" if has_cuda else None,
            low_cpu_mem_usage=True,
        )
        
        if not has_cuda:
            model = model.to("cpu")
        
        return model, tokenizer
    except Exception as e:
        try:
            st.warning(f"Could not load MedGemma model: {e}. Using text-based interpretation instead.")
        except:
            pass  # If streamlit is not available, just continue
        return None, None


def _format_questionnaire_data(questionnaire_data: Dict[str, float], fieldset) -> str:
    """Format questionnaire data into a readable string."""
    sections = {}
    for field in fieldset:
        if field.section not in sections:
            sections[field.section] = []
        value = questionnaire_data.get(field.key, 0.0)
        if field.input_type == "binary":
            display_value = "Yes" if value == 1.0 else "No"
        else:
            display_value = str(value)
        sections[field.section].append(f"- {field.label}: {display_value}")
    
    formatted = []
    for section, items in sections.items():
        formatted.append(f"\n{section}:")
        formatted.extend(items)
    
    return "\n".join(formatted)


def _generate_interpretation(
    prompt: str,
    model=None,
    tokenizer=None,
    max_length: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate interpretation using MedGemma model."""
    if model is None or tokenizer is None:
        # Fallback to a simple rule-based interpretation
        return _fallback_interpretation(prompt)
    
    try:
        # Format the prompt for the model
        formatted_prompt = f"""You are a medical AI assistant. Please provide a clear, patient-friendly interpretation of the following medical data:

{prompt}

Please explain:
1. What the results mean in simple terms
2. What factors may have contributed to these results
3. General recommendations (note: this is not medical advice)

Interpretation:"""
        
        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch is not None and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the interpretation part (after "Interpretation:")
        if "Interpretation:" in response:
            response = response.split("Interpretation:")[-1].strip()
        
        return response
    except Exception as e:
        st.warning(f"Error generating interpretation: {e}. Using fallback.")
        return _fallback_interpretation(prompt)


def _fallback_interpretation(prompt: str) -> str:
    """Simple fallback interpretation when model is not available."""
    return """**Interpretation:**

This is a preliminary AI-assisted analysis. The results shown are based on machine learning models trained on medical data.

**Important Notes:**
- These results are for informational purposes only and should not replace professional medical consultation
- Always consult with a qualified healthcare provider for diagnosis and treatment decisions
- The model predictions are based on patterns in training data and may not account for all individual factors

**Next Steps:**
- Review the results with your healthcare provider
- Discuss any concerns or questions you may have
- Follow up with recommended screenings or tests as advised by your doctor"""


def interpret_brain_results(
    prediction_label: str,
    probabilities: List[float],
    confidence: float,
    questionnaire_data: Optional[Dict[str, float]] = None,
    fieldset=None,
) -> str:
    """Generate interpretation for brain MRI results."""
    model, tokenizer = _load_medgemma()
    
    # Format the prediction data
    brain_classes = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
    prob_text = "\n".join([
        f"- {cls}: {prob*100:.1f}%" 
        for cls, prob in zip(brain_classes, probabilities)
    ])
    
    prompt = f"""Brain MRI Analysis Results:

**Prediction:** {prediction_label.title()}
**Confidence:** {confidence*100:.1f}%

**Probability Distribution:**
{prob_text}
"""
    
    if questionnaire_data and fieldset:
        q_data = _format_questionnaire_data(questionnaire_data, fieldset)
        prompt += f"\n**Risk Factor Questionnaire:**{q_data}"
    
    return _generate_interpretation(prompt, model, tokenizer)


def interpret_oral_results(
    prediction_label: str,
    probabilities: List[float],
    confidence: float,
    questionnaire_data: Optional[Dict[str, float]] = None,
    fieldset=None,
) -> str:
    """Generate interpretation for oral cancer screening results."""
    model, tokenizer = _load_medgemma()
    
    prob_text = "\n".join([
        f"- {label}: {prob*100:.1f}%" 
        for label, prob in zip(["Non-cancer", "Cancer"], probabilities)
    ])
    
    prompt = f"""Oral Cancer Screening Results:

**Prediction:** {prediction_label}
**Confidence:** {confidence*100:.1f}%

**Probability Distribution:**
{prob_text}
"""
    
    if questionnaire_data and fieldset:
        q_data = _format_questionnaire_data(questionnaire_data, fieldset)
        prompt += f"\n**Risk Factor Questionnaire:**{q_data}"
    
    return _generate_interpretation(prompt, model, tokenizer)


def interpret_cervical_results(
    risk_probability: float,
    probabilities: List[float],
    questionnaire_data: Dict[str, float],
    fieldset,
) -> str:
    """Generate interpretation for cervical cancer risk assessment."""
    model, tokenizer = _load_medgemma()
    
    risk_label = "High risk" if risk_probability >= 0.65 else "Moderate risk" if risk_probability >= 0.35 else "Low risk"
    
    prob_text = "\n".join([
        f"- {label}: {prob*100:.1f}%" 
        for label, prob in zip(["Negative", "Positive"], probabilities)
    ])
    
    q_data = _format_questionnaire_data(questionnaire_data, fieldset)
    
    prompt = f"""Cervical Cancer Risk Assessment Results:

**Risk Level:** {risk_label}
**Biopsy Positive Probability:** {risk_probability*100:.1f}%

**Probability Distribution:**
{prob_text}

**Risk Factor Questionnaire:**
{q_data}
"""
    
    return _generate_interpretation(prompt, model, tokenizer)


def interpret_questionnaire_only(
    questionnaire_data: Dict[str, float],
    fieldset,
    cancer_type: str,
) -> str:
    """Generate interpretation for questionnaire data only (no model prediction)."""
    model, tokenizer = _load_medgemma()
    
    q_data = _format_questionnaire_data(questionnaire_data, fieldset)
    
    prompt = f"""{cancer_type} Risk Factor Assessment:

**Questionnaire Responses:**
{q_data}

Please provide an assessment of the risk factors based on the provided information.
"""
    
    return _generate_interpretation(prompt, model, tokenizer)


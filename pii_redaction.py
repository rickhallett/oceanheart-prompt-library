import marimo as mo
import os
import json
import llm
from dotenv import load_dotenv

# Try specific SDK imports - add others as needed (anthropic, google.generativeai, etc.)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

__generated_with = "0.8.18"  # Or your Marimo version
app = mo.App(width="full")


@app.cell
def load_env_vars():
    load_dotenv()
    # Example: Ensure OPENAI_API_KEY is loaded if using OpenAI
    # Add checks for other keys as needed
    if openai:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    if anthropic:
        anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
    # Return the modules for use in other cells
    return os, json, llm, load_dotenv, openai, anthropic


@app.cell
def define_pii_categories():
    # Define PII categories based on HIPAA/GDPR
    # Refine this list based on specific needs
    pii_categories_list = [
        "PATIENT_NAME",
        "CLINICIAN_NAME",
        "OTHER_NAME",  # Generic name if type is uncertain
        "DATE",
        "TIME",
        "LOCATION",  # Hospitals, clinics, wards
        "ADDRESS",  # Street addresses
        "CITY",
        "STATE",
        "ZIP_CODE",
        "PHONE_NUMBER",
        "FAX_NUMBER",
        "EMAIL",
        "URL",
        "IP_ADDRESS",
        "MEDICAL_RECORD_NUMBER",
        "HEALTH_PLAN_BENEFICIARY_NUMBER",
        "ACCOUNT_NUMBER",
        "CERTIFICATE_LICENSE_NUMBER",
        "VEHICLE_IDENTIFIER_SERIAL_NUMBER",
        "DEVICE_IDENTIFIER_SERIAL_NUMBER",
        "BIOMETRIC_IDENTIFIER",  # Fingerprints, voice prints
        "FULL_FACE_PHOTO",
        "OTHER_UNIQUE_IDENTIFYING_NUMBER_CHARACTERISTIC_CODE",
        "SSN",  # Social Security Number
        "AGE_OVER_89",  # Specific ages over 89
        "ANY_OTHER_POTENTIALLY_IDENTIFYING_INFO",
    ]
    pii_categories_str = ", ".join(pii_categories_list)
    return pii_categories_list, pii_categories_str


@app.cell
def get_available_models(llm):
    """Dynamically gets available models using the llm library."""
    try:
        models_with_aliases = llm.get_models_with_aliases()
        available_models_dict = {m.model.model_id: m.model for m in models_with_aliases}
        # Add aliases
        for m in models_with_aliases:
            for alias in m.aliases:
                available_models_dict[alias] = m.model

        # Default model logic (try default, then first available)
        default_llm_model_id = llm.get_default_model_id()
        if default_llm_model_id not in available_models_dict:
            # Fallback to the first model ID if default is not found or None
            default_llm_model_id = next(iter(available_models_dict.keys()), None)

    except Exception as e:
        print(f"Error getting models: {e}")  # Log error
        available_models_dict = {"error": "Could not load models"}
        default_llm_model_id = "error"

    return available_models_dict, default_llm_model_id


@app.cell
def ui_elements(mo, pii_categories_str, get_available_models):
    available_models_dict, default_llm_model_id = get_available_models

    transcript_input = mo.ui.text_area(
        label="Paste Clinical Transcript Here",
        full_width=True,
        line_numbers=True,
        rows=15,
    )

    model_selector = mo.ui.dropdown(
        options=available_models_dict, value=default_llm_model_id, label="Select LLM"
    )

    initial_prompt_template = f"""
You are an expert in HIPAA and GDPR compliance. Your task is to redact specified Personally Identifiable Information (PII) from the provided clinical transcript.

PII Categories to redact: {pii_categories_str}

Instructions:
1. Identify all instances of the specified PII categories in the transcript.
2. For each identified PII instance, assign a confidence score (0.0 to 1.0) indicating your certainty that it is indeed PII of that category.
3. Replace each identified PII instance *in the transcript text* with a placeholder formatted as `[REDACTED_{{{{CATEGORY}}}}]`, where `{{{{CATEGORY}}}}` is the uppercase PII category (e.g., `[REDACTED_PATIENT_NAME]`, `[REDACTED_DATE]`). Use `[REDACTED_UNKNOWN]` if the category cannot be determined but it is clearly PII.
4. Generate a JSON object containing ONLY the following two keys:
   - "redacted_transcript": The full transcript with PII replaced by placeholders.
   - "redactions": A list of JSON objects, where each object represents one redaction and MUST have the following keys:
     - "original_text": The exact text that was redacted.
     - "redaction_type": The specific PII category identified (e.g., "PATIENT_NAME", "DATE"). Use "UNKNOWN" if the category is unclear.
     - "confidence_score": Your confidence score (float between 0.0 and 1.0).
     - "start_index": The starting character index of the redacted text in the *original* transcript.
     - "end_index": The ending character index of the redacted text in the *original* transcript.

Ensure the output is ONLY the JSON object, enclosed in triple backticks (```json ... ```), with no introductory text, explanations, or apologies.

Transcript:
```
{{{{transcript}}}}
```

JSON Output:
"""
    prompt_editor = mo.ui.text_area(
        label="Edit Prompt Template",
        value=initial_prompt_template,
        full_width=True,
        line_numbers=True,
        rows=25,
    )

    # Display UI elements
    mo.md(
        f"""## 1. Input Transcript
    {transcript_input}
    
    ## 2. Select Model
    {model_selector}
    
    ## 3. Prompt Template (Iterate Here)
    Use {{{{transcript}}}} and {{{{pii_categories}}}} placeholders (though pii_categories is already included in the default template).
    Ensure JSON output instructions are clear.
    {prompt_editor}
    """
    )

    return transcript_input, model_selector, prompt_editor


@app.cell
def process_transcript(mo, json, llm, transcript_input, model_selector, prompt_editor):
    """Processes the transcript using the selected model and prompt."""
    transcript_text = transcript_input.value
    selected_model_obj = model_selector.value
    prompt_template = prompt_editor.value

    # Stop execution if no transcript is provided
    mo.stop(not transcript_text, "Please paste a transcript above.")
    mo.stop(
        not selected_model_obj or not isinstance(selected_model_obj, llm.Model),
        "Please select a valid model.",
    )
    mo.stop(not prompt_template, "Please provide a prompt template.")

    llm_response_data = None
    error_message = None

    formatted_prompt = prompt_template.format(transcript=transcript_text)

    with mo.status.spinner(title=f"Redacting using {selected_model_obj.model_id}..."):
        try:
            # Execute the prompt
            # Note: Temperature/other settings can be added if needed
            # Check if the model requires a specific API key setup if not using default
            if (
                not selected_model_obj.key
                and "ollama" not in selected_model_obj.model_id
            ):
                # Attempt to load key if missing (basic example)
                api_key_name = (
                    f"{selected_model_obj.model_id.split('/')[0].upper()}_API_KEY"
                )
                api_key = os.getenv(api_key_name)
                if api_key:
                    selected_model_obj.key = api_key
                else:
                    raise ValueError(
                        f"API key {api_key_name} not found in .env for model {selected_model_obj.model_id}"
                    )

            response = selected_model_obj.prompt(formatted_prompt)
            response_text = response.text()

            # Attempt to parse the JSON response
            try:
                # Find JSON block within backticks
                json_match = response_text.split("```json\n")
                if len(json_match) > 1:
                    json_str = json_match[1].split("\n```")[0]
                else:
                    # Fallback if no backticks are found
                    json_str = response_text.strip()

                llm_response_data = json.loads(json_str)

                # Basic validation of expected keys
                if (
                    not isinstance(llm_response_data, dict)
                    or "redacted_transcript" not in llm_response_data
                    or "redactions" not in llm_response_data
                ):
                    raise ValueError(
                        "LLM response JSON is missing required keys ('redacted_transcript', 'redactions')."
                    )
                if not isinstance(llm_response_data["redactions"], list):
                    raise ValueError("LLM response 'redactions' key is not a list.")

            except json.JSONDecodeError as e:
                error_message = f"Error parsing LLM response as JSON: {e}\nRaw response:\n```\n{response_text}\n```"
            except ValueError as e:
                error_message = f"Error in LLM response structure: {e}\nRaw response:\n```\n{response_text}\n```"

        except Exception as e:
            error_message = f"Error calling LLM API: {e}"

    return (
        llm_response_data,
        error_message,
        transcript_text,
    )  # Pass transcript_text for display reference


@app.cell
def display_results(mo, process_transcript):
    """Displays the redacted transcript and the list of detected PII."""
    llm_response_data, error_message, transcript_text = process_transcript

    # Display any errors first
    if error_message:
        mo.md(
            f"""**Error During Processing:**
        ```
        {error_message}
        ```
        """
        ).callout(kind="danger")
        return  # Stop further processing in this cell if there's an error

    # Check if data is available (it should be if no error, but good practice)
    if not llm_response_data:
        # This case should ideally be covered by the mo.stop in the previous cell
        # or the error handling, but added as a safeguard.
        mo.md("No results generated. Please check inputs and model selection.").callout(
            kind="warn"
        )
        return

    # Display Redacted Transcript
    redacted_transcript = llm_response_data.get(
        "redacted_transcript", "Error: Redacted transcript not found in response."
    )
    output_elements = [
        mo.md(
            f"""## 4. Redacted Transcript
        ```
        {redacted_transcript}
        ```
        """
        )
    ]

    # Display Redactions List
    redactions_list = llm_response_data.get("redactions")

    if isinstance(redactions_list, list) and redactions_list:
        # Using pandas for a nicer table if available, otherwise markdown
        try:
            import pandas as pd

            df = pd.DataFrame(redactions_list)
            # Ensure required columns exist, fill missing with 'N/A'
            required_cols = [
                "original_text",
                "redaction_type",
                "confidence_score",
                "start_index",
                "end_index",
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = "N/A"
            # Reorder columns for consistency
            df = df[required_cols]
            output_elements.append(mo.md("## 5. Detected PII"))
            output_elements.append(
                mo.ui.table(df, page_size=10, label="Detected PII Items")
            )
        except ImportError:
            # Fallback to Markdown table if pandas is not installed
            markdown_table = "| Original Text | Type | Confidence | Start | End |\n"
            markdown_table += "|---|---|---|---|---|\n"
            for item in redactions_list:
                conf_score = item.get("confidence_score", "N/A")
                try:
                    conf_str = (
                        f"{float(conf_score):.2f}"
                        if isinstance(conf_score, (float, int))
                        else str(conf_score)
                    )
                except (ValueError, TypeError):
                    conf_str = str(
                        conf_score
                    )  # Handle non-numeric confidence gracefully

                markdown_table += f"| {item.get('original_text','')} | {item.get('redaction_type','N/A')} | {conf_str} | {item.get('start_index','N/A')} | {item.get('end_index','N/A')} |\n"
            output_elements.append(
                mo.md(
                    f"""## 5. Detected PII
            {markdown_table}
            """
                )
            )
    elif isinstance(redactions_list, list):
        output_elements.append(
            mo.md("## 5. Detected PII\nNo PII detected by the model.")
        )
    else:
        output_elements.append(
            mo.md(
                "## 5. Detected PII\nWarning: 'redactions' field in response was not a list."
            ).callout(kind="warn")
        )

    # Combine all output elements into a vertical stack
    mo.vstack(output_elements)

    # Return values needed by other potential cells (optional)
    return redacted_transcript, redactions_list


@app.cell
def main():
    # This cell can trigger the main execution if needed,
    # or simply ensure the app runs when executed as a script.
    # Currently, reactivity handles the flow.
    # mo.status.loading()
    return


if __name__ == "__main__":
    app.run()

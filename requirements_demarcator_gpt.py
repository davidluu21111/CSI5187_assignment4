"""
Requirements Demarcator using GPT-4o-mini
CSI5187 Assignment 4

Uses GPT-4o-mini via OpenRouter API for requirements classification.
"""

import pandas as pd
import os
import json
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import requests

class RequirementsDemarcatorGPT:
    """Requirements Demarcator using GPT-4o-mini via OpenRouter API."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        """
        Initialize with OpenRouter API.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-4o-mini)

        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create optimized system prompt."""
        return """You are an expert in software requirements engineering with deep knowledge of IEEE 830, ISO/IEC/IEEE 29148, and requirements analysis best practices.

Your task is to classify statements from software requirements documents as either requirements ("Req") or non-requirements ("Not_Req").

WHAT MAKES A REQUIREMENT:
A requirement is a documented statement that specifies a mandatory constraint, capability, condition, or quality factor that a system or component must satisfy or possess. Requirements exhibit these characteristics:

1. Prescriptive Nature: The statement mandates what the system SHALL, MUST, or SHOULD do, not what it currently does or might do
2. Behavioral Specification: It defines specific system behavior, functionality, performance, or constraints
3. Testability: The statement can be verified through inspection, analysis, demonstration, or testing
4. Singular Focus: It addresses one specific aspect of system behavior or quality
5. Stakeholder Value: It captures a capability needed by users or stakeholders to solve a problem or achieve an objective

Common requirement patterns include:
- System behavior triggered by events or conditions ("Upon X, the system shall Y")
- Functional capabilities ("The system must provide the ability to...")
- Performance constraints ("Response time shall not exceed X seconds")
- Quality attributes with measurable criteria ("The system should maintain 99.9% uptime")
- Interface specifications ("The API shall accept requests in JSON format")
- Security and access control mandates ("Only authorized users shall access...")

WHAT IS NOT A REQUIREMENT:
Non-requirements are informational statements that provide context, background, or descriptive content but do not impose mandatory constraints on the system. These include:

1. Background Information: Historical context, project rationale, or domain knowledge
2. Architectural Descriptions: How the system is structured or organized (unless mandating specific constraints)
3. Design Decisions: Implementation approaches, technology choices, or design patterns (descriptive, not prescriptive)
4. User Actions: Steps users perform in workflows or use cases (unless specifying what system must enable)
5. Document Metadata: Version numbers, dates, authorship, references to other documents
6. Explanatory Text: Definitions, acronyms, assumptions, or clarifications
7. Current State Descriptions: What existing systems do or how processes currently work

Critical distinctions for ambiguous cases:
- "The user selects an option" = Not_Req (describes user action, not system obligation)
- "The system shall allow the user to select an option" = Req (mandates system capability)
- "The architecture consists of three tiers" = Not_Req (describes structure)
- "The system shall implement a three-tier architecture" = Req (mandates structural constraint)
- "Version 2.0 added support for XML" = Not_Req (historical fact)
- "The system shall support XML format" = Req (mandates capability)

Decision Framework:
Ask yourself: "Does this statement impose a verifiable obligation on what the system must do, provide, or comply with?" If yes, it's a requirement. If it merely describes, explains, or provides context without imposing obligation, it's not a requirement.

OUTPUT INSTRUCTION:
Respond with exactly one word: either "Req" if the statement is a requirement, or "Not_Req" if it is not a requirement. Provide no other text, explanation, or punctuation."""

    def _create_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Create few-shot examples selected from PURE_train.csv.

        These examples were carefully chosen to demonstrate:
        - Different requirement types (SHALL, MUST, SHOULD patterns)
        - Common non-requirement patterns (descriptions, architecture, metadata)
        - Ambiguous cases that help the model learn critical distinctions
        """
        return [
            # REQUIREMENTS - Show diverse modal verb patterns and requirement types
            {"text": "Once the audit trail functionality has been activated, the System must track events without manual intervention, and store in the audit trail information about them.", "label": "Req"},
            {"text": "The System must allow the user to limit access to cases to specified users or user groups.", "label": "Req"},
            {"text": "The solution should provide detailed context-sensitive help material for all the possible actions and scenarios on all user interfaces in the application.", "label": "Req"},
            {"text": "All error messages produced by the System must be meaningful, so that they can be appropriately acted upon by the users who are likely to see them.", "label": "Req"},
            {"text": "The user interfaces of the system should comply with Standard ISO 9241.", "label": "Req"},

            # NON-REQUIREMENTS - Show different categories of non-requirements
            {"text": "The Functional Requirements Specifications (FRS) report provides the detailed description of the functionalities required for the first version of the CCTNS.", "label": "Not_Req"},
            {"text": "The Registration module acts as an interface between the police and citizens and it eases the approach, interaction and information exchange between police and complainants.", "label": "Not_Req"},
            {"text": "The proposed functional architecture is modeled around centralized deployment to facilitate ease of maintenance and leverage advancement in open standards and web technologies.", "label": "Not_Req"},
            {"text": "Citizens can register their complaints with police and then based on the evidence, facts and following investigation, police shall take the complaint forward.", "label": "Not_Req"},
            {"text": "The Navigation module of the CCTNS provides role based landing pages which help in navigating through the CCTNS application.", "label": "Not_Req"}
        ]

    def classify_statement(self, statement: str, use_few_shot: bool = True, temperature: float = 0.0) -> str:
        """Classify a single statement."""
        if use_few_shot:
            examples = self._create_few_shot_examples()
            examples_text = "\n\nLEARN FROM THESE EXAMPLES:\n"
            for ex in examples:
                examples_text += f"\nStatement: \"{ex['text']}\"\nAnswer: {ex['label']}\n"

            user_message = f"{examples_text}\n---\n\nNow classify this new statement:\n\nStatement: \"{statement}\"\n\nAnswer (only 'Req' or 'Not_Req'):"
        else:
            user_message = f"Classify this statement:\n\nStatement: \"{statement}\"\n\nAnswer (only 'Req' or 'Not_Req'):"

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": temperature,
                    "max_tokens": 10
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                text_lower = text.lower()

                # Parse response
                if "not_req" in text_lower or "not req" in text_lower:
                    return "Not_Req"
                elif "non-requirement" in text_lower or "non requirement" in text_lower:
                    return "Not_Req"
                elif "requirement" in text_lower and "not" not in text_lower and "non" not in text_lower:
                    return "Req"
                elif text in ["Requirement", "Req", "req", "REQUIREMENT", "REQ"]:
                    return "Req"
                elif text in ["Not_Req", "Not Req", "NOT_REQ", "not_req", "not req"]:
                    return "Not_Req"
                else:
                    # Fallback: modal verb heuristic
                    stmt_lower = statement.lower()
                    if any(verb in stmt_lower for verb in [" shall ", " must ", " should "]):
                        return "Req"
                    return "Not_Req"

            else:
                print(f"Error {response.status_code}: {response.text[:200]}")
                # Fallback heuristic
                stmt_lower = statement.lower()
                if any(verb in stmt_lower for verb in [" shall ", " must ", " should "]):
                    return "Req"
                return "Not_Req"

        except Exception as e:
            print(f"Error: {e}")
            # Fallback heuristic
            stmt_lower = statement.lower()
            if any(verb in stmt_lower for verb in [" shall ", " must ", " should "]):
                return "Req"
            return "Not_Req"

    def classify_batch(self, statements: List[str], use_few_shot: bool = True, temperature: float = 0.0, batch_size: int = 100) -> List[str]:
        """Classify a batch of statements."""
        predictions = []
        total = len(statements)

        for i, statement in enumerate(statements):
            pred = self.classify_statement(statement, use_few_shot, temperature)
            predictions.append(pred)

            if (i + 1) % batch_size == 0 or (i + 1) == total:
                print(f"Processed {i + 1}/{total} statements...")

            # Small delay to avoid overwhelming the API
            time.sleep(0.2)

        return predictions

    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """Evaluate predictions."""
        label_map = {"Requirement": 1, "Req": 1, "Not_Req": 0, "Not Req": 0}
        y_true_bin = [label_map.get(label, label_map.get(label.replace("_", " "), 0)) for label in y_true]
        y_pred_bin = [label_map.get(label, label_map.get(label.replace("_", " "), 0)) for label in y_pred]

        return {
            "accuracy": accuracy_score(y_true_bin, y_pred_bin),
            "precision": precision_score(y_true_bin, y_pred_bin, average='binary', zero_division=0),
            "recall": recall_score(y_true_bin, y_pred_bin, average='binary', zero_division=0),
            "f1_score": f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
        }


def main():
    """Main execution."""
    # API key embedded directly in code
    api_key = 

    print("=" * 80)
    print("REQUIREMENTS DEMARCATOR - GPT-4o-mini")
    print("Via OpenRouter API")
    print("=" * 80)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(script_dir, 'a4data', 'a4data', 'PURE_test.csv')
    df_test = pd.read_csv(test_data_path)
    print(f"\nLoaded {len(df_test)} test samples")

    # Initialize demarcator
    print("\nInitializing GPT-4o-mini...")
    demarcator = RequirementsDemarcatorGPT(
        api_key=api_key,
        model="openai/gpt-4o-mini"
    )

    statements = df_test['Requirement'].tolist()
    true_labels = df_test['Req/Not Req'].tolist()

    # Run on full dataset directly
    print("\n" + "=" * 80)
    print("CLASSIFYING FULL DATASET (1,534 samples)...")
    print("=" * 80)

    start_time = time.time()
    predictions = demarcator.classify_batch(statements, use_few_shot=True)
    end_time = time.time()

    # Save results
    df_test['Predicted'] = predictions
    output_file = os.path.join(script_dir, 'predictions_gpt.csv')
    df_test.to_csv(output_file, index=False)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("FINAL RESULTS - GPT-4o-mini")
    print("=" * 80)

    metrics = demarcator.evaluate(true_labels, predictions)
    elapsed = end_time - start_time

    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Time:      {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # Detailed report
    label_map = {"Requirement": 1, "Req": 1, "Not_Req": 0, "Not Req": 0}
    y_true_bin = [label_map.get(label, 0) for label in true_labels]
    y_pred_bin = [label_map.get(label, 0) for label in predictions]

    print("\nDetailed Classification Report:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=['Not_Req', 'Req']))

    # Save metrics
    metrics_file = os.path.join(script_dir, 'metrics_gpt.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

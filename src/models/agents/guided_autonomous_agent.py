import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
import joblib
import os
from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import (
    PreprocessorAdvanced,
)
from src.util.data_util import (
    get_feature_name,
    get_feature_reference_range,
    get_feature_uom,
    get_all_feature_groups,
    get_feature_group_keys,
    get_feature_group_title,
    validate_feature_exists,
    get_common_feature_aliases,
    get_clinical_group_aliases,
)

logger = logging.getLogger("PULSE_logger")


class ComputationalBudget:
    """Manages computational resources for autonomous reasoning."""

    def __init__(self, max_tokens: int, max_rounds: int, max_time: float):
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self.max_time = max_time
        self.used_tokens = 0
        self.used_rounds = 0
        self.start_time = time.time()

    def has_budget(self) -> bool:
        return (
            self.used_tokens < self.max_tokens
            and self.used_rounds < self.max_rounds
            and time.time() - self.start_time < self.max_time
        )

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.used_rounds

    def consume_tokens(self, tokens: int):
        self.used_tokens += tokens

    def consume_round(self):
        self.used_rounds += 1

    def update_from_result(self, result: Dict[str, Any]):
        """Update budget from step result."""
        self.consume_tokens(
            result.get("result", {}).get("num_input_tokens", 0)
            + result.get("result", {}).get("num_output_tokens", 0)
        )


class AutonomousToolkit:
    """Lightweight toolkit for autonomous clinical reasoning."""

    def __init__(self, pretrained_models: Optional[Dict] = None):
        self.xgboost_models = pretrained_models or {}
        self.preprocessor_advanced = PreprocessorAdvanced()

    def get_feature_guidance(
        self,
        patient_data: pd.Series,
        current_assessment: Dict,
        available_features: Set[str],
    ) -> Dict[str, float]:
        """Get XGBoost-guided feature importance scores."""
        try:
            # For now, return a simple heuristic-based importance
            # This will be replaced with actual XGBoost model when path is provided
            if not self.xgboost_models:
                return self._get_heuristic_importance(
                    available_features, current_assessment
                )

            # TODO: When XGBoost model path is provided, implement:
            # model = self.xgboost_models.get("feature_importance")
            # if model and hasattr(model, 'feature_importances_'):
            #     return self._get_xgboost_importance(model, patient_data, available_features)

            return self._get_heuristic_importance(
                available_features, current_assessment
            )

        except Exception as e:
            logger.warning("Error in feature guidance: %s", e)
            return {}

    def _get_heuristic_importance(
        self, available_features: Set[str], current_assessment: Dict
    ) -> Dict[str, float]:
        """Heuristic-based feature importance until XGBoost model is available."""
        importance_scores = {}

        # Define importance based on clinical relevance
        high_importance = ["crea", "bun", "urine", "lact", "ph", "bicar", "map", "hr"]
        medium_importance = ["wbc", "plt", "hgb", "na", "k", "glu", "temp", "resp"]

        for feature in available_features:
            if feature in high_importance:
                importance_scores[feature] = 0.8 + np.random.random() * 0.2
            elif feature in medium_importance:
                importance_scores[feature] = 0.5 + np.random.random() * 0.3
            else:
                importance_scores[feature] = 0.1 + np.random.random() * 0.4

        return importance_scores

    def calculate_complexity_score(self, patient_data: pd.Series) -> float:
        """Calculate case complexity based on number of abnormal parameters."""
        try:
            # Count abnormal values using preprocessor
            patient_df = pd.DataFrame([patient_data])
            aggregated_df = self.preprocessor_advanced.aggregate_feature_windows(
                patient_df
            )

            abnormal_count = 0
            total_count = 0

            for col in aggregated_df.columns:
                if col.endswith("_mean"):
                    feature_key = col.replace("_mean", "")
                    if validate_feature_exists(feature_key):
                        value = aggregated_df[col].iloc[0]
                        if not pd.isna(value):
                            normal_range = get_feature_reference_range(feature_key)
                            if normal_range != (0, 0):
                                total_count += 1
                                if value < normal_range[0] or value > normal_range[1]:
                                    abnormal_count += 1

            complexity = abnormal_count / max(total_count, 1)
            return min(complexity, 1.0)

        except Exception as e:
            logger.warning("Error calculating complexity: %s", e)
            return 0.5


class GuidedAutonomousAgent(PulseAgent):
    """
    Agent with guided autonomy - can plan its own reasoning but within clinical constraints.

    Key autonomy features:
    1. Self-generates reasoning strategies based on case complexity
    2. Dynamically adjusts confidence thresholds and iteration limits
    3. Self-evaluates reasoning quality and adapts approach
    4. Uses lightweight "mental models" for feature selection
    """

    def __init__(
        self,
        model: Any,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
        pretrained_models: Optional[Dict[str, Any]] = None,
        max_budget_tokens: int = 2000,
        max_budget_rounds: int = 4,
        max_budget_time: float = 30.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            metrics_tracker=metrics_tracker,
            **kwargs,
        )

        # Autonomous reasoning components
        self.reasoning_strategies = [
            "pattern_matching",
            "differential_diagnosis",
            "temporal_analysis",
            "hypothesis_testing",
        ]
        self.pretrained_models = pretrained_models or {}
        self.toolkit = AutonomousToolkit(pretrained_models)

        # Budget parameters
        self.max_budget_tokens = max_budget_tokens
        self.max_budget_rounds = max_budget_rounds
        self.max_budget_time = max_budget_time

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Use data_util feature groups
        self.vital_signs = set(get_feature_group_keys("vitals"))
        self.lab_groups = get_all_feature_groups()

        self.task_content = self._get_task_specific_content()

        self._define_autonomous_steps()

    def _define_autonomous_steps(self) -> None:
        """Define autonomous reasoning steps."""

        # Step 1: Case complexity assessment and strategy selection
        self.add_step(
            name="strategy_planning",
            system_message="You are a meta-cognitive clinical reasoner who plans the optimal reasoning strategy for each case.",
            prompt_template=self._create_strategy_planning_template(),
            input_formatter=self._format_initial_data,
            parse_json=True,
        )

        # Step 2: Initial assessment with selected strategy
        self.add_step(
            name="strategic_assessment",
            system_message="You are applying your chosen reasoning strategy to make an initial clinical assessment.",
            prompt_template=self._create_strategic_assessment_template(),
            input_formatter=self._format_strategic_data,
            parse_json=True,
        )

        # Step 3: Self-evaluation and adaptation
        self.add_step(
            name="self_evaluation",
            system_message="You are critically evaluating your own reasoning quality and deciding if adjustments are needed.",
            prompt_template=self._create_self_evaluation_template(),
            input_formatter=self._format_evaluation_data,
            parse_json=True,
        )

        # Step 4: Adaptive information gathering (if needed)
        self.add_step(
            name="adaptive_investigation",
            system_message="You are strategically gathering additional information based on your self-evaluation.",
            prompt_template=self._create_adaptive_investigation_template(),
            input_formatter=self._format_investigation_data,
            parse_json=True,
        )

        # Step 5: Final synthesis with confidence calibration
        self.add_step(
            name="confident_synthesis",
            system_message=None,  # Use default
            prompt_template=self._create_confident_synthesis_template(),
            input_formatter=self._format_synthesis_data,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient through guided autonomous workflow."""

        # Reset memory
        self.memory.reset()

        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        self.memory.set_current_sample(sample_id)

        # Filter out _na columns
        patient_data = self._filter_na_columns(patient_data)

        # Initialize computational budget
        budget = ComputationalBudget(
            max_tokens=self.max_budget_tokens,
            max_rounds=self.max_budget_rounds,
            max_time=self.max_budget_time,
        )

        # Initialize state
        state = {
            "patient_data": patient_data,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "available_features": set(patient_data.index),
            "used_features": set(),
            "assessment_history": [],
            "current_confidence": 0.0,
            "reasoning_round": 0,
            "selected_strategy": None,
            "confidence_threshold": 0.85,  # Will be set autonomously
            "max_rounds": 3,  # Will be set autonomously
            "evidence_strength": 0.0,
            "case_complexity": 0.0,
        }

        try:
            # Step 1: Strategy planning (always runs)
            strategy_result = self.run_step("strategy_planning", patient_data, state)
            budget.consume_tokens(
                strategy_result["result"]["num_input_tokens"]
                + strategy_result["result"]["num_output_tokens"]
            )

            # Extract autonomous parameters
            strategy_output = strategy_result["output"]
            state["selected_strategy"] = strategy_output.get(
                "selected_strategy", "pattern_matching"
            )
            state["case_complexity"] = float(
                strategy_output.get("complexity_assessment", "0.5").split("/")[0]
            )
            state["confidence_threshold"] = strategy_output.get(
                "confidence_threshold", 0.85
            )
            state["max_rounds"] = min(
                strategy_output.get("max_rounds", 3), budget.remaining_rounds
            )

            # Log autonomous decision
            self.memory.log_decision_point(
                step="strategy_planning",
                options=self.reasoning_strategies,
                chosen=state["selected_strategy"],
                reasoning=strategy_output.get("reasoning", ""),
            )

            logger.info(
                "Strategy selected: %s, Complexity: %.2f, Threshold: %.2f, Max rounds: %d",
                state["selected_strategy"],
                state["case_complexity"],
                state["confidence_threshold"],
                state["max_rounds"],
            )

            # Step 2: Strategic assessment
            assessment_result = self.run_step(
                "strategic_assessment", patient_data, state
            )
            budget.update_from_result(assessment_result)

            assessment_output = assessment_result["output"]
            confidence_normalized = self._extract_confidence(assessment_output)
            state["current_confidence"] = confidence_normalized

            state["assessment_history"].append(
                {
                    "step": "strategic_assessment",
                    "strategy": state["selected_strategy"],
                    "reasoning": assessment_output.get(
                        "explanation", assessment_output.get("reasoning", "")
                    ),
                    "probability": assessment_output.get("probability", 50),
                    "confidence": assessment_output.get("confidence", 50),
                }
            )

            # Update used features (vitals for initial assessment)
            state["used_features"].update(
                self._get_available_vitals(state["available_features"])
            )

            # Adaptive reasoning loop with budget awareness
            while (
                budget.has_budget()
                and state["reasoning_round"] < state["max_rounds"]
                and state["current_confidence"] < state["confidence_threshold"]
            ):

                state["reasoning_round"] += 1
                budget.consume_round()

                logger.info(
                    "Starting reasoning round %d, current confidence: %.3f",
                    state["reasoning_round"],
                    state["current_confidence"],
                )

                # Step 3: Self-evaluation
                eval_result = self.run_step("self_evaluation", assessment_output, state)
                budget.update_from_result(eval_result)

                eval_output = eval_result["output"]
                needs_investigation = eval_output.get("needs_more_investigation", False)

                # Log self-evaluation decision
                self.memory.log_decision_point(
                    step="self_evaluation",
                    options=["continue_investigation", "stop_investigation"],
                    chosen=(
                        "continue_investigation"
                        if needs_investigation
                        else "stop_investigation"
                    ),
                    reasoning=eval_output.get("reasoning", ""),
                )

                if not needs_investigation:
                    logger.info(
                        "Self-evaluation indicates sufficient information, stopping"
                    )
                    break

                # Step 4: Adaptive investigation
                investigation_result = self.run_step(
                    "adaptive_investigation", eval_output, state
                )
                budget.update_from_result(investigation_result)

                investigation_output = investigation_result["output"]
                selected_features = investigation_output.get("selected_features", [])

                if not selected_features:
                    logger.info("No additional features selected, stopping iteration")
                    break

                # Validate and get available features
                valid_features = self._validate_features(selected_features)
                available_labs = self._get_available_labs(
                    valid_features, state["available_features"]
                )

                # Filter out already used features
                new_features = available_labs - state["used_features"]
                if not new_features:
                    logger.info("No new features available, stopping iteration")
                    break

                logger.info("Investigating features: %s", list(new_features))
                state["used_features"].update(new_features)

                # Log investigation decision
                self.memory.log_decision_point(
                    step="adaptive_investigation",
                    options=list(available_labs),
                    chosen=str(list(new_features)),
                    reasoning=investigation_output.get("reasoning", ""),
                )

                # Analyze new features and update assessment
                new_feature_data = self._format_clinical_data(
                    patient_data=patient_data,
                    feature_keys=new_features,
                    include_demographics=False,
                    include_temporal_patterns=True,
                )

                # Create updated assessment prompt
                updated_assessment = self._create_updated_assessment_from_investigation(
                    new_feature_data, state, investigation_output
                )

                # Generate updated assessment
                updated_result = self.model._generate_standard(
                    input_text=updated_assessment,
                    custom_system_message="You are updating your clinical assessment with new information.",
                    parse_json=True,
                )

                budget.consume_tokens(
                    updated_result.get("num_input_tokens", 0)
                    + updated_result.get("num_output_tokens", 0)
                )

                # Add to memory
                self.memory.add_step(
                    step_name=f"updated_assessment_round_{state['reasoning_round']}",
                    input_data=updated_assessment,
                    output_data=updated_result["generated_text"],
                    system_message="You are updating your clinical assessment with new information.",
                    num_input_tokens=updated_result.get("num_input_tokens", 0),
                    num_output_tokens=updated_result.get("num_output_tokens", 0),
                    token_time=updated_result.get("token_time", 0),
                    infer_time=0,
                )

                # Update state
                assessment_output = updated_result["generated_text"]
                confidence_normalized = self._extract_confidence(assessment_output)
                state["current_confidence"] = confidence_normalized

                state["assessment_history"].append(
                    {
                        "step": f"investigation_round_{state['reasoning_round']}",
                        "new_features": list(new_features),
                        "reasoning": assessment_output.get(
                            "explanation", assessment_output.get("reasoning", "")
                        ),
                        "probability": assessment_output.get("probability", 50),
                        "confidence": assessment_output.get("confidence", 50),
                    }
                )

                logger.info(
                    "Updated assessment complete. New confidence: %.3f",
                    confidence_normalized,
                )

            logger.info(
                "Autonomous reasoning complete after %d rounds. Final confidence: %.3f",
                state["reasoning_round"],
                state["current_confidence"],
            )

            # Step 5: Final synthesis
            final_result = self.run_step("confident_synthesis", None, state)
            final_output = final_result["output"]

            # Aggregate token metrics
            all_steps = self.memory.samples.get(str(sample_id), [])
            total_input_tokens = sum(step.num_input_tokens for step in all_steps)
            total_output_tokens = sum(step.num_output_tokens for step in all_steps)
            total_token_time = sum(step.token_time for step in all_steps)
            total_infer_time = sum(step.infer_time for step in all_steps)

            return {
                "generated_text": final_output,
                "token_time": total_token_time,
                "infer_time": total_infer_time,
                "num_input_tokens": total_input_tokens,
                "num_output_tokens": total_output_tokens,
                "autonomy_trace": self.memory.get_react_trace(),
            }

        except Exception as e:
            logger.error("Error in guided autonomous workflow: %s", e, exc_info=True)
            return {
                "generated_text": {"error": f"Autonomous workflow error: {str(e)}"},
                "token_time": 0,
                "infer_time": 0,
                "num_input_tokens": 0,
                "num_output_tokens": 0,
                "autonomy_trace": self.memory.get_react_trace(),
            }

    def _create_strategy_planning_template(self):
        """Template for autonomous strategy selection."""

        def format_prompt(formatted_data, state):
            vital_signs = formatted_data["vital_signs"]
            demographics = formatted_data["demographics"]
            complexity_score = self.toolkit.calculate_complexity_score(
                state["patient_data"]
            )

            # Format vital signs summary
            vitals_summary = []
            abnormal_count = 0
            for feature_key, data in vital_signs.items():
                value_range = (
                    f"{data['min']:.1f}-{data['max']:.1f}"
                    if abs(data["min"] - data["max"]) > 0.01
                    else f"{data['mean']:.1f}"
                )
                normal_range = data["normal_range"]
                if normal_range != (0, 0):
                    if data["mean"] < normal_range[0] or data["mean"] > normal_range[1]:
                        abnormal_count += 1
                        vitals_summary.append(
                            f"- {data['name']}: {value_range} {data['unit']} (abnormal)"
                        )
                    else:
                        vitals_summary.append(
                            f"- {data['name']}: {value_range} {data['unit']} (normal)"
                        )
                else:
                    vitals_summary.append(
                        f"- {data['name']}: {value_range} {data['unit']}"
                    )

            return f"""Analyze this case and select the optimal reasoning strategy.

Patient Demographics: {demographics}

Vital Signs Overview:
{chr(10).join(vitals_summary)}

Case Complexity Indicators:
- Abnormal vital signs: {abnormal_count}/{len(vital_signs)}
- Calculated complexity score: {complexity_score:.2f}

Available reasoning strategies:
1. PATTERN_MATCHING: For clear, typical presentations with obvious patterns
2. DIFFERENTIAL_DIAGNOSIS: For complex cases requiring systematic differential consideration
3. TEMPORAL_ANALYSIS: For cases where temporal trends are the key diagnostic feature
4. HYPOTHESIS_TESTING: For ambiguous cases requiring systematic investigation

Strategy Selection Criteria:
- Simple cases (complexity < 0.3): Pattern matching usually sufficient
- Moderate cases (0.3-0.7): Differential diagnosis or temporal analysis
- Complex cases (> 0.7): Hypothesis testing with systematic investigation

Also determine your adaptive parameters based on case complexity and computational efficiency:
- Confidence threshold: Higher complexity → Higher threshold (0.7-0.9)
- Maximum investigation rounds: Simple cases 1-2, complex cases 3-4
- Investigation strategy: Targeted vs broad approach

Clinical goal: Determine risk of {self.task_content['complication_name']}

Respond in JSON format:
{{
    "selected_strategy": "strategy_name",
    "complexity_assessment": "{complexity_score:.2f}/1.0 - simple/moderate/complex",
    "confidence_threshold": 0.X,
    "max_rounds": X,
    "investigation_strategy": "targeted/broad/minimal",
    "reasoning": "why this strategy is optimal for this case complexity and clinical picture"
}}"""

        return format_prompt

    def _create_strategic_assessment_template(self):
        """Template for strategy-specific initial assessment."""

        def format_prompt(formatted_data, state):
            vital_signs = formatted_data["vital_signs"]
            demographics = formatted_data["demographics"]
            strategy = state.get("selected_strategy", "pattern_matching")

            strategy_instructions = {
                "pattern_matching": "Focus on recognizing typical patterns and established clinical presentations. Look for classic combinations of findings.",
                "differential_diagnosis": "Systematically consider multiple possible diagnoses. Evaluate how findings support or refute different conditions.",
                "temporal_analysis": "Pay special attention to trends and temporal patterns. Analyze how values change over time.",
                "hypothesis_testing": "Form initial hypotheses and identify what additional evidence would support or refute them.",
            }

            instruction = strategy_instructions.get(
                strategy, strategy_instructions["pattern_matching"]
            )

            # Format demographics
            demo_text = []
            if "age" in demographics:
                demo_text.append(f"Age: {demographics['age']}")
            if "sex" in demographics:
                demo_text.append(f"Sex: {demographics['sex']}")
            if "weight" in demographics:
                demo_text.append(f"Weight: {demographics['weight']}")
            demographics_str = (
                ", ".join(demo_text) if demo_text else "Demographics: Not available"
            )

            # Format vital signs
            vitals_text = self._format_clinical_text(vital_signs)
            vitals_str = "\n".join(vitals_text)

            return f"""Apply the {strategy.upper()} strategy to assess this ICU patient for risk of {self.task_content['complication_name']}.

Strategy Instructions: {instruction}

Patient Demographics: {demographics_str}

Current vital signs (over monitoring period):
{vitals_str}

Clinical Context: {self.task_content['task_info']}

Apply your selected reasoning strategy systematically. Consider temporal patterns where available and the clinical significance of combined findings.

Respond in JSON format:
{{
    "diagnosis": "preliminary-{self.task_content['complication_name']}-risk",
    "probability": "XX (integer between 0 and 100)",
    "explanation": "Your {strategy} reasoning including specific strategy application and clinical assessment",
    "confidence": "XX (integer between 0 and 100)",
    "strategy_effectiveness": "how well the chosen strategy fits this case"
}}

Note: With only vital signs available, confidence should typically be 50-70 unless clinical picture is very clear."""

        return format_prompt

    def _create_self_evaluation_template(self):
        """Template for self-evaluation of reasoning quality."""

        def format_prompt(formatted_data, state):
            current_assessment = formatted_data
            strategy_used = state.get("selected_strategy", "unknown")
            round_num = state.get("reasoning_round", 0)

            return f"""Critically evaluate your reasoning quality and decide if more investigation is needed.

Current Assessment (Round {round_num}):
- Strategy used: {strategy_used}
- Probability: {current_assessment.get('probability', 'unknown')}%
- Confidence: {current_assessment.get('confidence', 'unknown')}%
- Reasoning: {current_assessment.get('explanation', '')}

Self-evaluation criteria:
1. Evidence strength: How well does available data support your conclusion?
2. Reasoning completeness: Are there important gaps in your analysis?
3. Uncertainty handling: Have you appropriately acknowledged limitations?
4. Clinical coherence: Does your reasoning align with clinical knowledge?
5. Strategy effectiveness: Did your chosen strategy work well for this case?

Decision factors for continued investigation:
- Low confidence (<70%) suggests need for more data
- High uncertainty in reasoning indicates knowledge gaps
- Conflicting evidence requires clarification
- Clinical instinct suggests missing pieces

Rate each criterion (1-5) and decide if you need additional investigation:

Respond in JSON format:
{{
    "evidence_strength": 1-5,
    "reasoning_completeness": 1-5, 
    "uncertainty_handling": 1-5,
    "clinical_coherence": 1-5,
    "strategy_effectiveness": 1-5,
    "overall_quality": 1-5,
    "needs_more_investigation": true/false,
    "investigation_focus": "what specific area needs more data (if needed)",
    "confidence_adjustment": "increase/decrease/maintain",
    "reasoning": "detailed self-critique and justification for decision"
}}"""

        return format_prompt

    def _create_adaptive_investigation_template(self):
        """Template for adaptive information gathering."""

        def format_prompt(formatted_data, state):
            evaluation = formatted_data
            available_features = state["available_features"]
            used_features = state["used_features"]

            # Get feature importance scores from toolkit
            feature_scores = self.toolkit.get_feature_guidance(
                state["patient_data"],
                state["assessment_history"][-1] if state["assessment_history"] else {},
                available_features - used_features,
            )

            # Get available tests by clinical group
            available_by_group = self._get_lab_groups_available(available_features)

            # Filter out already used features
            filtered_available = {}
            for group_name, features in available_by_group.items():
                unused_features = [f for f in features if f not in used_features]
                if unused_features:
                    filtered_available[group_name] = unused_features

            # Format test list
            test_list = []
            for group_name, features in filtered_available.items():
                group_title = get_feature_group_title(group_name)
                test_list.append(f"\n{group_title}:")
                for feature in features:
                    importance_score = feature_scores.get(feature, 0.0)
                    test_list.append(
                        f"  - {feature} (importance: {importance_score:.2f})"
                    )

            investigation_focus = evaluation.get(
                "investigation_focus", "general assessment"
            )

            return f"""Design strategic information gathering based on your self-evaluation.

Self-evaluation results:
- Overall quality: {evaluation.get('overall_quality', 'unknown')}/5
- Investigation focus: {investigation_focus}
- Reasoning: {evaluation.get('reasoning', '')}

Available features (with importance scores):
{''.join(test_list)}

Selection principles for {investigation_focus}:
- Target your identified knowledge gaps
- Prioritize high-importance features (score > 0.6)
- Consider cost-benefit of additional testing
- Select 2-4 most informative features
- Focus on features that can distinguish between competing hypotheses

Feature selection strategy:
- High importance + relevant to focus area = highest priority
- Features that can rule out competing diagnoses
- Tests that provide complementary information

Use EXACT abbreviations from the list above.

Respond in JSON format:
{{
    "investigation_strategy": "targeted/broad/minimal",
    "selected_features": ["feature1", "feature2", "feature3"],
    "selection_rationale": {{
        "feature1": "why this feature was chosen",
        "feature2": "why this feature was chosen"
    }},
    "expected_impact": "how this will improve your assessment",
    "stopping_criteria": "when you'll have enough information",
    "reasoning": "overall investigation strategy and feature selection logic"
}}"""

        return format_prompt

    def _create_confident_synthesis_template(self):
        """Template for final synthesis with confidence calibration."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            clinical_data = formatted_data["clinical_data"]
            assessment_history = formatted_data["assessment_history"]
            autonomy_info = formatted_data.get("autonomy_info", {})

            # Format demographics
            demo_text = []
            if "age" in demographics:
                demo_text.append(f"Age: {demographics['age']}")
            if "sex" in demographics:
                demo_text.append(f"Sex: {demographics['sex']}")
            if "weight" in demographics:
                demo_text.append(f"Weight: {demographics['weight']}")
            demographics_str = (
                ", ".join(demo_text) if demo_text else "Demographics: Not available"
            )

            # Format all clinical data
            clinical_text = self._format_clinical_text(clinical_data)
            clinical_str = "\n".join(clinical_text)

            # Format assessment progression
            assessment_summary = []
            for i, assessment in enumerate(assessment_history):
                step_name = assessment.get("step", f"step_{i+1}")
                prob = assessment.get("probability", "unknown")
                conf = assessment.get("confidence", "unknown")
                reasoning = (
                    assessment.get("reasoning", "")[:100] + "..."
                    if len(assessment.get("reasoning", "")) > 100
                    else assessment.get("reasoning", "")
                )
                assessment_summary.append(
                    f"{i+1}. {step_name}: probability {prob}%, confidence {conf}% - {reasoning}"
                )

            return f"""Synthesize your autonomous reasoning process into a final clinical decision.

Patient Demographics: {demographics_str}

Clinical Data Summary (over monitoring period):
{clinical_str}

Autonomous Reasoning Progression:
Strategy used: {autonomy_info.get('strategy', 'unknown')}
Reasoning rounds: {autonomy_info.get('rounds', 0)}
Final confidence reached: {autonomy_info.get('final_confidence', 'unknown')}

Assessment Evolution:
{chr(10).join(assessment_summary)}

Task: Determine if this ICU patient will develop {self.task_content['complication_name']}.

Clinical Context: {self.task_content['task_info']}

Provide your final clinical decision based on your autonomous reasoning process. Consider:
- How your reasoning strategy helped analyze this case
- The progression of your confidence as you gathered more information
- The quality of evidence you accumulated
- Any remaining uncertainties or limitations

Your response should reflect the culmination of your autonomous decision-making process."""

        return format_prompt

    # Helper methods (reuse existing methods from ClinicalWorkflowAgent with minor adaptations)

    def _extract_confidence(self, output: Dict[str, Any]) -> float:
        """Extract confidence value from LLM output with fallback logic."""
        if "confidence" in output:
            confidence = output.get("confidence", 50)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 50
            return confidence / 100.0
        else:
            probability = output.get("probability", 50)
            if isinstance(probability, str):
                try:
                    probability = float(probability)
                except ValueError:
                    probability = 50
            return probability / 100.0

    def _get_available_vitals(self, available_features: Set[str]) -> Set[str]:
        """Get available vital signs from the data using data_util groups."""
        available_vitals = set()
        vitals_keys = get_feature_group_keys("vitals")

        for vital in vitals_keys:
            if any(feat.startswith(vital) for feat in available_features):
                available_vitals.add(vital)
        return available_vitals

    def _get_available_labs(
        self, requested_labs: List[str], available_features: Set[str]
    ) -> Set[str]:
        """Get available requested labs from the data."""
        available_labs = set()
        for lab in requested_labs:
            if any(feat.startswith(lab) for feat in available_features):
                available_labs.add(lab)
        return available_labs

    def _get_lab_groups_available(
        self, available_features: Set[str]
    ) -> Dict[str, List[str]]:
        """Get available lab tests organized by clinical groups."""
        available_by_group = {}

        for group_name, group_dict in self.lab_groups.items():
            if group_name == "vitals":  # Skip vitals as they're handled separately
                continue

            available_in_group = []
            for feature_key in group_dict.keys():
                if any(feat.startswith(feature_key) for feat in available_features):
                    available_in_group.append(feature_key)

            if available_in_group:
                available_by_group[group_name] = available_in_group

        return available_by_group

    def _validate_features(self, feature_list: List[str]) -> List[str]:
        """Validate that requested features exist in the data_util feature dictionary."""
        valid_features = []

        # Get mappings from data_util
        group_mappings = {}

        # Add official group mappings from data_util
        for group_key in [
            "bga",
            "coag",
            "electrolytes_met",
            "liver_kidney",
            "hematology_immune",
            "cardiac",
        ]:
            group_features = get_feature_group_keys(group_key)
            group_title = get_feature_group_title(group_key).lower()

            group_mappings[group_key] = group_features
            group_mappings[group_title] = group_features

        # Add clinical aliases from data_util
        clinical_aliases = get_clinical_group_aliases()
        for feature_tuple, aliases in clinical_aliases.items():
            feature_list_from_tuple = list(feature_tuple)
            for alias in aliases:
                group_mappings[alias] = feature_list_from_tuple

        # Get individual feature name mappings from data_util
        name_mappings = get_common_feature_aliases()

        for feature in feature_list:
            feature_lower = feature.lower().strip()

            # First check exact match (highest priority)
            if validate_feature_exists(feature):
                valid_features.append(feature)
                continue

            # Check lowercase exact match
            if validate_feature_exists(feature_lower):
                valid_features.append(feature_lower)
                continue

            # Check if it's a group name that needs expansion
            if feature_lower in group_mappings:
                group_features = group_mappings[feature_lower]
                added_features = []
                for group_feature in group_features:
                    if validate_feature_exists(group_feature):
                        valid_features.append(group_feature)
                        added_features.append(group_feature)
                if added_features:
                    logger.info(
                        "Expanded group '%s' to features: %s", feature, added_features
                    )
                continue

            # Check if it needs individual mapping
            if feature_lower in name_mappings:
                mapped_feature = name_mappings[feature_lower]
                if validate_feature_exists(mapped_feature):
                    valid_features.append(mapped_feature)
                    logger.info("Mapped '%s' to '%s'", feature, mapped_feature)
                    continue
                else:
                    logger.warning(
                        "Mapped feature '%s' not found in data_util", mapped_feature
                    )
                    continue

            # Check for partial matches in available feature groups
            found_match = False
            for group_name, group_dict in self.lab_groups.items():
                for feature_key in group_dict.keys():
                    if (
                        feature_lower in feature_key.lower()
                        or feature_key.lower() in feature_lower
                    ):
                        if validate_feature_exists(feature_key):
                            valid_features.append(feature_key)
                            logger.info(
                                "Partial match: '%s' -> '%s'", feature, feature_key
                            )
                            found_match = True
                            break
                if found_match:
                    break

            if not found_match:
                logger.warning("Feature '%s' not found in data_util", feature)

        return valid_features

    def _format_clinical_data(
        self,
        patient_data: pd.Series,
        feature_keys: Set[str],
        include_demographics: bool = False,
        include_temporal_patterns: bool = False,
    ) -> Dict[str, Any]:
        """Format clinical data using aggregate_feature_windows."""
        # Convert patient data to DataFrame for preprocessing
        patient_df = pd.DataFrame([patient_data])

        # Use aggregate_feature_windows to get min, max, mean for each feature
        aggregated_df = self.preprocessor_advanced.aggregate_feature_windows(patient_df)
        aggregated_row = aggregated_df.iloc[0]

        # Prepare result dictionary
        result = {}

        # Extract patient demographics if requested
        if include_demographics:
            patient_demographics = {}
            if "age" in patient_data.index:
                patient_demographics["age"] = patient_data["age"]
            if "sex" in patient_data.index:
                patient_demographics["sex"] = patient_data["sex"]
            if "weight" in patient_data.index:
                patient_demographics["weight"] = patient_data["weight"]
            result["demographics"] = patient_demographics

        # Format clinical features with aggregated values using data_util
        clinical_data = {}

        for feature_key in feature_keys:
            # Check if this feature has aggregated columns in the data
            if any(col.startswith(f"{feature_key}_") for col in aggregated_row.index):
                min_val = aggregated_row.get(f"{feature_key}_min", None)
                max_val = aggregated_row.get(f"{feature_key}_max", None)
                mean_val = aggregated_row.get(f"{feature_key}_mean", None)

                # Only include features with non-NaN mean values
                if not pd.isna(mean_val):
                    feature_info = {
                        "name": get_feature_name(feature_key),
                        "min": (
                            float(min_val) if not pd.isna(min_val) else float(mean_val)
                        ),
                        "max": (
                            float(max_val) if not pd.isna(max_val) else float(mean_val)
                        ),
                        "mean": float(mean_val),
                        "unit": get_feature_uom(feature_key),
                        "normal_range": get_feature_reference_range(feature_key),
                    }

                    # Add temporal pattern information if requested
                    if include_temporal_patterns:
                        feature_info["temporal_pattern"] = (
                            self._analyze_temporal_pattern(
                                patient_data,
                                feature_key,
                                feature_info["min"],
                                feature_info["max"],
                                feature_info["mean"],
                            )
                        )

                    clinical_data[feature_key] = feature_info

        # Store clinical data under appropriate key
        if include_demographics:
            result["vital_signs"] = clinical_data
        else:
            result = clinical_data

        return result

    def _format_clinical_text(self, clinical_data: Dict[str, Dict]) -> List[str]:
        """Format clinical data dictionary into human-readable text lines."""
        formatted_lines = []

        for feature_key, data in clinical_data.items():
            name = data["name"]
            min_val = data["min"]
            max_val = data["max"]
            mean_val = data["mean"]
            unit = data["unit"]
            normal_range = data["normal_range"]

            # Format value range or single value
            if abs(min_val - max_val) < 0.01:  # Essentially the same value
                value_str = f"{mean_val:.2f}"
            else:
                value_str = f"{min_val:.2f}-{max_val:.2f} (mean: {mean_val:.2f})"

            # Add normal range if available
            if unit and normal_range != (0, 0):
                base_text = f"- {name}: {value_str} {unit} (normal: {normal_range[0]}-{normal_range[1]} {unit})"
            else:
                base_text = f"- {name}: {value_str} {unit if unit else ''}"

            # Add temporal pattern information if available
            if "temporal_pattern" in data:
                base_text += f", {data['temporal_pattern']}"

            formatted_lines.append(base_text)

        return formatted_lines

    def _format_initial_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format initial data for strategy planning."""
        patient_data = input_data
        vitals_keys = get_feature_group_keys("vitals")
        available_vitals = {
            vital
            for vital in vitals_keys
            if any(feat.startswith(vital) for feat in patient_data.index)
        }

        return self._format_clinical_data(
            patient_data=patient_data,
            feature_keys=available_vitals,
            include_demographics=True,
            include_temporal_patterns=True,
        )

    def _format_strategic_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for strategic assessment."""
        return self._format_initial_data(state, input_data)

    def _format_evaluation_data(self, state: Dict[str, Any], input_data: Any) -> Any:
        """Format data for self-evaluation."""
        return input_data  # Pass through the assessment output

    def _format_investigation_data(self, state: Dict[str, Any], input_data: Any) -> Any:
        """Format data for adaptive investigation."""
        return input_data  # Pass through the evaluation output

    def _format_synthesis_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format comprehensive data for final synthesis."""
        patient_data = state["patient_data"]

        # Get demographics
        demographics = {}
        if "age" in patient_data.index:
            demographics["age"] = patient_data["age"]
        if "sex" in patient_data.index:
            demographics["sex"] = patient_data["sex"]
        if "weight" in patient_data.index:
            demographics["weight"] = patient_data["weight"]

        # Get all used clinical data with temporal patterns
        all_clinical_data = self._format_clinical_data(
            patient_data=patient_data,
            feature_keys=state["used_features"],
            include_demographics=False,
            include_temporal_patterns=True,
        )

        return {
            "demographics": demographics,
            "clinical_data": all_clinical_data,
            "assessment_history": state["assessment_history"],
            "autonomy_info": {
                "strategy": state.get("selected_strategy", "unknown"),
                "rounds": state.get("reasoning_round", 0),
                "final_confidence": f"{state.get('current_confidence', 0.0)*100:.1f}%",
                "case_complexity": f"{state.get('case_complexity', 0.0):.2f}",
            },
        }

    def _create_updated_assessment_from_investigation(
        self, new_feature_data: Dict, state: Dict[str, Any], investigation_output: Dict
    ) -> str:
        """Create prompt for updated assessment after investigation."""
        # Format new lab results
        formatted_labs = self._format_clinical_text(new_feature_data)
        labs_str = "\n".join(formatted_labs)

        previous_assessment = state["assessment_history"][-1]

        return f"""Update your clinical assessment with newly investigated information.

Previous Assessment:
- Strategy: {state.get('selected_strategy', 'unknown')}
- Reasoning: {previous_assessment.get('reasoning', '')}
- Previous probability: {previous_assessment.get('probability', 'unknown')}%
- Previous confidence: {previous_assessment.get('confidence', 'unknown')}%

Investigation Strategy: {investigation_output.get('investigation_strategy', 'targeted')}
Expected Impact: {investigation_output.get('expected_impact', '')}

New Laboratory Results (over monitoring period):
{labs_str}

Integration Instructions:
- How do these new findings change your assessment?
- Do they support or refute your previous reasoning?
- What is your updated probability and confidence?
- Are there any surprising findings that require strategy adjustment?

Respond in JSON format:
{{
    "diagnosis": "updated-{self.task_content['complication_name']}-assessment",
    "probability": "XX (integer between 0 and 100)",
    "explanation": "How the new information changes your assessment and interpretation",
    "confidence": "XX (integer between 0 and 100)",
    "strategy_effectiveness": "how well your investigation strategy worked"
}}"""

    def _get_task_specific_content(self) -> Dict[str, str]:
        """Get task-specific content for prompts."""
        task = self.task_name
        if task == "mortality":
            return {
                "complication_name": "mortality",
                "task_info": "ICU mortality refers to death occurring during the ICU stay. Key risk factors include hemodynamic instability, respiratory failure, multi-organ dysfunction, and severe metabolic derangements.",
            }
        elif task == "aki":
            return {
                "complication_name": "aki",
                "task_info": "Acute kidney injury (AKI) is defined by rapid decline in kidney function with increased creatinine (≥1.5x baseline or ≥0.3 mg/dL increase in 48h) or decreased urine output (<0.5 mL/kg/h for 6-12h). Common causes include sepsis, hypotension, and nephrotoxins.",
            }
        elif task == "sepsis":
            return {
                "complication_name": "sepsis",
                "task_info": "Sepsis is life-threatening organ dysfunction caused by dysregulated host response to infection. Diagnosed by SOFA score increase ≥2 points with suspected infection. Key indicators include fever, tachycardia, tachypnea, altered mental status, and laboratory abnormalities.",
            }
        return {
            "complication_name": "complications",
            "task_info": "General ICU complications assessment.",
        }

    def _filter_na_columns(self, patient_data: pd.Series) -> pd.Series:
        """Filter out columns with '_na' suffixes like Sarvari preprocessor does."""
        # Convert to DataFrame for regex filtering, then back to Series
        temp_df = pd.DataFrame([patient_data])
        filtered_df = temp_df.filter(regex=r"^(?!.*_na(_\d+)?$)")
        return filtered_df.iloc[0]

    def _analyze_temporal_pattern(
        self,
        patient_data: pd.Series,
        feature_key: str,
        min_val: float,
        max_val: float,
        mean_val: float,
    ) -> str:
        """Simple temporal pattern analysis returning brief text assessment."""
        try:
            # Extract time-windowed values for this feature
            time_series_values = []
            for col in patient_data.index:
                if col.startswith(f"{feature_key}_") and col != f"{feature_key}_na":
                    try:
                        val = patient_data[col]
                        if not pd.isna(val):
                            time_series_values.append(float(val))
                    except (ValueError, TypeError):
                        continue

            if len(time_series_values) < 3:
                return "insufficient data for trend analysis"

            # Simple trend analysis
            x = np.arange(len(time_series_values))
            y = np.array(time_series_values)

            # Calculate linear regression slope
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator == 0:
                return "stable trend"
            else:
                slope = numerator / denominator

            # Normalize slope by mean value to get relative change rate
            relative_slope = (slope / y_mean * 100) if y_mean != 0 else 0

            # Determine trend direction and strength
            if abs(relative_slope) < 2:
                trend = "stable trend"
            elif relative_slope >= 8:
                trend = "rapidly increasing"
            elif relative_slope >= 4:
                trend = "moderately increasing"
            elif relative_slope > 0:
                trend = "slowly increasing"
            elif relative_slope <= -8:
                trend = "rapidly decreasing"
            elif relative_slope <= -4:
                trend = "moderately decreasing"
            else:
                trend = "slowly decreasing"

            # Use categorize_features for abnormality assessment
            patient_df = pd.DataFrame([patient_data])
            categorized_df = self.preprocessor_advanced.categorize_features(
                df=patient_df,
                base_features={feature_key},
                X_cols=patient_data.index,
                num_categories=5,
                for_llm=True,
            )

            status = (
                categorized_df[feature_key].iloc[0]
                if feature_key in categorized_df.columns
                else "unknown range"
            )

            return f"{trend}, {status}"

        except Exception as e:
            logger.warning(
                "Error analyzing temporal pattern for %s: %s", feature_key, e
            )
            return "stable trend"

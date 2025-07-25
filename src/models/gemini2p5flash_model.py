# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_2_5_flash.ipynb
import logging
import os
import random
import time
import warnings
from typing import Any, Dict

# import vertexai
from google import genai
from google.genai.types import (CreateBatchJobConfig, GenerateContentConfig,
                                ThinkingConfig)

from src.models.pulse_model import PulseLLMModel
from src.util.config_util import set_seeds
from src.util.model_util import (parse_llm_output, prompt_template_hf)

# from vertexai.generative_models import (GenerationConfig, GenerativeModel, ThinkingConfig)


warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class Gemini2p5Model(PulseLLMModel):
    """Gemini2p5 model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Gemini2p5Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "Gemini2p5Model")
        super().__init__(model_name, params, **kwargs)

        required_params = [
            "max_new_tokens",
            "model_id",
            "thinking_budget",
            "temperature",
        ]
        self.check_required_params(params, required_params)

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project_id or not location:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in your .env file."
            )
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS must be set in your .env file to the path of your service account key."
            )

        logger.info(
            f"Initializing Vertex AI for project: {project_id}, location: {location}"
        )
        # Initialize Vertex AI. The SDK will automatically use the credentials from GOOGLE_APPLICATION_CREDENTIALS.
        # vertexai.init(project=project_id, location=location)

        self.client = genai.Client(
            vertexai=True, project=params.get("project_id", None), location="global"
        )
        self.model_id = params["model_id"]
        self.prompting_id = params.get("prompting_id", None)
        self.max_new_tokens = params["max_new_tokens"]
        self.thinking_budget = params["thinking_budget"]

        # self.model = GenerativeModel(self.model_id)
        self.is_agent = False
        self.agent_instance = None
        self.is_loaded = True  # Gemini models are loaded by default

    def _generate_standard(
        self,
        input_text: str,
        custom_system_message: str = None,
        parse_json: bool = True,
        generate_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """Standard generation logic for non-agent models."""
        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(
            input_text, custom_system_message, self.model_name, task=self.task_name
        )

        max_output_tokens = (
            self.max_new_tokens + self.thinking_budget
            if self.thinking_budget != -1
            else 10000
        )

        incl_thought = (
            True if self.thinking_budget > 0 or self.thinking_budget == -1 else False
        )

        infer_start = time.perf_counter()
        # Retry logic for rate limiting
        max_retries = 3
        base_delay = 30

        for attempt in range(max_retries + 1):
            try:
                infer_start = time.perf_counter()
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=input_text,
                    config=GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=self.params.get("temperature", 0.4),
                        top_p=1.0,
                        top_k=32,
                        thinking_config=ThinkingConfig(
                            thinking_budget=self.thinking_budget,
                            include_thoughts=incl_thought,
                        ),
                    ),
                )
                infer_time = time.perf_counter() - infer_start
                break  # Success, exit retry loop

            except Exception as e:
                error_message = str(e)

                if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt) + random.uniform(0, 5)
                        logger.warning(
                            f"Rate limit hit (429). Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for rate limiting"
                        )

                # Re-raise the exception if it's not a rate limit error or max retries exceeded
                raise e
        infer_time = time.perf_counter() - infer_start

        usage_metadata = response.usage_metadata
        num_input_tokens = usage_metadata.prompt_token_count
        num_output_tokens = usage_metadata.candidates_token_count

        if incl_thought:
            num_thinking_tokens = usage_metadata.thoughts_token_count
        else:
            num_thinking_tokens = 0

        thinking_output = ""
        answer_output = ""

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_output = part.text
            else:
                answer_output = part.text
                logger.debug("Decoded output:\n %s", answer_output)

        # Parse the output if parse_json is True
        if parse_json:
            generated_text = parse_llm_output(answer_output)
        else:
            generated_text = response

        logger.info(
            "Inference time: %.4fs | Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
            infer_time,
            num_input_tokens,
            num_output_tokens,
            num_thinking_tokens,
        )

        # Return consistent result structure
        return {
            "generated_text": generated_text,
            "thinking_output": thinking_output,
            "token_time": 0.0,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": num_thinking_tokens,
        }

    def evaluate_batched(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set using batch processing when available.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        # Creating an array of json tasks
        tasks = []
        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            idx = X[0]
            if self.is_agent:
                X_input = X[1]  # Full pandas Series with all patient features
            else:
                X_input = X[1].iloc[0]  # Single text prompt for standard models
            y[1].iloc[0]

            # system_message
            input_text = prompt_template_hf(
                X_input, None, self.model_name, task=self.task_name
            )

            tasks.append(
                {"contents": input_text, "generation_config": {"temperature": 0.4}}
            )

        job_name = f"batch_{self.task_name}_{self.dataset_name}"
        batch = self.client.batches.create(
            model=self.model_id,
            src=tasks,
            config=CreateBatchJobConfig(display_name=f"task-{idx}"),
        )

        # Wait for the batch job to complete
        batch_job = self.client.batches.get(name=job_name)

        completed_states = set(
            [
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ]
        )

        batch_job = self.client.batches.get(name=job_name)  # Initial get
        while batch_job.state.name not in completed_states:
            logger.debug(f"Current state: {batch_job.state.name}")
            time.sleep(60)
            batch_job = self.client.batches.get(name=job_name)

        logger.info(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == "JOB_STATE_FAILED":
            logger.error(f"Error: {batch_job.error}")

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            pass

            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                print(f"Response {i+1}:")
                if inline_response.response:
                    # Accessing response, structure may vary.
                    try:
                        print(inline_response.response.text)
                    except AttributeError:
                        print(inline_response.response)  # Fallback
                elif inline_response.error:
                    print(f"Error: {inline_response.error}")

            # # Sort by custom_id to maintain order
            # results.sort(key=lambda x: x.custom_id.split("-")[1])

            # metrics_tracker = MetricsTracker(
            #     self.model_name,
            #     self.task_name,
            #     self.dataset_name,
            #     self.save_dir,
            # )
            # for r in results:
            #     y_true = test_loader[1].loc[int(r.custom_id.split("-")[1])].iloc[0]
            #     X_input = test_loader[0].loc[int(r.custom_id.split("-")[1])].iloc[0]

            #     model_output = r.result
            #     if model_output.type == "errored":
            #         logger.warning(
            #             "Error in model output for task %s: %s",
            #             r.custom_id,
            #             model_output.error,
            #         )
            #         metrics_tracker.add_results(np.nan, y_true)
            #         metrics_tracker.add_metadata_item(
            #             {
            #                 "Input Prompt": X_input,
            #                 "Target Label": y_true,
            #                 "Predicted Probability": np.nan,
            #                 "Predicted Diagnosis": "error",
            #                 "Predicted Explanation": "error",
            #                 "Tokenization Time": 0,
            #                 "Inference Time": 0,
            #                 "Input Tokens": np.nan,
            #                 "Output Tokens": np.nan,
            #                 "Thinking Tokens": np.nan,
            #                 "Thinking Output": "error",
            #             }
            #         )
            #         continue

            #     num_input_tokens = r.result.message.usage.input_tokens
            #     num_output_tokens = r.result.message.usage.output_tokens
            #     num_thinking_tokens = (
            #         0  # Model provides a summary and not the actual reasoning tokens
            #     )

            #     thinking_output = ""
            #     answer_output = ""
            #     for block in r.result.message.content:
            #         if block.type == "thinking":
            #             thinking_output = block.thinking.encode(
            #                 "ascii", errors="replace"
            #             ).decode("ascii")

            #         elif block.type == "text":
            #             answer_output = block.text

            #     answer_output = answer_output.encode("ascii", errors="replace").decode(
            #         "ascii"
            #     )
            #     logger.debug("Decoded output:\n %s", answer_output)
            #     result_dict = parse_llm_output(answer_output)

            #     logger.info(
            #         "Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
            #         num_input_tokens,
            #         num_output_tokens,
            #         num_thinking_tokens,
            #     )

            #     # Handle case where generated_text is a string instead of dict (when parsing fails)
            #     if isinstance(result_dict, dict):
            #         predicted_probability = float(result_dict.get("probability", np.nan))
            #         predicted_diagnosis = result_dict.get("diagnosis", "error")
            #         generated_explanation = result_dict.get("explanation", "error")
            #     else:
            #         predicted_probability = np.nan
            #         predicted_diagnosis = "error"
            #         generated_explanation = "error"

            #     logger.info(
            #         "Predicted probability: %s | True label: %s",
            #         predicted_probability,
            #         y_true,
            #     )

            #     metrics_tracker.add_results(predicted_probability, y_true)
            #     metrics_tracker.add_metadata_item(
            #         {
            #             "Input Prompt": X_input,
            #             "Target Label": y_true,
            #             "Predicted Probability": predicted_probability,
            #             "Predicted Diagnosis": predicted_diagnosis,
            #             "Predicted Explanation": generated_explanation,
            #             "Tokenization Time": 0,
            #             "Inference Time": 0,
            #             "Input Tokens": num_input_tokens,
            #             "Output Tokens": num_output_tokens,
            #             "Thinking Tokens": num_thinking_tokens,
            #             "Thinking Output": thinking_output,
            #         }
            #     )
            #     if len(metrics_tracker.items) > 100:
            #         # Log metadata periodically to avoid memory issues
            #         metrics_tracker.log_metadata()

            # metrics_tracker.log_metadata(save_to_file=self.save_metadata)
            # metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
            # if save_report:
            #     metrics_tracker.save_report(prompting_id=self.prompting_id)

            # logger.info("Test evaluation completed for %s", self.model_name)
            # logger.info("Test metrics: %s", metrics_tracker.summary)

from typing import Dict, Any, Optional
import logging
import os
from pathlib import Path
from urllib.parse import urljoin
from getpass import getpass
import openai
import phoenix as px
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import SpanAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from litellm.integrations.custom_logger import CustomLogger
import litellm

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)


def get_working_dir() -> Path:
    working_dir_str = os.getenv("PHOENIX_WORKING_DIR", str(Path.home().resolve() / ".phoenix"))
    if working_dir_str is not None:
        return Path(working_dir_str)
    return Path.home().resolve() / ".phoenix"


def get_env_port() -> int:
    port = os.getenv("PHOENIX_PORT", "6006")
    if port.isnumeric():
        return int(port)
    raise ValueError(f"Invalid value for environment variable PHOENIX_PORT: {port}. Value must be an integer.")


def get_env_host() -> str:
    return os.getenv("PHOENIX_HOST", "0.0.0.0")


def get_env_collector_endpoint() -> Optional[str]:
    return os.getenv("PHOENIX_COLLECTOR_ENDPOINT")


def get_env_project_name() -> str:
    return os.getenv("PHOENIX_PROJECT_NAME", "default")


class _OpenInferenceExporter(OTLPSpanExporter):
    def __init__(self) -> None:
        host = get_env_host()
        if host == "0.0.0.0":
            host = "127.0.0.1"
        endpoint = urljoin(f"http://{host}:{get_env_port()}", "/v1/traces")
        logger.debug(f"Exporter endpoint set to: {endpoint}")
        super().__init__(endpoint)

    def export(self, spans):
        logger.debug(f"Exporting {len(spans)} span(s)")
        return super().export(spans)


class LitellmInstrumentor:
    _tracer_provider = None

    def __init__(self) -> None:
        if LitellmInstrumentor._tracer_provider is None:
            self.resource = Resource.create({ResourceAttributes.PROJECT_NAME: get_env_project_name()})
            logger.debug(f"Resource created with attributes: {self.resource.attributes}")
            LitellmInstrumentor._tracer_provider = trace_sdk.TracerProvider(resource=self.resource)
            logger.debug("TracerProvider initialized")
            LitellmInstrumentor._tracer_provider.add_span_processor(BatchSpanProcessor(_OpenInferenceExporter()))
            logger.debug("BatchSpanProcessor added to TracerProvider")

        self.tracer_provider = LitellmInstrumentor._tracer_provider
        self.tracer = self.tracer_provider.get_tracer(__name__)
        logger.debug("Tracer initialized")

    def log_interaction(self, input_args: Dict[str, Any], response: Any, session_name: str = "default",
                        parent_span: Any = None) -> None:
        logger.debug("Starting log_interaction")
        context = trace_api.set_span_in_context(parent_span) if parent_span else None
        with self.tracer.start_as_current_span(session_name, context=context) as span:
            logger.debug("Span started")

            # Set default values and check for key existence
            response_object = getattr(response, 'object', 'default_object')
            messages_content = input_args.get('messages', [{}])[0].get('content', 'noMessagesRetrieved')
            model_name = input_args.get('model', 'noModelDetected')
            output_value = response.choices[0].get('message', {}).get('content', 'noOutputRetrieved')
            usage = response.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            span.update_name(response_object)
            span.set_attribute(SpanAttributes.INPUT_VALUE, messages_content)
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens)
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, 'LLM')
            span.set_status(trace_api.StatusCode.OK)
            span.add_event("interaction_details_registered!")
            logger.debug(f"Span {span.get_span_context().span_id} created and populated with attributes")

class PhoenixLogger(CustomLogger):
    def __init__(self):
        self.instrumentor = LitellmInstrumentor()

    def log_pre_api_call(self, model, messages, kwargs):
        logger.debug(f"Pre-API Call with kwargs: {kwargs}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        logger.debug(f"Post-API Call with kwargs: {kwargs}, response: {response_obj}")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        logger.debug(f"On Stream with kwargs: {kwargs}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.instrumentor.log_interaction(kwargs, response_obj)
        logger.debug(f"On Success with kwargs: {kwargs}")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        logger.debug(f"On Failure with kwargs: {kwargs}")


def test_litellm_with_custom_logger() -> None:

    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    px.launch_app()

    phoenix_logger = PhoenixLogger()

    # Assign the custom logger to LiteLLM
    litellm.callbacks = [phoenix_logger]

    questions = [
        "How can Arize AI help me evaluate my LLM models?",
        "What are the key features of Arize AI for model monitoring?",
        "Can Arize AI handle real-time data for model performance analysis?",
    ]
    for current_question in questions:
        messages = [{"content": current_question, "role": "user"}]
        response = litellm.completion(model="gpt-4-turbo", messages=messages, stream=False)
        logger.debug(f"Received response: {response}")


def test_litellm_with_phoenix_logger() -> None:

    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    px.launch_app()

    # Assign the custom logger to LiteLLM
    litellm.success_callback = ["phoenix"]

    questions = [
        "How can Arize AI help me evaluate my LLM models?",
        "What are the key features of Arize AI for model monitoring?",
        "Can Arize AI handle real-time data for model performance analysis?",
    ]
    for current_question in questions:
        messages = [{"content": current_question, "role": "user"}]
        response = litellm.completion(model="gpt-4-turbo", messages=messages, stream=False)
        logger.debug(f"Received response: {response}")


if __name__ == '__main__':
    #test_litellm_with_custom_logger()
    test_litellm_with_phoenix_logger()
    print('Tests finished!')

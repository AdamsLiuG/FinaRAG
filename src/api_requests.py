import os
import json
import ipaddress
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_fixed


def _env_flag(*names: str, default: bool = False) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default



class BaseCompatibleProcessor:
    def __init__(self, provider: str = "qwen"):
        load_dotenv()
        self.provider = provider.lower()
        self.api_key = self._get_env_value(f"{self.provider.upper()}_API_KEY", "LLM_API_KEY")
        self.base_url = self._get_env_value(f"{self.provider.upper()}_BASE_URL", "LLM_BASE_URL")
        self.default_model = self._get_env_value(
            f"{self.provider.upper()}_MODEL",
            "LLM_MODEL",
            default="Qwen/Qwen2.5-72B-Instruct"
        )
        max_tokens = self._get_env_value(
            f"{self.provider.upper()}_MAX_TOKENS",
            "LLM_MAX_TOKENS"
        )
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.use_stream = _env_flag(
            f"{self.provider.upper()}_STREAM",
            "LLM_STREAM",
            default=self.provider == "qwen"
        )
        self.enable_thinking = _env_flag(
            f"{self.provider.upper()}_ENABLE_THINKING",
            "LLM_ENABLE_THINKING",
            default=False if self.provider == "qwen" else True
        )

    @staticmethod
    def _get_env_value(*names: str, default: str = None) -> Optional[str]:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return default

    def _get_request_url(self) -> str:
        if not self.base_url:
            raise ValueError(
                f"Missing API base URL for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_BASE_URL or LLM_BASE_URL."
            )

        normalized_base = self.base_url.rstrip("/")
        if normalized_base.endswith("/chat/completions"):
            return normalized_base
        return f"{normalized_base}/chat/completions"

    def _is_dashscope_compatible(self) -> bool:
        return bool(self.base_url and "dashscope.aliyuncs.com" in self.base_url)

    def _should_bypass_env_proxy(self) -> bool:
        if not self.base_url:
            return False

        hostname = urlparse(self.base_url).hostname
        if not hostname:
            return False

        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return True

        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return hostname.endswith(".local")

    def _parse_structured_response(self, response_text: str, response_format: Type[BaseModel], model: str) -> Dict:
        def _try_parse(raw: str) -> Dict:
            """Repair JSON, decode, and validate; raises on any failure including null."""
            repaired = repair_json(raw)
            parsed = json.loads(repaired)
            if parsed is None:
                raise ValueError(f"Model returned JSON null: {raw!r}")
            validated = response_format.model_validate(parsed)
            return validated.model_dump()

        try:
            return _try_parse(response_text)
        except Exception:
            reparsed_response = self._reparse_response(response_text, response_format, model)
            try:
                return _try_parse(reparsed_response)
            except Exception:
                raise ValueError(
                    f"Structured response parsing failed after reparsing attempt. "
                    f"Raw model output was: {reparsed_response!r}"
                )

    @staticmethod
    def _augment_system_prompt_with_schema(system_content: str, response_format: Type[BaseModel]) -> str:
        if "strictly follow this schema" in system_content.lower():
            return system_content

        schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
        schema_instruction = (
            "\n\n---\n\n"
            "Return valid JSON only. The JSON must strictly follow this schema:\n"
            f"```json\n{schema}\n```"
        )
        return f"{system_content}{schema_instruction}"

    def _reparse_response(self, response: str, response_format: Type[BaseModel], model: str) -> Union[str, Dict]:
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        reparsed = self.send_message(
            model=model,
            temperature=0,
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        if isinstance(reparsed, str):
            return reparsed
        return json.dumps(reparsed, ensure_ascii=False)

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None
    ):
        if model is None:
            model = self.default_model
        if is_structured and response_format is not None:
            system_content = self._augment_system_prompt_with_schema(system_content, response_format)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        if seed is not None:
            payload["seed"] = seed
        if temperature is not None:
            payload["temperature"] = temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.provider == "qwen":
            if self.use_stream:
                payload["stream"] = True
            if self._is_dashscope_compatible():
                payload["enable_thinking"] = self.enable_thinking
            elif not self.enable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_kwargs = {
            "url": self._get_request_url(),
            "headers": headers,
            "json": payload,
            "timeout": 300,
            "stream": bool(payload.get("stream")),
        }

        if self._should_bypass_env_proxy():
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(**request_kwargs)
        else:
            response = requests.post(**request_kwargs)
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} Client Error for url: {response.url}\n{response.text[:1000]}",
                response=response
            )

        if payload.get("stream"):
            content = self._read_streaming_content(response, model)
        else:
            completion = response.json()
            choices = completion.get("choices") or []
            if not choices:
                raise ValueError(f"Provider returned no choices. Response: {json.dumps(completion)[:1000]}")

            message_payload = choices[0].get("message") or {}
            message = message_payload.get("content")
            if isinstance(message, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message
                )
            else:
                content = message

            if not content:
                content = (
                    message_payload.get("reasoning")
                    or message_payload.get("reasoning_content")
                    or ""
                )

            usage = completion.get("usage", {})
            self.response_data = {
                "model": completion.get("model", model),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }
            print(self.response_data)

        if is_structured and response_format is not None:
            return self._parse_structured_response(content, response_format, model)

        return content

    def _read_streaming_content(self, response: requests.Response, model: str) -> str:
        collected_content = []
        collected_reasoning = []
        response_model = model
        usage = {}

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            response_model = chunk.get("model", response_model)
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            collected_content.append(delta.get("content") or "")
            collected_reasoning.append(
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or ""
            )

        final_content = "".join(collected_content).strip() or "".join(collected_reasoning).strip()
        if not final_content:
            raise ValueError("Streaming response completed without any content.")

        self.response_data = {
            "model": response_model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
        print(self.response_data)
        return final_content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "qwen"] ="qwen"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseCompatibleProcessor(provider="openai")
        elif self.provider == "qwen":
            self.processor = BaseCompatibleProcessor(provider="qwen")
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = self.provider in {"ibm", "gemini", "qwen"}
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        use_schema_prompt = self.provider in {"ibm", "gemini", "qwen"}
        system_prompt = (
            prompts.RephrasedQuestionsPrompt.system_prompt_with_schema
            if use_schema_prompt else prompts.RephrasedQuestionsPrompt.system_prompt
        )
        answer_dict = self.processor.send_message(
            system_content=system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict

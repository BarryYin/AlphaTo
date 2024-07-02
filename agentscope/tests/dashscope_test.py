# -*- coding: utf-8 -*-
"""dashscope test"""
import unittest
from unittest.mock import patch, MagicMock

from agentscope.models import (  # pylint: disable=no-name-in-module
    ModelResponse,
    DashScopeChatWrapper,
    DashScopeImageSynthesisWrapper,
    DashScopeTextEmbeddingWrapper,
)


class TestDashScopeChatWrapper(unittest.TestCase):
    """Test DashScope Chat Wrapper"""

    def setUp(self) -> None:
        self.config_name = "test_config"
        self.model_name = "test_model"
        self.api_key = "test_api_key"
        self.wrapper = DashScopeChatWrapper(
            config_name=self.config_name,
            model_name=self.model_name,
            api_key=self.api_key,
        )

    @patch("agentscope.models.dashscope_model.dashscope.Generation.call")
    def test_call_success(self, mock_generation_call: MagicMock) -> None:
        """Test call success"""
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.request_id = "test_request_id"
        mock_response.usage = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        mock_response.output = {
            "choices": [{"message": {"content": "Hello, world!"}}],
        }

        mock_generation_call.return_value = mock_response

        # Define test input
        messages = [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello!"},
        ]

        # Call the wrapper method
        response = self.wrapper(messages)

        # Verify the response
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.text, "Hello, world!")
        self.assertEqual(response.raw, mock_response)

        # Verify call to dashscope.Generation.call
        mock_generation_call.assert_called_once_with(
            model=self.model_name,
            messages=messages,
            result_format="message",
        )

    @patch("agentscope.models.dashscope_model.dashscope.Generation.call")
    def test_call_failure(self, mock_generation_call: MagicMock) -> None:
        """test call failure"""
        # Set up the mock response for a failed API call
        mock_response = MagicMock()
        mock_response.status_code = "400"
        mock_response.request_id = "Test_request_id"
        mock_response.code = "Error Code"
        mock_response.message = "Error Message"
        mock_generation_call.return_value = mock_response

        # Define test input
        messages = [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello!"},
        ]

        # Call the wrapper method and expect an exception
        with self.assertRaises(RuntimeError) as context:
            self.wrapper(messages)

        # Assert the expected exception message contains the error details
        self.assertIn("Error Code", str(context.exception))
        self.assertIn("Error Message", str(context.exception))
        self.assertIn("400", str(context.exception))
        self.assertIn("Test_request_id", str(context.exception))

        # Verify call to dashscope.Generation.call
        mock_generation_call.assert_called_once_with(
            model=self.model_name,
            messages=messages,
            result_format="message",
        )


class TestDashScopeImageSynthesisWrapper(unittest.TestCase):
    """Test DashScope Image Synthesis Wrapper"""

    def setUp(self) -> None:
        self.config_name = "config_name"
        self.model_name = "test_model"
        self.api_key = "test_api_key"
        self.wrapper = DashScopeImageSynthesisWrapper(
            config_name=self.config_name,
            model_name=self.model_name,
            api_key=self.api_key,
        )

    @patch(
        "agentscope.file_manager.file_manager.save_image",
        side_effect=lambda x: f'/local/path/{x.split("/")[-1]}',
    )
    @patch("agentscope.models.dashscope_model.dashscope.ImageSynthesis.call")
    def test_image_synthesis_wrapper_call_success(
        self,
        mock_call: MagicMock,
        mock_save_image: MagicMock,
    ) -> None:
        """Test call success"""
        # Set up the mock response for a successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = {
            "results": [{"url": "http://example.com/image.jpg"}],
        }
        mock_call.return_value = mock_response
        # Call the wrapper with prompt
        prompt = "Generate an image of a sunset"
        response = self.wrapper(prompt, save_local=False)
        # Check if response is an instance of ModelResponse and contains
        # expected data
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(
            response.image_urls,
            ["http://example.com/image.jpg"],
        )

        # Call the wrapper method with save_local set to True
        response_with_save = self.wrapper(prompt, save_local=True)

        # Verify save_image call and local image url in response
        mock_save_image.assert_called_with("http://example.com/image.jpg")
        self.assertEqual(
            response_with_save.image_urls,
            ["/local/path/image.jpg"],
        )

    @patch("agentscope.models.dashscope_model.dashscope.ImageSynthesis.call")
    def test_image_synthesis_wrapper_call_failure(
        self,
        mock_call: MagicMock,
    ) -> None:
        """Test call failure"""
        # Set up the mock response for a failed API call
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.request_id = "Test_request_id"
        mock_response.code = "Error Code"
        mock_response.message = "Error Message"
        mock_call.return_value = mock_response

        # Call the wrapper with prompt and expect a RuntimeError
        prompt = "Generate an image of a sunset"
        with self.assertRaises(RuntimeError) as context:
            self.wrapper(prompt, save_local=False)

        # Assert the expected exception message
        self.assertIn("Error Code", str(context.exception))
        self.assertIn("Error Message", str(context.exception))
        self.assertIn("Test_request_id", str(context.exception))
        self.assertIn("400", str(context.exception))

        # Verify call to dashscope.ImageSynthesis.call
        mock_call.assert_called_once_with(
            model=self.model_name,
            prompt=prompt,
            n=1,  # Assuming this is a default value used to call the API
        )


class TestDashScopeTextEmbeddingWrapper(unittest.TestCase):
    """Test DashScope Text Embedding Wrapper"""

    def setUp(self) -> None:
        # Initialize DashScopeTextEmbeddingWrapper instance
        self.wrapper = DashScopeTextEmbeddingWrapper(
            config_name="test_config",
            model_name="test_model",
            api_key="test_key",
        )

    @patch("agentscope.models.dashscope_model.dashscope.TextEmbedding.call")
    def test_call_success(self, mock_call: MagicMock) -> None:
        """Test call success"""
        # Mock a successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.usage = {
            "total_tokens": 8,
        }
        mock_response.output = {
            "embeddings": [{"embedding": [0.1, 0.2, 0.3]}],
        }
        mock_call.return_value = mock_response

        # Call the wrapper with a mock API
        texts = ["Hello, world!"]
        response = self.wrapper(texts)

        # Assert the API was called correctly
        mock_call.assert_called_once_with(
            input=texts,
            model=self.wrapper.model_name,
            **self.wrapper.generate_args,
        )

        # Assert the response is as expected
        self.assertEqual(response.embedding, [[0.1, 0.2, 0.3]])

    @patch("agentscope.models.dashscope_model.dashscope.TextEmbedding.call")
    def test_call_failure(self, mock_call: MagicMock) -> None:
        """Test call failure"""
        # Mock a failed API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.request_id = "Test_request_id"
        mock_response.code = "Error Code"
        mock_response.message = "Error Message"
        mock_call.return_value = mock_response

        # Call the wrapper with a mock API and expect an exception
        texts = ["Hello, world!"]
        with self.assertRaises(RuntimeError) as context:
            self.wrapper(texts)

        # Assert the expected exception message
        self.assertIn("Error Code", str(context.exception))
        self.assertIn("Error Message", str(context.exception))
        self.assertIn("Test_request_id", str(context.exception))
        self.assertIn("400", str(context.exception))

        # Verify call to dashscope.TextEmbedding.call
        mock_call.assert_called_once_with(
            input=texts,
            model=self.wrapper.model_name,
            **self.wrapper.generate_args,
        )


if __name__ == "__main__":
    unittest.main()

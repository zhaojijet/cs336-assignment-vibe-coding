import unittest
from mlx_lm import load, generate


class TestMLXInference(unittest.TestCase):
    def test_model_loading_and_generation(self):
        # Use a small model for testing if possible, but user has Qwen 1.5B setup
        model_path = "mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit"
        print(f"Loading model from {model_path}...")
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

        prompt = "User: What is 2 + 2?\nAssistant: <think>"
        print(f"Generating for prompt: {prompt}")

        try:
            response = generate(
                model, tokenizer, prompt=prompt, max_tokens=20, verbose=False
            )
            print(f"Response: {response}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            self.fail(f"Failed to generate text: {e}")


if __name__ == "__main__":
    unittest.main()

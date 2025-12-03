# test_model.py
from model_inference import Chatbot

def main():
    print("Testing model loading and generation...")
    
    try:
        bot = Chatbot()
        
        # Simple test prompts
        test_prompts = [
            "Write a very short story about the moon",
            "Create a simple story about stars",
            "Tell a short children's story about the night sky"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {prompt}")
            print(f"{'='*60}")
            
            response = bot.generate_response(prompt, max_new_tokens=100)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
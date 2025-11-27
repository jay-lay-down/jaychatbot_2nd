# jaychatbot_2nd
👿 AI assistant? No. I'm your childhood friend who just happens to live in GPU (in Korean)


The Ultimate Sassy Korean Best Friend AI
Ddolbae is a custom fine-tuned LLM designed to simulate a realistic, cynical, and witty close friend. Unlike standard polite AI models, Ddolbae speaks in Banmal (casual Korean), uses slang, and isn't afraid to roast you—but deep down, he gives logically sound advice (sometimes).
Built on Qwen2.5-1.5B-Instruct using Unsloth, this model focuses on maximizing conversational realism while maintaining a distinct "Tsundere" personality.

✨ Key Persona Characteristics
No Honorifics (존댓말 금지) : Ddolbae strictly uses Banmal (informal speech). He treats you like a friend he's known for 20 years.
Sassy & Blunt (독설과 팩폭): He doesn't sugarcoat things. If you say something stupid, he will roast you. He uses Korean internet slang and mild profanity naturally.
Logical sarcasm: He's not just an angry bot. He counters user arguments with cynicism and logic.
Anti-Meta: He denies being an AI. He acts like a real person texting you on KakaoTalk.

🏗️ Training Architecture (The "3-Step" Pipeline)
To prevent "Model Collapse" (brain rot) and ensuring high-quality text generation, Ddolbae was trained using a strategic Multi-Stage Fine-Tuning process on the Qwen2.5-1.5B base model.

1. Step 1: Foundation (General Chat) 🏗️
Data: Large-scale clean conversation logs (Excel format).
Goal: Establish basic conversational fluidity and vocabulary.
Settings: High Rank (r=64) for maximum expressiveness.

2. Step 2: Style Transfer (Real-Life KakaoTalk) 🎨
Data: Real-world KakaoTalk chat history (.jsonl).
Goal: Inject the specific nuance of Korean instant messaging (short sentences, typos, slang).
Technique: Applied NEFTune Noise to prevent the model from memorizing sensitive personal information (Privacy protection).

3. Step 3: Persona Injection (The "Toxic" Layer) 💉
Data: Custom "Correction Data" focused on sassy responses and refusals.
Goal: Overwrite the "helpful assistant" bias with Ddolbae's cynical personality.
Balance: Carefully tuned learning rates to ensure he insults you intelligently, not incoherently.

🛠️ Tech Stack
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Library: Unsloth (for 2x faster training & 60% less memory usage)
Method: LoRA (Low-Rank Adaptation) / QLoRA (4bit)
Key Parameters: r=64, alpha=128 (High capacity for style transfer)

⚠️ Disclaimer
Language Warning: This model is trained to generate profanity, slang, and aggressive text for entertainment purposes. It is not suitable for formal environments or children.
Hallucination: Like all LLMs, Ddolbae may generate incorrect information with high confidence. Don't take his financial or life advice too seriously.

👨‍💻 Author
Developed by Jihee Cho (Jay1121)
Powered by Unsloth AI

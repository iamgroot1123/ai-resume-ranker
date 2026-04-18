import os
from openai import OpenAI

# ---------------------------------------------------------
# PASTE YOUR API KEYS IN THIS LIST (DON'T COMMIT THIS FILE!)
# ---------------------------------------------------------
KEYS = [
    "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"
]

def check_key(key: str):
    if not key or not key.startswith("sk-"):
        print(f"⚠️  Skipping malformed key: {key}")
        return

    print(f"\n=======================================================")
    print(f"Testing key: {key[:12]}...{key[-5:]}")
    try:
        client = OpenAI(api_key=key)
        
        # This will throw an AuthenticationError if the key is completely invalid
        models_response = client.models.list()
        model_ids = {m.id for m in models_response.data}
        
        print(f"✅ Status: VALID & AUTHENTICATED")
        print(f"-------------------------------------------------------")
        print("Model Access:")
        
        targets = ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
        for t in targets:
            if t in model_ids:
                print(f"  🟢 {t:<20} -> Supported")
            else:
                print(f"  🔴 {t:<20} -> Not Supported")
                
    except Exception as e:
        print(f"❌ Status: INVALID OR REJECTED")
        print(f"   Reason: {str(e)}")
        
if __name__ == "__main__":
    if not KEYS or len(KEYS) == 0 or KEYS[0].startswith("#"):
        print("\n[INFO] Please open 'check_keys.py' in your editor and paste your API keys into the KEYS array.")
        print("[INFO] Then run: source venv/bin/activate && python check_keys.py\n")
    else:
        print(f"Testing {len(KEYS)} keys...\n")
        for k in KEYS:
            check_key(k.strip())
        print("\n=======================================================\nDone!")

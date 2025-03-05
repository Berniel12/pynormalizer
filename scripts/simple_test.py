#!/usr/bin/env python3
"""
Very simple script to test the OpenAI API directly.
"""
import os
import sys
import json
from datetime import datetime

print("Starting simple OpenAI test...")

# Try to install the OpenAI package if not already installed
try:
    from openai import OpenAI
    print("OpenAI package is already installed")
except ImportError:
    print("Installing OpenAI package...")
    os.system("pip install openai")
    from openai import OpenAI
    print("OpenAI package installed successfully")

# Check if API key is provided
if len(sys.argv) > 1:
    api_key = sys.argv[1]
    print("Using API key from command line argument")
elif "OPENAI_API_KEY" in os.environ:
    api_key = os.environ["OPENAI_API_KEY"]
    print("Using API key from environment variable")
else:
    print("Error: No API key provided")
    print("Usage: python simple_test.py YOUR_API_KEY")
    sys.exit(1)

# Create a simple test message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, can you help me normalize some tender data?"}
]

# Create OpenAI client
client = OpenAI(api_key=api_key)
print("Created OpenAI client")

# Call the API
print("Calling OpenAI API...")
try:
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=100
    )
    
    # Print the response
    print("\nResponse from OpenAI API:")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage.total_tokens} tokens")
    print(f"Content: {response.choices[0].message.content}")
    
    # Save the response to a file
    debug_dir = "debug_dumps"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{debug_dir}/simple_test_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "content": response.choices[0].message.content
        }, f, indent=2)
    
    print(f"Response saved to {filename}")
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error calling OpenAI API: {str(e)}")
    print("Test failed!") 
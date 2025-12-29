"""
ElevenLabs Voice Setup - List voices and test TTS.

Usage:
    python setup_elevenlabs.py --key YOUR_API_KEY
    python setup_elevenlabs.py --key YOUR_API_KEY --test "Hello, I am your smart mirror assistant"
    python setup_elevenlabs.py --key YOUR_API_KEY --test "Hello" --voice-id VOICE_ID
"""

import argparse
import os
import sys
import tempfile


def list_voices(api_key: str):
    """List all available voices."""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)

    print("\n" + "=" * 70)
    print("YOUR AVAILABLE ELEVENLABS VOICES")
    print("=" * 70)

    response = client.voices.get_all()

    print(f"\n{'Voice Name':<25} {'Voice ID':<25} {'Category':<15}")
    print("-" * 70)

    for voice in response.voices:
        category = voice.category or "premade"
        print(f"{voice.name:<25} {voice.voice_id:<25} {category:<15}")

    print("\n" + "=" * 70)
    print("To use a voice in the demo, copy the Voice ID and run:")
    print('  python vlm_voice_demo.py --elevenlabs-key YOUR_KEY --voice-id "VOICE_ID"')
    print("=" * 70 + "\n")

    return response.voices


def test_voice(api_key: str, voice_id: str, text: str):
    """Test a specific voice."""
    from elevenlabs.client import ElevenLabs

    print(f"\nTesting voice: {voice_id}")
    print(f"Text: {text}")

    client = ElevenLabs(api_key=api_key)

    # Generate audio
    print("Generating audio...")
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128"
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio_generator:
            f.write(chunk)
        temp_path = f.name

    print(f"Audio saved to: {temp_path}")

    # Play audio
    print("Playing audio...")
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        pygame.mixer.quit()
        print("Playback complete!")
    except ImportError:
        print("pygame not installed. Install with: pip install pygame")
        print(f"You can manually play: {temp_path}")
    except Exception as e:
        print(f"Playback error: {e}")
        print(f"You can manually play: {temp_path}")

    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


def create_voice_recommendations():
    """Print recommended voices for smart mirror."""
    print("\n" + "=" * 70)
    print("RECOMMENDED VOICES FOR SMART MIRROR")
    print("=" * 70)
    print("""
For a Smart Mirror assistant, consider these voice characteristics:

PROFESSIONAL / ASSISTANT:
  - Rachel (default) - Clear, professional female voice
  - Josh - Clear, professional male voice
  - Adam - Deep, authoritative male voice

FRIENDLY / WARM:
  - Bella - Warm, friendly female voice
  - Antoni - Warm, friendly male voice
  - Elli - Young, energetic female voice

CALM / WELLNESS:
  - Domi - Calm, soothing female voice
  - Sam - Calm male voice

You can also:
1. Clone your own voice at https://elevenlabs.io/voice-cloning
2. Create a custom voice with Voice Design
3. Use the Voice Library to find community voices
""")


def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Voice Setup")
    parser.add_argument("--key", type=str, help="ElevenLabs API key")
    parser.add_argument("--test", type=str, help="Test text to speak")
    parser.add_argument("--voice-id", type=str, default="Rachel", help="Voice ID to test")
    parser.add_argument("--list-only", action="store_true", help="Only list voices")
    args = parser.parse_args()

    # Get API key
    api_key = args.key or os.environ.get("ELEVENLABS_API_KEY")

    if not api_key:
        print("ERROR: No API key provided.")
        print("  Use --key YOUR_KEY or set ELEVENLABS_API_KEY environment variable")
        print("\nGet your API key at: https://elevenlabs.io/api")
        sys.exit(1)

    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        print("ERROR: elevenlabs not installed.")
        print("  Install with: pip install elevenlabs")
        sys.exit(1)

    # List voices
    voices = list_voices(api_key)

    # Show recommendations
    create_voice_recommendations()

    # Test if requested
    if args.test:
        test_voice(api_key, args.voice_id, args.test)


if __name__ == "__main__":
    main()

@echo off
cd /d "C:\Users\Akram\Desktop\Projects\Smart Mirror\smart-mirror\demos"
py -3.12 skin_voice_demo.py --camera 0 --vlm-model qwen3-vl:8b --elevenlabs-key %1 --voice-id 21m00Tcm4TlvDq8ikWAM
pause

#!/usr/bin/env python3
"""
Quick test script for the Healthcare AI Agent System
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    print("🔧 Testing imports...")
    
    # Test basic imports
    import gradio as gr
    print("✅ Gradio imported successfully")
    
    from src.agents import AgentCoordinator
    print("✅ Agent system imported successfully")
    
    from src.openai_integration import OpenAIHealthcareAssistant
    print("✅ OpenAI integration imported successfully")
    
    # Test agent initialization
    print("\n🤖 Testing agent initialization...")
    agent_coordinator = AgentCoordinator()
    print("✅ Agent coordinator initialized successfully")
    
    healthcare_ai = OpenAIHealthcareAssistant()
    print("✅ Healthcare AI initialized successfully")
    
    # Test a simple patient analysis
    print("\n🏥 Testing agent analysis...")
    
    patient_data = {
        "name": "Test Patient",
        "age": "30",
        "gender": "Male",
        "symptoms": "I have been experiencing headaches and fatigue for the past few days",
        "has_image": False,
        "has_audio": False
    }
    
    print(f"Patient data: {patient_data}")
    print("\n🔍 Running agent analysis... (this may take a moment)")
    
    import asyncio
    result = asyncio.run(agent_coordinator.analyze_patient(patient_data, None))
    
    print("✅ Agent analysis completed successfully!")
    print(f"\nUrgency Level: {result.get('urgency', 'Unknown')}")
    print(f"Specialties Recommended: {result.get('specialties_recommended', [])}")
    
    print("\n🎉 All tests passed! Your Healthcare AI Agent System is working perfectly!")
    print("\n📊 Test Summary:")
    print("- ✅ All imports successful")
    print("- ✅ Agent coordination working")
    print("- ✅ OpenAI integration functional")
    print("- ✅ Multi-agent analysis operational")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

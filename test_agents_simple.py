#!/usr/bin/env python3
"""
Simple Agent Testing Script
Tests each agent with direct examples you can run easily
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_agents_simply():
    """Simple test of all agents with readable output"""
    
    print("🏥 SIMPLE HEALTHCARE AI AGENT TESTS")
    print("=" * 60)
    
    # Test Agent Coordinator (main system)
    print("\n🎯 Testing Full Agent Coordinator System:")
    print("-" * 50)
    
    from src.agents import AgentCoordinator
    coordinator = AgentCoordinator()
    
    # Test Emergency Case
    print("\n📋 Emergency Test Case:")
    emergency_data = {
        "name": "John Emergency",
        "age": "45",
        "gender": "Male", 
        "symptoms": "Severe chest pain, difficulty breathing, sweating profusely",
        "has_image": False,
        "has_audio": False
    }
    
    try:
        result = await coordinator.analyze_patient(emergency_data)
        print(f"✅ Analysis completed!")
        print(f"📊 Response keys: {list(result.keys())}")
        if 'agent_results' in result:
            for agent_name, agent_result in result['agent_results'].items():
                print(f"\n🤖 {agent_name.upper()} AGENT:")
                print(f"📝 Analysis: {agent_result.get('analysis', 'No analysis')[:200]}...")
        
        # Test follow-up
        if result:
            print(f"\n💬 Testing Follow-up Question:")
            followup = await coordinator.handle_followup_question(
                "How urgent is this? Should I go to the ER?", 
                result,
                emergency_data
            )
            print(f"✅ Follow-up response: {followup.get('response', 'No response')[:150]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Dermatology Case
    print("\n📋 Dermatology Test Case:")
    skin_data = {
        "name": "Sarah Skin",
        "age": "30",
        "gender": "Female",
        "symptoms": "Dark mole on back changed color and size, irregular edges",
        "has_image": False,
        "has_audio": False
    }
    
    try:
        result = await coordinator.analyze_patient(skin_data)
        print(f"✅ Analysis completed!")
        if 'agent_results' in result:
            for agent_name, agent_result in result['agent_results'].items():
                print(f"\n🤖 {agent_name.upper()} AGENT:")
                print(f"📝 Analysis: {agent_result.get('analysis', 'No analysis')[:200]}...")
                
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 SIMPLE AGENT TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.startswith('sk-'):
        print("❌ Error: OpenAI API key not found")
        print("Make sure OPENAI_API_KEY is set in your .env file")
        exit(1)
        
    print("🔑 API key found - starting simple tests...")
    asyncio.run(test_agents_simply())

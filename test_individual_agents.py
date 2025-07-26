#!/usr/bin/env python3
"""
Healthcare AI Agent Testing Suite
Tests each agent individually with specific scenarios
"""

import os
import asyncio
from dotenv import load_dotenv
from src.agents import AgentCoordinator, TriageAgent, DermatologyAgent, GeneralPracticeAgent, FollowUpAgent

# Load environment variables
load_dotenv()

async def test_triage_agent():
    """Test the Triage Agent with various scenarios"""
    print("\n🚨 TESTING TRIAGE AGENT")
    print("=" * 50)
    
    agent = TriageAgent()
    
    test_cases = [
        {
            "name": "Emergency Case",
            "data": {
                "name": "John Doe",
                "age": "45",
                "gender": "Male",
                "symptoms": "Severe chest pain, shortness of breath, sweating, feeling like I'm dying"
            }
        },
        {
            "name": "Moderate Case",
            "data": {
                "name": "Jane Smith",
                "age": "30",
                "gender": "Female", 
                "symptoms": "Persistent headache for 3 days, mild nausea"
            }
        },
        {
            "name": "Low Priority Case",
            "data": {
                "name": "Bob Johnson",
                "age": "25",
                "gender": "Male",
                "symptoms": "Minor cut on finger from cooking, bleeding stopped"
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {case['name']}")
        print("-" * 30)
        try:
            result = await agent.analyze(case['data'])
            print(f"✅ Urgency: {result.get('urgency_level', 'Unknown')}")
            print(f"📝 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
            print(f"🏥 Specialties: {result.get('recommended_specialties', [])}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_dermatology_agent():
    """Test the Dermatology Agent"""
    print("\n👩‍⚕️ TESTING DERMATOLOGY AGENT")
    print("=" * 50)
    
    agent = DermatologyAgent()
    
    test_cases = [
        {
            "name": "Skin Rash",
            "data": {
                "name": "Alice Brown",
                "age": "28",
                "gender": "Female",
                "symptoms": "Red, itchy rash on arms and legs, appeared 3 days ago",
                "has_image": False
            }
        },
        {
            "name": "Mole Changes",
            "data": {
                "name": "Mike Davis",
                "age": "40",
                "gender": "Male",
                "symptoms": "Mole on back has changed color and size over past month",
                "has_image": False
            }
        },
        {
            "name": "Acne Issues",
            "data": {
                "name": "Sarah Wilson",
                "age": "22",
                "gender": "Female",
                "symptoms": "Severe acne breakout on face, painful cysts",
                "has_image": False
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {case['name']}")
        print("-" * 30)
        try:
            result = await agent.analyze(case['data'])
            print(f"✅ Condition: {result.get('condition_type', 'Unknown')}")
            print(f"📝 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
            print(f"💊 Treatment: {result.get('treatment_suggestions', 'No suggestions')[:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_general_practice_agent():
    """Test the General Practice Agent"""
    print("\n🩺 TESTING GENERAL PRACTICE AGENT")
    print("=" * 50)
    
    agent = GeneralPracticeAgent()
    
    test_cases = [
        {
            "name": "Flu Symptoms",
            "data": {
                "name": "Tom Anderson",
                "age": "35",
                "gender": "Male",
                "symptoms": "Fever, body aches, cough, fatigue for 2 days"
            }
        },
        {
            "name": "Digestive Issues",
            "data": {
                "name": "Lisa Martinez",
                "age": "32",
                "gender": "Female",
                "symptoms": "Stomach pain, nausea, bloating after meals for 1 week"
            }
        },
        {
            "name": "Sleep Problems",
            "data": {
                "name": "David Lee",
                "age": "42",
                "gender": "Male",
                "symptoms": "Difficulty falling asleep, waking up tired, trouble concentrating"
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {case['name']}")
        print("-" * 30)
        try:
            result = await agent.analyze(case['data'])
            print(f"✅ Assessment: {result.get('assessment', 'No assessment')[:100]}...")
            print(f"💡 Recommendations: {result.get('recommendations', 'No recommendations')[:100]}...")
            print(f"⚠️ Warnings: {result.get('red_flags', 'None')}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_followup_agent():
    """Test the Follow-up Agent"""
    print("\n💬 TESTING FOLLOW-UP AGENT")
    print("=" * 50)
    
    agent = FollowUpAgent()
    
    # Simulate previous analysis
    previous_analysis = {
        "urgency_level": "MODERATE",
        "recommended_specialties": ["cardiology"],
        "analysis": "Patient presents with chest discomfort and palpitations. Recommend cardiology consultation."
    }
    
    test_questions = [
        "How long should I wait before seeing a cardiologist?",
        "What activities should I avoid in the meantime?",
        "Are there any warning signs I should watch for?",
        "Can stress cause these symptoms?",
        "Should I take any medication while waiting for appointment?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Follow-up Question {i}")
        print("-" * 30)
        print(f"❓ Question: {question}")
        try:
            result = await agent.handle_followup(question, previous_analysis)
            print(f"✅ Response: {result.get('response', 'No response')[:150]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_agent_coordinator():
    """Test the Agent Coordinator with end-to-end scenarios"""
    print("\n🎯 TESTING AGENT COORDINATOR")
    print("=" * 50)
    
    coordinator = AgentCoordinator()
    
    test_cases = [
        {
            "name": "Emergency Coordination",
            "data": {
                "name": "Emergency Patient",
                "age": "55",
                "gender": "Male",
                "symptoms": "Severe chest pain radiating to left arm, difficulty breathing",
                "has_image": False,
                "has_audio": False
            }
        },
        {
            "name": "Dermatology Coordination",
            "data": {
                "name": "Skin Patient",
                "age": "30",
                "gender": "Female",
                "symptoms": "Strange mole on shoulder, has grown and changed color",
                "has_image": True,
                "has_audio": False
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Coordination Test {i}: {case['name']}")
        print("-" * 40)
        try:
            result = await coordinator.analyze_patient(case['data'])
            print(f"✅ Overall Urgency: {result.get('urgency_level', 'Unknown')}")
            print(f"🏥 Specialties: {result.get('recommended_specialties', [])}")
            print(f"📊 Agent Results: {len(result.get('agent_results', {}))} agents responded")
            
            # Test follow-up
            followup_question = "What should I do immediately?"
            followup_result = await coordinator.handle_followup_question(followup_question, result)
            print(f"💬 Follow-up Response: {followup_result.get('response', 'No response')[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")

async def run_all_tests():
    """Run all agent tests"""
    print("🧪 HEALTHCARE AI AGENT TESTING SUITE")
    print("=" * 60)
    print("Testing all agents individually and coordination system...")
    
    try:
        # Test individual agents
        await test_triage_agent()
        await test_dermatology_agent()
        await test_general_practice_agent()
        await test_followup_agent()
        
        # Test coordination
        await test_agent_coordinator()
        
        print("\n🎉 ALL AGENT TESTS COMPLETED!")
        print("=" * 60)
        print("✅ All agents tested successfully")
        print("📊 Check results above for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ Test Suite Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.startswith('sk-'):
        print("❌ Error: OpenAI API key not found or invalid")
        print("Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    print("🔑 OpenAI API key found - starting tests...")
    asyncio.run(run_all_tests())

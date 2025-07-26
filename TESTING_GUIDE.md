# 🧪 Healthcare AI Agent Testing Guide

## Quick Test Commands

### 1. Run Individual Agent Tests (Automated)
```bash
cd /Users/gauravvraii/AA_LAST_SEM/CareerBot/healthcare_ai_assistant/healthcare-ai-assistant
PYTHONPATH=. /Users/gauravvraii/AA_LAST_SEM/CareerBot/healthcare_ai_assistant/venv/bin/python test_individual_agents.py
```

### 2. Run Basic Agent Test (Quick)
```bash
PYTHONPATH=. /Users/gauravvraii/AA_LAST_SEM/CareerBot/healthcare_ai_assistant/venv/bin/python test_agents.py
```

### 3. Launch App for Manual Testing
```bash
PYTHONPATH=. /Users/gauravvraii/AA_LAST_SEM/CareerBot/healthcare_ai_assistant/venv/bin/python app.py
```

## Manual Testing Scenarios (In Gradio Interface)

### 🚨 **TRIAGE AGENT Testing**

#### Test Case 1: Emergency
- **Name**: John Emergency
- **Age**: 45
- **Gender**: Male
- **Symptoms**: "Severe chest pain, difficulty breathing, sweating profusely, feels like heart attack"
- **Expected**: Urgency = EMERGENCY, Specialty = cardiology

#### Test Case 2: High Priority
- **Name**: Sarah Urgent
- **Age**: 28
- **Gender**: Female
- **Symptoms**: "Severe abdominal pain, vomiting blood, very weak"
- **Expected**: Urgency = HIGH, Specialty = gastroenterology

#### Test Case 3: Moderate
- **Name**: Mike Moderate
- **Age**: 35
- **Gender**: Male
- **Symptoms**: "Persistent headaches for 1 week, mild nausea"
- **Expected**: Urgency = MODERATE, Specialty = neurology

#### Test Case 4: Low Priority
- **Name**: Lisa Minor
- **Age**: 25
- **Gender**: Female
- **Symptoms**: "Small cut on finger, bleeding stopped, just want advice"
- **Expected**: Urgency = LOW, Specialty = general practice

---

### 👩‍⚕️ **DERMATOLOGY AGENT Testing**

#### Test Case 1: Skin Cancer Concern
- **Name**: Bob Skin
- **Age**: 50
- **Gender**: Male
- **Symptoms**: "Mole on back changed color from brown to black, getting larger, irregular edges"
- **Expected**: High priority dermatology referral

#### Test Case 2: Rash/Allergic Reaction
- **Name**: Amy Rash
- **Age**: 30
- **Gender**: Female
- **Symptoms**: "Red itchy rash all over arms and legs, started after trying new detergent"
- **Expected**: Allergy assessment, treatment suggestions

#### Test Case 3: Acne Treatment
- **Name**: Teen Acne
- **Age**: 17
- **Gender**: Female
- **Symptoms**: "Severe acne breakout on face and back, painful cysts, affecting self-esteem"
- **Expected**: Acne treatment recommendations

#### Test Case 4: Eczema Management
- **Name**: Child Eczema
- **Age**: 8
- **Gender**: Male
- **Symptoms**: "Dry, scaly patches on elbows and knees, very itchy, worse at night"
- **Expected**: Eczema management advice

---

### 🩺 **GENERAL PRACTICE AGENT Testing**

#### Test Case 1: Flu Symptoms
- **Name**: Tom Flu
- **Age**: 32
- **Gender**: Male
- **Symptoms**: "Fever 101°F, body aches, cough, fatigue for 3 days, no appetite"
- **Expected**: Flu management, rest recommendations

#### Test Case 2: Digestive Issues
- **Name**: Maria Stomach
- **Age**: 40
- **Gender**: Female
- **Symptoms**: "Stomach pain after eating, bloating, nausea, constipation for 2 weeks"
- **Expected**: Digestive assessment, dietary advice

#### Test Case 3: Mental Health
- **Name**: Alex Stress
- **Age**: 28
- **Gender**: Non-binary
- **Symptoms**: "Constant anxiety, trouble sleeping, difficulty concentrating at work"
- **Expected**: Mental health support, referrals

#### Test Case 4: Preventive Care
- **Name**: Helen Wellness
- **Age**: 55
- **Gender**: Female
- **Symptoms**: "No specific symptoms, just want general health checkup advice"
- **Expected**: Preventive care recommendations

---

### 💬 **FOLLOW-UP AGENT Testing**

After any analysis, test these follow-up questions:

#### General Follow-up Questions:
1. "How urgent is this really? Should I go to ER?"
2. "What can I do at home to feel better?"
3. "How long should I wait before seeing a doctor?"
4. "What symptoms should worry me more?"
5. "Are there any medications I should avoid?"

#### Specific Follow-up Questions:
1. "Can you explain what [medical term] means?"
2. "What tests might the doctor order?"
3. "Is this condition contagious?"
4. "Will this affect my daily activities?"
5. "Should I tell my family about this?"

---

### 🖼️ **IMAGE ANALYSIS Testing**

#### Test with Skin Images:
1. Upload any skin condition image
2. Describe symptoms related to the image
3. Check if dermatology agent is triggered
4. Verify image analysis in results

#### Test with General Medical Images:
1. Upload X-ray, wound, or other medical image
2. Include relevant symptoms
3. Check appropriate agent routing

---

### 🎵 **AUDIO ANALYSIS Testing**

1. Record voice describing symptoms
2. Test with different accents/speeds
3. Verify transcription accuracy
4. Check if analysis matches voice content

---

## Expected Agent Behavior Patterns

### 🚨 **Triage Agent Should:**
- Classify urgency correctly (LOW → EMERGENCY)
- Recommend appropriate specialties
- Identify red flags and emergencies
- Route severe cases properly

### 👩‍⚕️ **Dermatology Agent Should:**
- Focus on skin-related conditions
- Suggest dermatology referrals when needed
- Provide skin care advice
- Identify concerning skin changes

### 🩺 **General Practice Agent Should:**
- Handle common medical conditions
- Provide lifestyle recommendations
- Suggest appropriate specialists when needed
- Give general health advice

### 💬 **Follow-up Agent Should:**
- Answer relevant follow-up questions
- Provide clarifications
- Maintain context from previous analysis
- Offer additional guidance

---

## Testing Checklist

### ✅ **Basic Functionality**
- [ ] All agents initialize without errors
- [ ] OpenAI API connection works
- [ ] Each agent responds to appropriate cases
- [ ] Urgency classification works correctly

### ✅ **Agent Coordination**
- [ ] Cases route to correct agents
- [ ] Multiple agents can analyze same case
- [ ] Results are properly combined
- [ ] Follow-up maintains context

### ✅ **Edge Cases**
- [ ] Very short symptom descriptions
- [ ] Very long symptom descriptions
- [ ] Unclear or ambiguous symptoms
- [ ] Multiple unrelated symptoms

### ✅ **Error Handling**
- [ ] Invalid input handling
- [ ] API rate limit handling
- [ ] Network error recovery
- [ ] Graceful failure modes

---

## Performance Expectations

### ⏱️ **Response Times**
- Triage Agent: 3-8 seconds
- Dermatology Agent: 5-12 seconds
- General Practice Agent: 4-10 seconds
- Follow-up Agent: 3-7 seconds

### 🎯 **Accuracy Expectations**
- Emergency detection: >95%
- Specialty routing: >90%
- General recommendations: >85%
- Follow-up relevance: >90%

---

## Troubleshooting

### Common Issues:
1. **"OpenAI API Error"**: Check API key in .env file
2. **"Agent timeout"**: Check internet connection
3. **"Import errors"**: Verify PYTHONPATH is set
4. **"No response"**: Check if symptoms are clear enough

### Debug Commands:
```bash
# Check imports
python -c "from src.agents import AgentCoordinator; print('OK')"

# Check API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key found:', bool(os.getenv('OPENAI_API_KEY')))"

# Test basic functionality
python test_agents.py
```

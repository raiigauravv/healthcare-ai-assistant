# 🏥 Healthcare AI Assistant

A sophisticated multimodal AI-powered healthcare assistant that provides preliminary health analysis based on text descriptions, medical images, and audio recordings. Built with OpenAI's latest models and deployed on Hugging Face Spaces.

![Healthcare AI Assistant](https://img.shields.io/badge/Status-Live%20Demo-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4%2BVision%2BWhisper-purple)

## 🚀 Live Demo

**[Try the Healthcare AI Assistant](https://huggingface.co/spaces/gauravvraii/healthcare-ai-assistant)**

## ⚠️ Important Disclaimer

This is a **demonstration tool for educational purposes only**. It does not provide medical advice and should not be used as a substitute for professional medical consultation, diagnosis, or treatment. Always seek advice from qualified healthcare professionals.

## 🎯 Features

### 🔍 **Multimodal Analysis**
- **Text Input**: Describe symptoms in natural language
- **Image Analysis**: Upload medical photos (rashes, wounds, swelling, etc.)
- **Audio Input**: Voice descriptions of symptoms using speech-to-text
- **Audio-Only Mode**: Complete analysis from voice input alone

### 👤 **Personalized Experience**
- **Patient Information**: Name, age, and gender integration
- **Personalized Responses**: AI addresses patients by name
- **Demographics-Aware**: Age and gender-relevant medical insights
- **Contextual Analysis**: Tailored recommendations based on patient profile

### 🤖 **Advanced AI Integration**
- **GPT-4**: Comprehensive health analysis and recommendations
- **GPT-4 Vision**: Medical image interpretation
- **Whisper AI**: High-accuracy speech transcription
- **Real-time Processing**: Instant analysis and feedback

## 🛠️ Technology Stack

### **Frontend & UI**
- **[Gradio 4.0+](https://gradio.app/)**: Modern web interface with healthcare theme
- **HTML/CSS**: Custom styling for medical application aesthetics
- **JavaScript**: Interactive elements and real-time updates

### **Backend & AI**
- **[Python 3.11+](https://python.org/)**: Core application language
- **[OpenAI API v1.0+](https://openai.com/)**: Advanced AI models integration
  - **GPT-4**: Text analysis and medical reasoning
  - **GPT-4 Vision**: Medical image analysis
  - **Whisper**: Speech-to-text transcription

### **Data Processing**
- **[PIL (Pillow)](https://pillow.readthedocs.io/)**: Image processing and format conversion
- **[Librosa](https://librosa.org/)**: Audio processing and analysis
- **[NumPy](https://numpy.org/)**: Numerical computing and array operations

### **Deployment & Infrastructure**
- **[Hugging Face Spaces](https://huggingface.co/spaces)**: Cloud deployment platform
- **Git**: Version control and collaboration
- **Virtual Environments**: Isolated Python environments

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │    │   Gradio UI      │    │  OpenAI APIs    │
│                 │    │                  │    │                 │
│ • Text          │───▶│ • Interface      │───▶│ • GPT-4         │
│ • Image         │    │ • Validation     │    │ • GPT-4 Vision  │
│ • Audio         │    │ • Processing     │    │ • Whisper       │
│ • Demographics  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐            │
         │              │ Data Ingestion   │            │
         │              │                  │            │
         │              │ • Text Parser    │            │
         └──────────────│ • Image Handler  │◀───────────┘
                        │ • Audio Processor│
                        │                  │
                        └──────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ AI Analysis &    │
                        │ Response Gen.    │
                        │                  │
                        │ • Medical Insights│
                        │ • Personalization│
                        │ • Recommendations│
                        └──────────────────┘
```

## 📁 Project Structure

```
healthcare-ai-assistant/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
├── README.md             # This file
└── src/
    ├── openai_integration.py    # OpenAI API client
    ├── ingestion.py            # Multimodal data processing
    └── preprocess.py           # Data preprocessing utilities
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/raiigauravv/heathcare-ai-assistant.git
cd heathcare-ai-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Supported File Formats

- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Audio**: MP3, WAV, M4A, FLAC, OGG

## 📝 Usage Examples

### 1. Text-Only Analysis
```
Patient: "John Doe"
Age: "25"
Gender: "Male"
Symptoms: "I have been experiencing severe headaches for the past 3 days, along with sensitivity to light and nausea."
```

### 2. Image + Text Analysis
```
Patient: "Jane Smith"
Age: "35"
Gender: "Female"
Symptoms: "Strange rash appeared on my arm yesterday"
Image: [Upload photo of rash]
```

### 3. Audio-Only Analysis
```
Patient: "Mike Johnson"
Age: "45"
Gender: "Male"
Audio: [Record voice describing symptoms]
```

## 🔒 Privacy & Security

- **No Data Storage**: Patient information is not stored permanently
- **Secure API Calls**: All communications encrypted via HTTPS
- **Environment Variables**: Sensitive keys stored securely
- **Open Source**: Full transparency of code and processes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🩺 Medical Disclaimer

**IMPORTANT**: This application is for educational and demonstration purposes only. It:

- ❌ Does NOT provide medical diagnosis
- ❌ Does NOT replace professional medical advice
- ❌ Should NOT be used for emergency medical situations
- ✅ Provides general health information only
- ✅ Encourages consultation with healthcare professionals

**In case of medical emergency, contact emergency services immediately.**

## 🙏 Acknowledgments

- **OpenAI** for providing advanced AI models
- **Hugging Face** for the deployment platform
- **Gradio** for the excellent UI framework
- **Python Community** for the amazing libraries

## 📞 Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/raiigauravv/heathcare-ai-assistant/issues)
- **Live Demo**: [Try it out](https://huggingface.co/spaces/gauravvraii/healthcare-ai-assistant)

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

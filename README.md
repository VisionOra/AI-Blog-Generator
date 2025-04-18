# AI Blog Generator

A multi-agent system for automated blog content creation leveraging CrewAI and large language models.

## Overview

The AI Blog Generator is a sophisticated system that employs multiple AI agents to collaboratively produce high-quality blog posts:

- **Content Planner**: Researches and plans the blog post structure
- **Content Writer**: Drafts the blog content based on the plan
- **Editor**: Refines and polishes the content for publication
- **Banner Designer** (optional): Creates prompts for banner image generation

The system can also generate banner images for blog posts using DALL-E 3.

## How to Start

### Prerequisites

- Python 3.10+
- OpenAI API key for GPT models and DALL-E image generation
- Google API key (optional, for Gemini models)
- SerperDev API key for search capabilities

### Installation

```bash
# Clone the repository
git clone https://github.com/VisionOra/AI-Blog-Generator.git
cd AI-Blog-Generator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY='your-openai-api-key'
export SERPER_API_KEY='your-serper-api-key'
export GOOGLE_API_KEY='your-google-api-key'  # Optional
```

### Usage

```bash
# Generate a blog post on a specific topic
python blog_writer.py --topic "Artificial Intelligence in Healthcare"

# Additional options
python blog_writer.py --topic "Future of Remote Work" --output "remote_work_blog.md" --custom-llm --image-size "1792x1024"
```

## Tools and Technologies

- **CrewAI**: Framework for orchestrating multiple AI agents
- **LangChain**: For integrating with various language models
- **OpenAI's GPT-3.5/4**: For content generation
- **Google's Gemini** (optional): Alternative LLM option
- **DALL-E 3**: For generating banner images
- **SerperDev**: For web search capabilities

## Command Line Arguments

- `--topic`: Topic for the blog post (required)
- `--output`: Output file for the blog post
- `--custom-llm`: Use Google's Gemini model instead of OpenAI
- `--include-designer`: Include the designer agent in the workflow
- `--no-image`: Skip banner image generation
- `--image-size`: Banner image size (choices: "1024x1024", "1792x1024", "1024x1792")

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/blog-writer-multi-agent.git
cd blog-writer-multi-agent
```

2. **Install dependencies**:

```bash
pip install openai requests python-dotenv crewai crewai-tools langchain-google-genai langchain-openai
```

3. **Set up environment variables** by creating a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_dev_api_key
GOOGLE_API_KEY=your_google_api_key (optional for Gemini)
```

## 🔑 API Keys

To use this system, you'll need to obtain the following API keys:

1. **OpenAI API Key** (Required for LLM and image generation)
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Navigate to API Keys section and create a new secret key
   - Add credit to your account for API usage

2. **Serper API Key** (Required for web search capabilities)
   - Sign up at [serper.dev](https://serper.dev)
   - Create an API key from your dashboard
   - Free tier available, paid tiers for more requests

3. **Google API Key** (Optional, for Gemini LLM)
   - Sign up at [AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Enable the Generative Language API

## 💡 How It Works

The system follows a collaborative workflow that mimics a real content creation team:

1. **Planning Phase**: The Content Planner agent researches the given topic using the Serper search tool, identifies key points, trends, and target audience information, and creates a detailed content outline.

2. **Writing Phase**: The Content Writer agent takes the outline from the Planner and crafts a complete blog post, focusing on structure, flow, and engaging content.

3. **Editing Phase**: The Editor agent reviews the draft from the Writer, improving style, checking facts, and ensuring the content aligns with best practices.

4. **Image Generation** (Optional): The system generates a customized banner image using DALL-E 3, based on the blog post content.

5. **Output**: The final blog post is saved as a Markdown file with the banner image embedded.

## 📋 Usage

Run the script with a topic for your blog post:

```bash
python blog_writer.py --topic "The Future of Artificial Intelligence"
```

### Optional Arguments

- `--output`: Specify a custom output filename
- `--custom-llm`: Use Google's Gemini model instead of OpenAI
- `--include-designer`: Add a designer agent to the workflow
- `--no-image`: Skip banner image generation
- `--image-size`: Set banner image size (default: 1792x1024)

Example with options:

```bash
python blog_writer.py --topic "Sustainable Living Tips" --output sustainability_blog.md --image-size 1024x1024
```

## 🌟 Use Cases

- **Content Marketing**: Generate blog posts for your company's website
- **Personal Blogging**: Create drafts for your personal blog on various topics
- **Educational Content**: Produce informative articles on complex subjects
- **SEO Content**: Generate search-optimized content with relevant keywords
- **Research Summaries**: Create comprehensive summaries of research topics

## 🔮 Architectural Concept

The Blog Writer Multi-Agent system demonstrates an emerging paradigm in AI application development: specialized AI agents collaborating to complete complex tasks. This approach offers several advantages:

- **Specialization**: Each agent excels at a specific part of the content creation process
- **Modularity**: Easily add or remove agents to customize the workflow
- **Quality Control**: Multiple review stages ensure higher content quality
- **Emergent Capabilities**: The combined system achieves results beyond individual agents

## 🔧 Technology Stack

- **Python**: Core programming language
- **CrewAI**: Framework for agent collaboration
- **LangChain**: Tool integration and LLM interface
- **OpenAI API**: Powers GPT models and DALL-E 3
- **Google Generative AI**: Optional Gemini model integration
- **Serper API**: Web search capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Elevating AI creativity—one image at a time! 🌟**

--- 


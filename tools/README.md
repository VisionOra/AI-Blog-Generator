# Blog Writer Multi-Agent System

A multi-agent system for automatically generating blog posts using AI agents with CrewAI.

## Overview

This system uses multiple AI agents to collaborate on creating blog posts:

1. **Content Planner**: Researches the topic and creates a detailed outline
2. **Content Writer**: Writes the blog post based on the planner's outline
3. **Editor**: Reviews and improves the final blog post
4. **Banner Image Designer**: Creates a banner image for the blog post using DALL-E 3

## How to Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for generation and images)
- Google API key (optional, for Gemini model)
- Serper Dev API key (for search tools)

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install crewai langchain-openai langchain-google-genai openai python-dotenv requests
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

### Usage

Run the blog writer with a topic:

```bash
python main.py --topic "Artificial Intelligence in Healthcare"
```

Additional options:
- `--output`: Specify the output file (default: topic name with underscores)
- `--custom-llm`: Use Google's Gemini model instead of GPT
- `--no-image`: Skip banner image generation

## Tools and Technologies

- **CrewAI**: Multi-agent framework for AI collaboration
- **LangChain**: Framework for working with language models
- **OpenAI**: GPT models for text generation and DALL-E 3 for image generation
- **Google Gemini** (optional): Alternative LLM for text generation

## Configuration

The system uses YAML files for configuration:
- `config/agents.yaml`: Defines the roles and goals of agents
- `config/tasks.yaml`: Defines the tasks for each agent

## Output

- Blog post in markdown format
- Banner image (optional)
- Planner's outline stored in `planner_output.md` 
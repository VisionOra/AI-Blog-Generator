# Blog Writer Multi-Agent System

A sophisticated multi-agent system for automatically generating SEO-optimized blog posts using AI agents with CrewAI.

## Overview

This system uses multiple specialized AI agents to collaborate on creating professional blog posts:

1. **Content Planner**: Researches the topic, identifies target audience, and creates a detailed outline with SEO keywords
2. **Content Writer**: Writes an engaging blog post based on the planner's outline and SEO strategy
3. **Editor**: Reviews and improves the final blog post for grammar, style, and brand alignment
4. **Banner Image Designer**: Creates a custom banner image prompt for DALL-E 3 featuring Artilence branding

## How to Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for content generation and images)
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
python tools/main.py --topic "Artificial Intelligence in Healthcare"
```

Additional options:
- `--output`: Specify the output file path (default: topic name with underscores)
- `--custom-llm`: Use Google's Gemini model instead of GPT
- `--no-image`: Skip banner image generation

## Tools and Technologies

- **CrewAI**: Framework for organizing AI agents into collaborative crews
- **LangChain**: Tools for connecting language models with external data sources
- **OpenAI GPT-3.5/4**: Primary language models for text generation
- **Google Gemini** (optional): Alternative LLM for text generation
- **DALL-E 3**: Image generation for blog post banners
- **SerperDev**: Search tool for agent research capabilities

## Configuration

The system uses YAML files for configuration:
- `config/agents.yaml`: Defines each agent's role, goal, and backstory
- `config/tasks.yaml`: Specifies detailed task descriptions and expected outputs

## Output

The system generates:
- Complete blog post in markdown format
- SEO-optimized content with proper structure
- Custom banner image with Artilence branding (#04C996, black, and white)
- Intermediate files:
  - `planner_output.md`: Content plan and outline
  - `writer_output.md`: Initial draft
  - `final_blog_post.md`: Edited final version with image

## Architecture

The `BlogWriter` class orchestrates the entire process:
1. Initializes agents with appropriate models and tools
2. Executes tasks in sequence (planning → writing → editing → designing)
3. Generates banner image using DALL-E 3 with custom prompt
4. Saves final blog post with embedded image reference 
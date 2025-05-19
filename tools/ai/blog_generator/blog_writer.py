from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import yaml

# Load environment variables
load_dotenv()

def generate_image(prompt, size="1024x1024", output_dir="blog_images"):
    """
    Generate an image using OpenAI's DALL-E 3 API
    
    Args:
        prompt (str): The prompt for image generation
        size (str): The size of the image (default: "1024x1024")
        output_dir (str): Directory to save the image
        
    Returns:
        str: Path to the generated image or None if generation failed
    """
    # Validate prompt and provide fallback if needed
    if not prompt or not prompt.strip():
        fallback_topic = getattr(BlogWriter, '_current_topic', "Professional blog post")
        prompt = f"Create a professional banner image for a blog about {fallback_topic}."

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
        
    try:
        # Initialize OpenAI client and generate image
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        # Download and save the image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        
        return filepath
    
    except Exception as e:
        return None

@CrewBase
class BlogWriter:
    """A crew for writing blog posts with a multi-agent approach"""        
    def __init__(self, use_custom_llm=False, topic=None, keywords=None):
        self.use_custom_llm = use_custom_llm
        self.topic = topic
        self.keywords = keywords
        self.blog_content = None
        self.search_tool = SerperDevTool()
        
        # Initialize LLM based on use_custom_llm flag
        if use_custom_llm:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key: raise ValueError("GOOGLE_API_KEY not found")
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, temperature=0.7)
        else:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Load YAML configurations from the configuration folder
        self.agents_config = {}
        self.tasks_config = {}
        
        # Get the absolute path to the configuration files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_file = os.path.join(current_dir, 'configuration', 'agents.yaml')
        tasks_file = os.path.join(current_dir, 'configuration', 'tasks.yaml')
        
        # Load agents config
        try:
            with open(agents_file, 'r') as f:
                self.agents_config = yaml.safe_load(f)
        except Exception as e:
            pass
            
        # Load tasks config
        try:
            with open(tasks_file, 'r') as f:
                self.tasks_config = yaml.safe_load(f)
        except Exception as e:
            pass
            
    @agent
    def planner(self):
        agent_config = self.agents_config.get("planner", {}).copy()
        
        # Format any placeholders in the configuration
        if 'goal' in agent_config and '{topic}' in agent_config['goal']:
            agent_config['goal'] = agent_config['goal'].format(topic=self.topic)
        if 'backstory' in agent_config and '{topic}' in agent_config['backstory']:
            agent_config['backstory'] = agent_config['backstory'].format(topic=self.topic)
            
        # Set default values if not in config
        role = agent_config.get('role', "Research Planner")
        goal = agent_config.get('goal', f"Create comprehensive outlines for high-quality blog posts about {self.topic}.")
        backstory = agent_config.get('backstory', "You are an expert content strategist who excels at researching topics and creating detailed content plans.")
            
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm,
            tools=[self.search_tool]
        )
    
    @agent
    def writer(self):
        agent_config = self.agents_config.get("writer", {}).copy()
        
        # Format any placeholders in the configuration
        if 'goal' in agent_config and '{topic}' in agent_config['goal']:
            agent_config['goal'] = agent_config['goal'].format(topic=self.topic)
        if 'backstory' in agent_config and '{topic}' in agent_config['backstory']:
            agent_config['backstory'] = agent_config['backstory'].format(topic=self.topic)
        
        # Set default values if not in config
        role = agent_config.get('role', "Content Writer")
        goal = agent_config.get('goal', f"Write engaging, informative blog posts about {self.topic}.")
        backstory = agent_config.get('backstory', "You are a professional writer with expertise in creating clear, engaging content.")
            
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def editor(self):
        agent_config = self.agents_config.get("editor", {}).copy()
        
        # Format any placeholders in the configuration
        if 'goal' in agent_config and '{topic}' in agent_config['goal']:
            agent_config['goal'] = agent_config['goal'].format(topic=self.topic)
        if 'backstory' in agent_config and '{topic}' in agent_config['backstory']:
            agent_config['backstory'] = agent_config['backstory'].format(topic=self.topic)
        
        # Set default values if not in config
        role = agent_config.get('role', "Content Editor")
        goal = agent_config.get('goal', "Polish and improve written content to make it more engaging, readable, and valuable.")
        backstory = agent_config.get('backstory', "You are a meticulous editor with years of experience improving content.")
            
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def designer(self):
        agent_config = self.agents_config.get("designer", {}).copy()
        
        # Format any placeholders in the configuration
        if 'goal' in agent_config and '{topic}' in agent_config['goal']:
            agent_config['goal'] = agent_config['goal'].format(topic=self.topic)
        if 'backstory' in agent_config and '{topic}' in agent_config['backstory']:
            agent_config['backstory'] = agent_config['backstory'].format(topic=self.topic)
        
        # Set default values if not in config
        role = agent_config.get('role', "Image Designer")
        goal = agent_config.get('goal', f"Create visually appealing image prompts for blog post banner images about {self.topic}.")
        backstory = agent_config.get('backstory', "You are an expert at creating descriptive prompts for AI image generation.")
            
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm
        )
    
    @task
    def planning_task(self):
        # Get the base task configuration
        task_config = self.tasks_config.get("planning_task", {}).copy()
        
        # Format the description with topic and other variables
        description = ""
        if 'description' in task_config and task_config['description']:
            description = task_config['description'].format(topic=self.topic)
        else:
            # Use a default description if none exists in config
            description = f"""Create a comprehensive outline and research plan for a blog post on the topic: {self.topic}.
Provide a detailed structure including main sections, sub-points, and key information to cover.
Identify target audience and suggest a suitable tone.
The final output should be a structured plan that the writer can easily follow."""
                
        # Add keywords to description if they exist for this instance
        if self.keywords:
            description += f"\n\nIncorporate the following keywords naturally into the research and outline: {self.keywords}"

        # Get expected output from config or use default
        expected_output = task_config.get('expected_output', "A detailed blog post outline and research plan, potentially guided by keywords.")
            
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.planner()
        )
    
    @task
    def writing_task(self):
        # Get the base task configuration
        task_config = self.tasks_config.get("writing_task", {}).copy()
        
        # Format the description with topic and other variables
        description = ""
        if 'description' in task_config and task_config['description']:
            description = task_config['description'].format(topic=self.topic)
        else:
            description = f"Write a comprehensive blog post based on the outline provided by the planner about {self.topic}."
            
        # Get expected output from config or use default
        expected_output = task_config.get('expected_output', "A well-written, engaging blog post in markdown format.")
            
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.writer()
        )
    
    @task
    def editing_task(self):
        # Get the base task configuration
        task_config = self.tasks_config.get("editing_task", {}).copy()
        
        # Format the description with topic and other variables if needed
        description = ""
        if 'description' in task_config and task_config['description']:
            description = task_config['description'].format(topic=self.topic)
        else:
            description = "Review and improve the blog post written by the writer. Fix any grammatical issues, improve readability, and ensure the content is engaging and valuable."
            
        # Get expected output from config or use default
        expected_output = task_config.get('expected_output', "A polished, professional blog post ready for publication.")
            
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.editor()
        )
    
    @task
    def designing_task(self):
        # Get the base task configuration
        task_config = self.tasks_config.get("designing_task", {}).copy()
        
        # Format the description with topic and other variables
        description = ""
        if 'description' in task_config and task_config['description']:
            description = task_config['description'].format(topic=self.topic)
        else:
            description = f"Create a detailed image prompt for a banner image about the topic: {self.topic}."
            
        # Get expected output from config or use default
        expected_output = task_config.get('expected_output', "A detailed image generation prompt for DALL-E 3.")
            
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.designer()
        )
    
    @crew
    def crew(self):
        agents_for_blog = [self.planner(), self.writer(), self.editor()]
        tasks_for_blog = [self.planning_task(), self.writing_task(), self.editing_task()]

        blog_crew = Crew(
            agents=agents_for_blog, 
            tasks=tasks_for_blog, 
            verbose=True 
            )
        
        return blog_crew
    
    def generate_blog(self, topic=None, keywords=None):
        # This method generates and returns the text content of the blog.
        if topic: self.topic = topic
        
        try:
            # Generate blog using the crew
            result = self.crew().kickoff(inputs={"topic": self.topic})
        except Exception as e:
            raise
            
        blog_content_raw = str(result)
        self.blog_content = blog_content_raw.strip()
        
        # Extract content directly from agents as fallback
        actual_blog_content = None
        try:
            for task_instance in self.crew().tasks:
                if hasattr(task_instance, 'agent') and hasattr(task_instance.agent, 'role') and task_instance.agent.role.lower() == "editor":
                    if hasattr(task_instance, 'output') and task_instance.output: 
                        editor_content = str(task_instance.output)
                        if "# " in editor_content and len(editor_content) > 100: 
                            actual_blog_content = editor_content
                            break 
            if not actual_blog_content:
                for task_instance in self.crew().tasks:
                    if hasattr(task_instance, 'agent') and hasattr(task_instance.agent, 'role') and task_instance.agent.role.lower() == "writer":
                        if hasattr(task_instance, 'output') and task_instance.output:
                            writer_content = str(task_instance.output)
                            if "# " in writer_content and len(writer_content) > 100: 
                                actual_blog_content = writer_content
                                break
        except Exception as e:
            pass
        
        if actual_blog_content:
            self.blog_content = actual_blog_content.strip()

        # Fallback and formatting logic
        if len(self.blog_content) < 100 or ('# ' not in self.blog_content and '## ' not in self.blog_content):
            self.blog_content = self._generate_fallback_blog_content(self.topic if self.topic else "Fallback Topic")
            
        if self.blog_content and not self.blog_content.startswith("# ") and not self.blog_content.startswith("## "):
            self.blog_content = f"# {self.topic if self.topic else 'Fallback Title'}\n\n{self.blog_content}"
            
        if self.blog_content and "##" not in self.blog_content and len(self.blog_content) > 800: 
            sections = ["Introduction", "Key Points", "Conclusion"]
            paragraphs = [p for p in self.blog_content.split("\n\n") if p.strip()]
            if len(paragraphs) >= 3:
                formatted_content = [paragraphs[0]] 
                formatted_content.append(f"\n## {sections[0]}\n"); formatted_content.append(paragraphs[1])
                if len(paragraphs) > 2:
                    formatted_content.append(f"\n## {sections[1]}\n")
                    for p_content in paragraphs[2:-1]: formatted_content.append(p_content)
                    formatted_content.append(f"\n## {sections[2]}\n"); formatted_content.append(paragraphs[-1])
                else: 
                    formatted_content.append(f"\n## {sections[2]}\n"); formatted_content.append(paragraphs[-1])
                self.blog_content = "\n\n".join(formatted_content)
        
        return self.blog_content
        
    def _generate_fallback_blog_content(self, topic):
        """
        Generate a fallback blog post structure if the main generation failed
        
        Args:
            topic (str): The topic for the fallback blog
            
        Returns:
            str: Formatted blog content in markdown
        """
        title = f"# {topic}"
        
        intro = f"""
## Introduction

Welcome to this comprehensive guide on {topic}. In this article, we'll explore the key aspects, 
latest developments, and practical applications of this fascinating subject."""

        main_content = f"""
## Key Points

{topic} encompasses a wide range of concepts and technologies that are continuously evolving. 
Let's examine some of the most important aspects that make it relevant today.

### Core Concepts

Understanding the fundamental principles is essential for mastering {topic}. 
These building blocks form the foundation of all advanced applications and developments in the field.

### Recent Developments

The landscape of {topic} is constantly changing with new research and technological advancements. 
Staying updated with these changes is crucial for anyone involved in this domain."""

        conclusion = f"""
## Conclusion

{topic} represents a significant area of opportunity and growth. By understanding its core principles 
and keeping pace with the latest developments, you can leverage its potential for innovation and 
problem-solving in various domains. As we continue to witness advancements in this field, its impact 
on our daily lives and professional endeavors will only grow stronger."""
        
        return title + intro + main_content + conclusion
        
    def generate_banner_image_with_prompt(self, prompt, size="1792x1024", image_output_dir="blog_images"):
        """
        Generate a banner image using a provided prompt
        
        Args:
            prompt (str): The prompt for image generation
            size (str): Image size (default: "1792x1024")
            image_output_dir (str): Directory to save the image
            
        Returns:
            str: Path to the generated image or None
        """
        # Ensure prompt is not empty
        if not prompt or not prompt.strip():
            prompt = f"Create a professional banner image for a blog about '{self.topic}'."
            
        return generate_image(prompt, size=size, output_dir=image_output_dir)
    
    def generate_banner_image_for_blog(self, topic=None, image_output_dir="blog_images"):
        """
        Generate a banner image for a blog post
        
        Args:
            topic (str): Topic for the banner image, defaults to instance topic
            image_output_dir (str): Directory to save the image
            
        Returns:
            str: Path to the generated image or None
        """
        effective_topic = topic if topic else self.topic
        if not effective_topic:
            raise ValueError("Topic must be provided for banner image generation")
            
        # Create a generic image prompt
        prompt = f"Create a professional, visually appealing banner image for a blog post about {effective_topic}. Use vibrant colors and modern design elements. Include Artilence branding with the main color #04C996, along with black and white accents."
        
        return generate_image(prompt, size="1792x1024", output_dir=image_output_dir)
    
    def save_blog_to_file(self, topic=None, output_file_name=None, base_output_dir=None):
        """Generate a blog post and save it to a file. Returns the file path."""
        # Use provided topic or instance topic
        effective_topic = topic if topic else self.topic
        if not effective_topic: 
            raise ValueError("Topic must be provided for save_blog_to_file.")
            
        # Store topic for potential future use
        BlogWriter._current_topic = effective_topic
        
        # Create a slug for the topic
        topic_slug = effective_topic.lower().replace(' ', '_')
        topic_slug = "".join(c for c in topic_slug if c.isalnum() or c in ('_', '-')).rstrip()
        if not topic_slug: 
            topic_slug = "untitled_blog"
            
        # Create directory for blog output
        current_blog_instance_dir = os.path.join(base_output_dir if base_output_dir else os.getcwd(), topic_slug)
        os.makedirs(current_blog_instance_dir, exist_ok=True)
        
        # Generate blog content
        blog_text_content = self.generate_blog(effective_topic) 

        # Determine output filename
        if output_file_name is None: 
            output_file_name = f"{topic_slug}_blog.md"
        
        # Save blog to file
        final_output_file_path = os.path.join(current_blog_instance_dir, output_file_name)
        with open(final_output_file_path, 'w', encoding='utf-8') as f:
            f.write(blog_text_content)
        
        return final_output_file_path 
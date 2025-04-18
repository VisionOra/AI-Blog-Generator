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

# Load environment variables
load_dotenv()

def generate_image(prompt, size="1024x1024", output_dir="blog_images"):
    """Generate an image using OpenAI's DALL-E 3 API"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        print(f"Generating image with prompt: {prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        
        print(f"Image generated successfully and saved to {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

@CrewBase
class BlogWriter:
    """A crew for writing blog posts with a multi-agent approach"""
    
    def __init__(self, use_custom_llm=False, generate_banner_image=True, topic=None):
        self.use_custom_llm = use_custom_llm
        self.generate_banner_image = generate_banner_image
        self.topic = topic
        self.search_tool = SerperDevTool()
        self.image_prompt = None
        self.blog_content = None
        
        if use_custom_llm:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro-preview-03-25",
                google_api_key=gemini_api_key,
                temperature=0.7,
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
    
    @agent
    def planner(self):
        return Agent(
            config=self.agents_config["planner"],
            verbose=True,
            llm=self.llm,
            tools=[self.search_tool]
        )
    
    @agent
    def writer(self):
        return Agent(
            config=self.agents_config["writer"],
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def editor(self):
        return Agent(
            config=self.agents_config["editor"],
            verbose=True,
            llm=self.llm
        )
    
    @agent
    def designer(self):
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            llm=self.llm
        )
    
    @task
    def planning_task(self):
        return Task(
            config=self.tasks_config["planning_task"],
            agent=self.planner()
        )
    
    @task
    def writing_task(self):
        return Task(
            config=self.tasks_config["writing_task"],
            agent=self.writer()
        )
    
    @task
    def editing_task(self):
        return Task(
            config=self.tasks_config["editing_task"],
            agent=self.editor()
        )
    
    @task
    def designing_task(self):
        return Task(
            config=self.tasks_config["designing_task"],
            agent=self.designer()
        )
    
    @crew
    def crew(self):
        agents = [self.planner(), self.writer(), self.editor()]
        tasks = [self.planning_task(), self.writing_task(), self.editing_task()]
        
        if self.generate_banner_image:
            agents.append(self.designer())
            tasks.append(self.designing_task())
        
        return Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
    
    def generate_blog(self, topic=None):
        """Generate a blog post on the specified topic"""
        if topic:
            self.topic = topic
        
        # Generate the blog content
        result = self.crew().kickoff(inputs={"topic": self.topic})
        self.blog_content = str(result)
        
        # Extract the image prompt from the designer's output if present
        if self.generate_banner_image:
            # Look for a section that contains the image prompt
            lines = self.blog_content.split('\n')
            prompt_section_start = None
            
            for i, line in enumerate(lines):
                if "IMAGE PROMPT:" in line or "BANNER IMAGE PROMPT:" in line:
                    prompt_section_start = i
                    break
            
            if prompt_section_start is not None:
                # Extract the prompt
                prompt_line = lines[prompt_section_start]
                self.image_prompt = prompt_line.split(":", 1)[1].strip()
                
                # Remove the prompt line from the blog content
                self.blog_content = '\n'.join(lines[:prompt_section_start] + lines[prompt_section_start+1:])
                
                # Generate the image with the extracted prompt
                image_path = self.generate_banner_image_with_prompt(self.image_prompt)
                return self.blog_content, image_path
            else:
                # If no specific prompt was found, generate a generic one
                image_path = self.generate_banner_image_for_blog(self.topic, self.blog_content)
                return self.blog_content, image_path
        
        return self.blog_content, None
    
    def generate_banner_image_with_prompt(self, prompt, size="1792x1024"):
        """Generate a banner image using the designer's prompt"""
        return generate_image(prompt, size=size, output_dir="blog_images")
    
    def generate_banner_image_for_blog(self, topic, blog_post):
        """Generate a banner image for the blog post using a generic prompt"""
        # Use the designer agent to create a prompt without using the search tool
        designer = Agent(
            role="Image Prompt Engineer",
            goal=f"Create a detailed, descriptive prompt for generating a banner image related to {topic}",
            backstory="You are a specialist in crafting detailed image prompts for AI image generators. Your job is to create vivid, specific descriptions for DALL-E 3.",
            verbose=True,
            llm=self.llm
        )
        
        designing_task = Task(
            description=f"""Create a detailed image prompt for a banner image about the topic: {topic}.
            The prompt should be detailed, descriptive, and visually rich to generate a high-quality image with DALL-E 3.
            Include specific details about composition, style, mood, lighting, colors, and subjects.
            Remember to include Artilence branding with the main color #04C996, along with black and white.
            Use a vibrant and warm color palette overall.
            Just provide the image prompt directly without any explanation.""",
            expected_output="A detailed image generation prompt for DALL-E 3",
            agent=designer
        )
        
        prompt_result = designing_task.execute()
        prompt = str(prompt_result).strip()
        
        # Ensure the prompt includes Artilence branding if not already present
        if "Artilence" not in prompt:
            prompt += " Include Artilence branding with the main color #04C996, along with black and white elements."
        
        # Log the generated prompt
        print(f"Designer agent generated prompt: {prompt}")
        
        # Use the prompt to generate the image
        return generate_image(prompt, size="1792x1024", output_dir="blog_images")
    
    def save_blog_to_file(self, topic=None, output_file=None):
        """Generate a blog post and save it to a file"""
        result, image_path = self.generate_blog(topic)
        
        if output_file is None:
            # Create a filename from the topic
            topic_to_use = topic if topic else self.topic
            output_file = f"{topic_to_use.lower().replace(' ', '_')}_blog.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # Append image reference to the blog post if an image was successfully generated
        if image_path:
            with open(output_file, 'r+', encoding='utf-8') as f:
                content = f.read()
                f.seek(0, 0)
                relative_path = os.path.relpath(image_path, os.path.dirname(output_file))
                f.write(f"![Banner Image for {self.topic}]({relative_path})\n\n{content}")
            
            print(f"Added image reference to blog post: {relative_path}")
        
        return output_file, image_path 
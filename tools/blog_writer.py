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
from typing import Optional # Restore Optional if needed for type hints

# Load environment variables
load_dotenv()

def generate_image(prompt, size="1024x1024", output_dir="blog_images"):
    """Generate an image using OpenAI's DALL-E 3 API"""
    # Validate prompt - ensure it's not empty
    if not prompt or not prompt.strip():
        print("Warning: Empty image prompt received. Using fallback prompt instead.")
        # Use topic from the BlogWriter instance if available, otherwise use a generic prompt
        fallback_topic = getattr(BlogWriter, '_current_topic', "Professional blog post")
        prompt = f"Create a professional, visually appealing banner image for a blog post about {fallback_topic}. Use vibrant colors, modern design elements, and high-quality composition."
    
    # Ensure the absolute path for output_dir is used
    if not os.path.isabs(output_dir):
        # Assuming the script is run from the project root or django_app directory
        # This might need adjustment depending on execution context
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
        # if called from django_app, base_path needs to be project root AI-Blog-Generator
        # if __file__ is tools/blog_writer.py, then os.path.dirname(__file__) is tools/
        # then base_path becomes AI-Blog-Generator/ (project root)
        # if called from django_app/api/views.py, then this __file__ context is different
        # For now, let's assume output_dir will be passed as an absolute path from the Django view
        pass # We will ensure output_dir is absolute when called

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Log the prompt being used
    print(f"Generating image with prompt: {prompt}")
    
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
    def __init__(self, use_custom_llm=False, topic=None, keywords=None):
        # This __init__ handles arguments passed during instantiation.
        # It initializes necessary attributes.
        self.use_custom_llm = use_custom_llm
        self.topic = topic
        self.keywords = keywords # Store keywords
        self.blog_content = None
        self.search_tool = SerperDevTool()
        
        # Initialize LLM based on use_custom_llm flag
        if use_custom_llm:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key: raise ValueError("GOOGLE_API_KEY not found")
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key, temperature=0.7)
        else:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # --- DO NOT CALL super().__init__() here --- 
        # To avoid potential recursion issues observed previously.
        
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
        # Base description
        task_description = f"""Create a comprehensive outline and research plan for a blog post on the topic: {self.topic}.
Provide a detailed structure including main sections, sub-points, and key information to cover.
Identify target audience and suggest a suitable tone.
The final output should be a structured plan that the writer can easily follow."""
        
        # Add keywords to description if they exist for this instance
        if self.keywords:
            task_description += f"\n\nIncorporate the following keywords naturally into the research and outline: {self.keywords}"

        # Load base config for the task if available, otherwise empty dict
        task_config = self.tasks_config.get("planning_task", {}).copy()
        # Override description and expected_output in the config for this specific run
        task_config['description'] = task_description
        task_config['expected_output'] = "A detailed blog post outline and research plan, potentially guided by keywords." # Updated expected output
        
        return Task(
            config=task_config, # Use the modified config
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
        # print("--- BlogWriter: crew() method called ---") # Diagnostic
        # This crew MUST be for text-only blog generation
        agents_for_blog = [self.planner(), self.writer(), self.editor()]
        tasks_for_blog = [self.planning_task(), self.writing_task(), self.editing_task()]
        
        # Remove conditional image generation logic
        # if self.generate_banner_image:
        #     # This block should not exist as image generation is removed from this crew
        #     pass # Or raise an error if this path is somehow reached

        # print(f"--- BlogWriter: Planner verbose: {agents_for_blog[0].verbose} ---")
        # print(f"--- BlogWriter: Writer verbose: {agents_for_blog[1].verbose} ---")
        # print(f"--- BlogWriter: Editor verbose: {agents_for_blog[2].verbose} ---")

        blog_crew = Crew(
            agents=agents_for_blog, 
            tasks=tasks_for_blog, 
            verbose=True 
            )
        
        # print(f"--- BlogWriter: Crew verbose: {blog_crew.verbose} ---") # Diagnostic
        return blog_crew
    
    def generate_blog(self, topic=None, keywords=None):
        # This method generates and returns ONLY the text content of the blog.
        print(f"--- BlogWriter: generate_blog() called for topic: {topic} ---") # Diagnostic
        if topic: self.topic = topic
        
        # Print LLM and Tool status before kickoff
        print(f"--- BlogWriter: LLM object: {self.llm} ---")
        print(f"--- BlogWriter: Search Tool object: {self.search_tool} ---")
        
        print(f"--- BlogWriter: About to kickoff crew for topic: {self.topic} --- Keywords: {self.keywords} ---") # Diagnostic
        try:
            # The crew is already configured to be text-only (planner, writer, editor)
            result = self.crew().kickoff(inputs={"topic": self.topic})
            print(f"--- BlogWriter: Crew kickoff finished. Result type: {type(result)} ---") # Diagnostic
        except Exception as e:
            print(f"--- BlogWriter: ERROR during crew kickoff: {e} ---") # Diagnostic
            raise # Re-raise the exception to be caught by the calling view
            
        blog_content_raw = str(result)
        
        # --- Simplified Content Processing --- 
        # Assume the result from kickoff is the primary content. 
        # Remove all logic related to image prompt detection and splitting.
        
        self.blog_content = blog_content_raw.strip()
        
        # Extract content directly from agents (this existing logic can stay as a fallback/refinement)
        actual_blog_content = None
        try:
            for task_instance in self.crew().tasks:
                if hasattr(task_instance, 'agent') and hasattr(task_instance.agent, 'role') and task_instance.agent.role.lower() == "editor":
                    if hasattr(task_instance, 'output') and task_instance.output: 
                        editor_content = str(task_instance.output)
                        if "# " in editor_content and len(editor_content) > 100: actual_blog_content = editor_content; break 
            if not actual_blog_content:
                for task_instance in self.crew().tasks:
                    if hasattr(task_instance, 'agent') and hasattr(task_instance.agent, 'role') and task_instance.agent.role.lower() == "writer":
                        if hasattr(task_instance, 'output') and task_instance.output:
                            writer_content = str(task_instance.output)
                            if "# " in writer_content and len(writer_content) > 100: actual_blog_content = writer_content; break
        except Exception as e: print(f"Error extracting content directly from agents: {e}")
        
        if actual_blog_content:
            print("Using blog content directly from editor/writer agent output.")
            self.blog_content = actual_blog_content.strip()
        else:
            print("Using the full kickoff result as blog content.")
            # self.blog_content is already set from blog_content_raw above

        # --- End Simplified Content Processing --- 

        # Fallback and formatting logic (remains unchanged)
        if len(self.blog_content) < 100 or ('# ' not in self.blog_content and '## ' not in self.blog_content) :
            print("Warning: Extracted blog content appears invalid or too short. Generating fallback content.")
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
        
        # Ensure only the final string content is returned
        return self.blog_content 
        
    def _generate_fallback_blog_content(self, topic):
        """Generate a fallback blog post structure if the main generation failed"""
        title = f"# {topic}"
        intro = f"\n\n## Introduction\n\nWelcome to this comprehensive guide on {topic}. In this article, we'll explore the key aspects, latest developments, and practical applications of this fascinating subject."
        main_content = f"\n\n## Key Points\n\n{topic} encompasses a wide range of concepts and technologies that are continuously evolving. Let's examine some of the most important aspects that make it relevant today.\n\n### Core Concepts\n\nUnderstanding the fundamental principles is essential for mastering {topic}. These building blocks form the foundation of all advanced applications and developments in the field.\n\n### Recent Developments\n\nThe landscape of {topic} is constantly changing with new research and technological advancements. Staying updated with these changes is crucial for anyone involved in this domain."
        conclusion = f"\n\n## Conclusion\n\n{topic} represents a significant area of opportunity and growth. By understanding its core principles and keeping pace with the latest developments, you can leverage its potential for innovation and problem-solving in various domains. As we continue to witness advancements in this field, its impact on our daily lives and professional endeavors will only grow stronger."
        
        return title + intro + main_content + conclusion
        
    def generate_banner_image_with_prompt(self, prompt, size="1792x1024", image_output_dir="blog_images"):
        """Generate a banner image using the designer's prompt"""
        # Ensure prompt is not empty
        if not prompt or not prompt.strip():
            print("Warning: Empty prompt in generate_banner_image_with_prompt. Using fallback.")
            prompt = f"Create a visually appealing banner image for a blog about '{self.topic}'. The image should be professional, modern, and relevant to the topic."
            
        return generate_image(prompt, size=size, output_dir=image_output_dir)
    
    def generate_banner_image_for_blog(self, topic, blog_post, image_output_dir="blog_images"):
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
        
        # Ensure the prompt is not empty
        if not prompt or len(prompt) < 10:  # Assuming a proper prompt should be at least 10 chars
            print(f"Warning: Designer agent returned insufficient prompt: '{prompt}'. Using fallback.")
            prompt = f"Create a professional, visually appealing banner image for a blog post about {topic}. Use vibrant colors and modern design elements. Include Artilence branding with the main color #04C996, along with black and white accents."
        
        # Ensure the prompt includes Artilence branding if not already present
        if "Artilence" not in prompt:
            prompt += " Include Artilence branding with the main color #04C996, along with black and white elements."
        
        # Log the generated prompt
        print(f"Designer agent generated prompt: {prompt}")
        
        # Use the prompt to generate the image
        return generate_image(prompt, size="1792x1024", output_dir=image_output_dir)
    
    def save_blog_to_file(self, topic=None, output_file_name=None, base_output_dir=None):
        """Generate a blog post (text only) and save it to a file.
           This method MUST ONLY return the file path of the saved markdown blog post (a string).
        """
        effective_topic = topic if topic else self.topic
        if not effective_topic: raise ValueError("Topic must be provided for save_blog_to_file.")
        BlogWriter._current_topic = effective_topic
        topic_slug = effective_topic.lower().replace(' ', '_')
        topic_slug = "".join(c for c in topic_slug if c.isalnum() or c in ('_', '-')).rstrip()
        if not topic_slug: topic_slug = "untitled_blog"
        current_blog_instance_dir = os.path.join(base_output_dir if base_output_dir else os.getcwd(), topic_slug)
        os.makedirs(current_blog_instance_dir, exist_ok=True)
        
        # Generate TEXT blog content ONLY
        blog_text_content = self.generate_blog(effective_topic) 

        if output_file_name is None: 
            output_file_name = f"{topic_slug}_blog.md"
        
        final_output_file_path = os.path.join(current_blog_instance_dir, output_file_name)

        with open(final_output_file_path, 'w', encoding='utf-8') as f:
            f.write(blog_text_content)
        
        return final_output_file_path 
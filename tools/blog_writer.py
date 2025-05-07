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
        blog_content_raw = str(result)
        
        # Check if the result looks like an image prompt rather than a blog post
        # Common indicators of image prompts rather than actual blog content
        image_prompt_indicators = [
            "image prompt:",
            "banner image",
            "for the banner image",
            "image should depict",
            "color palette should",
            "composition should",
            "style of the image",
            "lighting should be",
            "the background should",
            "artilence branding",
            "#04c996",
        ]
        
        # Extract the actual blog content from the planner, writer, and editor agents
        actual_blog_content = None
        try:
            # First try to get final blog directly from editor output
            for task in self.crew().tasks:
                if hasattr(task, 'agent') and hasattr(task.agent, 'role') and task.agent.role.lower() == "editor":
                    if hasattr(task, 'output') and task.output:
                        editor_content = str(task.output)
                        # Check if editor output looks like a proper blog post (has a title and paragraphs)
                        if "# " in editor_content and len(editor_content) > 500:
                            actual_blog_content = editor_content
                            break
            
            # If we couldn't get editor content, try the writer
            if not actual_blog_content:
                for task in self.crew().tasks:
                    if hasattr(task, 'agent') and hasattr(task.agent, 'role') and task.agent.role.lower() == "writer":
                        if hasattr(task, 'output') and task.output:
                            writer_content = str(task.output)
                            # Check if writer output looks like a proper blog post (has a title and paragraphs)
                            if "# " in writer_content and len(writer_content) > 500:
                                actual_blog_content = writer_content
                                break
        except Exception as e:
            print(f"Error extracting content directly from agents: {e}")
        
        # If we got actual blog content from an agent, use it instead of the raw result
        if actual_blog_content:
            print("Using blog content directly from editor/writer agent")
            blog_content_raw = actual_blog_content
        
        # Process the blog content to extract only the actual blog
        lines = blog_content_raw.split('\n')
        blog_content_lines = []
        image_prompt_lines = []
        in_image_prompt = False
        skip_line = False
        
        # First, let's check if the raw content is primarily an image prompt
        # by counting how many image prompt indicators it contains
        indicators_found = sum(1 for indicator in image_prompt_indicators if indicator in blog_content_raw.lower())
        content_is_mostly_prompt = indicators_found >= 3 and '# ' not in blog_content_raw[:500]
        
        if content_is_mostly_prompt:
            # Likely not a proper blog at all - we'll need to generate content
            print("Warning: Output appears to be primarily image prompt text rather than blog content.")
            # Save the image prompt text
            self.image_prompt = blog_content_raw
            # Generate a proper blog post structure
            self.blog_content = self._generate_fallback_blog_content(topic)
            return self.blog_content, None  # No image path since we already have the prompt
        
        # Process line by line to extract blog content and image prompt
        for i, line in enumerate(lines):
            # Check for image prompt markers
            if "image prompt:" in line.lower() or "banner image:" in line.lower():
                in_image_prompt = True
                image_prompt_lines.append(line.replace("Image Prompt:", "").replace("Banner Image:", "").strip())
                skip_line = True
                continue
                
            # Check if this line is the start of an image description
            if i > 0 and not in_image_prompt:
                for indicator in image_prompt_indicators:
                    if indicator in line.lower():
                        # If the line contains multiple indicators, it's likely describing an image
                        indicators_in_line = sum(1 for ind in image_prompt_indicators if ind in line.lower())
                        if indicators_in_line >= 2:
                            in_image_prompt = True
                            image_prompt_lines.append(line)
                            skip_line = True
                            break
                        # Special case for "for the banner image" which is a strong indicator
                        elif "for the banner image" in line.lower():
                            in_image_prompt = True
                            image_prompt_lines.append(line)
                            skip_line = True
                            break
            
            # Check if we're transitioning from image prompt back to blog content
            if in_image_prompt and (line.startswith("# ") or line.startswith("## ")):
                in_image_prompt = False
                skip_line = False
                
            # Process the line
            if in_image_prompt:
                if not skip_line:  # Don't add the prompt marker line twice
                    image_prompt_lines.append(line)
            else:
                blog_content_lines.append(line)
            
            # Reset skip_line
            skip_line = False
        
        # Process result to extract image prompt if it's mixed in with the blog content
        self.blog_content = "\n".join(blog_content_lines).strip()
        self.image_prompt = "\n".join(image_prompt_lines).strip()
        
        # Check if we ended up with valid blog content
        if len(self.blog_content) < 200 or '# ' not in self.blog_content:
            print("Warning: Extracted blog content appears invalid or too short. Generating fallback content.")
            self.blog_content = self._generate_fallback_blog_content(topic)
        
        # Ensure the blog has a proper title
        if self.blog_content and not self.blog_content.startswith("# "):
            self.blog_content = f"# {self.topic}\n\n{self.blog_content}"
            
        # Ensure the content is properly formatted with sections if very long
        if self.blog_content and "##" not in self.blog_content and len(self.blog_content) > 1000:
            # Add some basic headings if none exist
            sections = ["Introduction", "Key Points", "Conclusion"]
            paragraphs = [p for p in self.blog_content.split("\n\n") if p.strip()]
            
            if len(paragraphs) >= 4:  # Title + at least 3 paragraphs
                # Only add headings if we have enough paragraphs to work with
                formatted_content = [paragraphs[0]]  # Title/first paragraph
                
                # Add Introduction section
                formatted_content.append(f"\n## {sections[0]}\n")
                formatted_content.append(paragraphs[1])
                
                # Add middle sections - Key Points
                formatted_content.append(f"\n## {sections[1]}\n")
                # Add middle paragraphs (all except first and last)
                for p in paragraphs[2:-1]:
                    formatted_content.append(p)
                
                # Add conclusion
                formatted_content.append(f"\n## {sections[2]}\n")
                formatted_content.append(paragraphs[-1])
                
                self.blog_content = "\n\n".join(formatted_content)
        
        # If no valid image prompt was found, generate a generic one
        if not self.image_prompt or len(self.image_prompt) < 50:
            self.image_prompt = f"Create a professional, visually appealing banner image for a blog post about {topic}. The image should use vibrant colors including Artilence's main color #04C996, along with black and white accents. It should be modern, clean, and conceptually represent the topic in an engaging way."
        
        # Determine image output directory, defaulting if not set by save_blog_to_file
        image_dir_to_use = getattr(self, '_current_image_dir', "blog_images")

        # Generate an image if requested
        if self.generate_banner_image:
            # Generate the image with the extracted or generated prompt
            image_path = self.generate_banner_image_with_prompt(self.image_prompt, image_output_dir=image_dir_to_use)
            return self.blog_content, image_path
        
        return self.blog_content, None
        
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
        """Generate a blog post and save it to a file within a specific base directory."""
        
        effective_topic = topic if topic else self.topic
        if not effective_topic:
            raise ValueError("Topic must be provided either at initialization or when calling save_blog_to_file.")

        # Store the current topic for potential use in fallback image prompts
        BlogWriter._current_topic = effective_topic
            
        # Sanitize topic to create a valid directory name
        # Replace spaces with underscores, remove characters not suitable for filenames/paths
        topic_slug = effective_topic.lower().replace(' ', '_')
        topic_slug = "".join(c for c in topic_slug if c.isalnum() or c in ('_', '-')).rstrip()
        if not topic_slug: # handle cases where topic might become empty after sanitization
            topic_slug = "untitled_blog"

        if base_output_dir:
            # All outputs for this specific blog post will go into a subfolder named after the topic slug
            current_blog_instance_dir = os.path.join(base_output_dir, topic_slug)
        else:
            # Default behavior: save in a subfolder (named by topic_slug) in the current working directory
            current_blog_instance_dir = os.path.join(os.getcwd(), topic_slug)
        
        os.makedirs(current_blog_instance_dir, exist_ok=True)
        
        # Define image output directory within the current blog's instance directory
        image_specific_output_dir = os.path.join(current_blog_instance_dir, "blog_images")
        os.makedirs(image_specific_output_dir, exist_ok=True)

        # Set the image directory for generate_blog to use
        self._current_image_dir = image_specific_output_dir
        
        try:
            result, image_path = self.generate_blog(effective_topic)
        finally:
            # Clean up the temporary attribute
            if hasattr(self, '_current_image_dir'):
                del self._current_image_dir

        if output_file_name is None:
            # Default filename if not provided
            output_file_name = f"{topic_slug}_blog.md"
        
        # final_output_file_path is the full path to the markdown file
        final_output_file_path = os.path.join(current_blog_instance_dir, output_file_name)

        with open(final_output_file_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # Append image reference to the blog post if an image was successfully generated
        if image_path: # image_path from generate_image is absolute
            markdown_dir = os.path.dirname(final_output_file_path)
            # Make image_path relative to the markdown file for the link in the markdown
            relative_image_path_for_markdown = os.path.relpath(image_path, start=markdown_dir)
            
            with open(final_output_file_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                f.seek(0, 0)
                # Use a platform-agnostic path for the markdown link
                md_image_link = relative_image_path_for_markdown.replace(os.sep, '/')
                f.write(f"![Banner Image for {effective_topic}]({md_image_link})\n\n{content}")
            
            print(f"Added image reference to blog post: {md_image_link}")
        
        return final_output_file_path, image_path # image_path is still absolute here 
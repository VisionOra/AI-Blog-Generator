"""
Blog Writer Multi-Agent System

This script implements a multi-agent system for blog writing using CrewAI.
It includes three agents: planner, writer, and editor who collaborate to produce a blog post.
It also generates a banner image for each blog post using DALL-E 3.
"""

# Apply typing patch for Python 3.10 compatibility

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Load environment variables
load_dotenv()

def generate_image(prompt, size="1024x1024", output_dir="blog_images"):
    """
    Generate an image using OpenAI's DALL-E 3 API
    
    Args:
        prompt (str): The description of the image to generate
        size (str): Size of the image. Options: "1024x1024", "1792x1024", or "1024x1792"
        output_dir (str): Directory to save the generated image
    
    Returns:
        str: Path to the saved image
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        # Generate the image
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        # Get the image URL
        image_url = response.data[0].url
        
        # Download the image
        image_response = requests.get(image_url)
        
        # Generate a filename based on current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        
        print(f"Image generated successfully and saved to {filepath}")
        print(f"Prompt: {prompt}")
        return filepath
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

class BlogCrewAgents:
    """Class to manage the creation and configuration of agents for the blog writing crew."""

    def __init__(self, use_custom_llm=False):
        """
        Initialize the blog crew agents with optional custom Gemini LLM.
        
        Args:
            use_custom_llm (bool): Whether to use a custom Gemini LLM.
        """
        # Set up tools
        self.search_tool = SerperDevTool()

        # Set up LLM
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
            # Use default ChatOpenAI as fallback
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )

    def create_planner_agent(self):
        """Create and return the content planner agent."""
        return Agent(
            role="Content Planner",
            goal="Plan engaging and factually accurate content on {topic}",
            backstory=("You're working on planning a blog article "
                      "about the topic: {topic} in 'https://medium.com/'."
                      "You collect information that helps the "
                      "audience learn something "
                      "and make informed decisions. "
                      "You have to prepare a detailed "
                      "outline and the relevant topics and sub-topics that has to be a part of the"
                      "blogpost."
                      "Your work is the basis for "
                      "the Content Writer to write an article on this topic."),
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            tools=[self.search_tool]
        )

    def create_writer_agent(self):
        """Create and return the content writer agent."""
        return Agent(
            role="Content Writer",
            goal=("Write insightful and factually accurate "
                 "opinion piece about the topic: {topic}"),
            backstory=("You're working on a writing "
                      "a new opinion piece about the topic: {topic} in 'https://medium.com/'. "
                      "You base your writing on the work of "
                      "the Content Planner, who provides an outline "
                      "and relevant context about the topic. "
                      "You follow the main objectives and "
                      "direction of the outline, "
                      "as provide by the Content Planner. "
                      "You also provide objective and impartial insights "
                      "and back them up with information "
                      "provide by the Content Planner. "
                      "You acknowledge in your opinion piece "
                      "when your statements are opinions "
                      "as opposed to objective statements."),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def create_editor_agent(self):
        """Create and return the editor agent."""
        return Agent(
            role="Editor",
            goal=("Edit a given blog post to align with "
                 "the writing style of the organization 'https://medium.com/'. "),
            backstory=("You are an editor who receives a blog post "
                      "from the Content Writer. "
                      "Your goal is to review the blog post "
                      "to ensure that it follows journalistic best practices,"
                      "provides balanced viewpoints "
                      "when providing opinions or assertions, "
                      "and also avoids major controversial topics "
                      "or opinions when possible."),
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def create_designer_agent(self):
        """Create and return the designer agent."""
        return Agent(
            role="Senior Banner Image Designer",
            goal=("Create a banner image for the blog post "
                "topic {topic} that is engaging and visually appealing."),
            backstory=("You are a Senior Banner Image Designer "
                    "who receives a request to create a banner image "
                    "for the blog post topic: {topic}. Your goal is to "
                    "create a visually appealing and engaging image "
                    "that captures the essence of the blog post "
                    "and attracts readers' attention. "
                    "You have to ensure that the image is relevant."),
            allow_delegation=False,
            verbose=True,
            tools=[self.search_tool],
            llm=self.llm
        )


class BlogCrewTasks:
    """Class to manage task creation for the blog writing crew."""
    
    @staticmethod
    def create_planning_task(planner_agent, search_tool):
        """Create the content planning task."""
        return Task(
            description=(
                "1. Prioritize the latest trends, key players, "
                    "and noteworthy news on {topic}.\n"
                "2. Identify the target audience, considering "
                    "their interests and pain points.\n"
                "3. Develop a detailed content outline including "
                    "an introduction, key points, and a call to action.\n"
                "4. Include SEO keywords and relevant data or sources."
            ),
            expected_output="A comprehensive content plan document "
                "with an outline, audience analysis, "
                "SEO keywords, and resources.",
            agent=planner_agent,
            tools=[search_tool]
        )
    
    @staticmethod
    def create_writing_task(writer_agent):
        """Create the content writing task."""
        return Task(
            description=(
                "1. Use the content plan to craft a compelling "
                    "blog post on {topic}.\n"
                "2. Incorporate SEO keywords naturally.\n"
                "3. Sections/Subtitles are properly named "
                    "in an engaging manner.\n"
                "4. Ensure the post is structured with an "
                    "engaging introduction, insightful body, "
                    "and a summarizing conclusion.\n"
                "5. Proofread for grammatical errors and "
                    "alignment with the brand's voice.\n"
            ),
            expected_output="A well-written blog post "
                "in markdown format, ready for publication, "
                "each section should have 2 or 3 paragraphs.",
            agent=writer_agent,
        )
    
    @staticmethod
    def create_editing_task(editor_agent):
        """Create the content editing task."""
        return Task(
            description=("Proofread the given blog post for "
                        "grammatical errors and "
                        "alignment with the brand's voice."
                        ),
            expected_output="A well-written blog post in markdown format, "
                           "without the word markdown in the beginning "
                           "ready for publication, "
                           "each section should have 2 or 3 paragraphs.",
            agent=editor_agent,
        )
    
    @staticmethod
    def create_designing_task(designer_agent, search_tool):
        """Create the banner image design task."""
        return Task(
            description=("Create a banner image for the blog post "
                       "make sure it is relevant and engaging."),
            expected_output="A visually appealing banner image "
                          "related to the blog post topic.",
            agent=designer_agent,
            tools=[search_tool]
        )


class BlogCrew:
    """Main class to manage the blog creation crew and workflow."""
    
    def __init__(self, use_custom_llm=False, include_designer=False, generate_banner_image=True):
        """
        Initialize the blog crew.
        
        Args:
            use_custom_llm (bool): Whether to use a custom Gemini LLM.
            include_designer (bool): Whether to include the designer agent.
            generate_banner_image (bool): Whether to generate a banner image using DALL-E 3.
        """
        self.agents_manager = BlogCrewAgents(use_custom_llm)
        # Create agents
        self.planner = self.agents_manager.create_planner_agent()
        self.writer = self.agents_manager.create_writer_agent()
        self.editor = self.agents_manager.create_editor_agent()
        
        # Optional designer agent
        self.designer = None
        if include_designer:
            self.designer = self.agents_manager.create_designer_agent()
        
        # Create tasks
        self.planning_task = BlogCrewTasks.create_planning_task(
            self.planner, self.agents_manager.search_tool)
        self.writing_task = BlogCrewTasks.create_writing_task(self.writer)
        self.editing_task = BlogCrewTasks.create_editing_task(self.editor)
        
        # Optional designing task
        self.designing_task = None
        if include_designer:
            self.designing_task = BlogCrewTasks.create_designing_task(
                self.designer, self.agents_manager.search_tool)
        
        # Set up the agents and tasks lists
        self.agents = [self.planner, self.writer, self.editor]
        self.tasks = [self.planning_task, self.writing_task, self.editing_task]
        
        if include_designer:
            self.agents.append(self.designer)
            self.tasks.append(self.designing_task)
        
        # Create the crew
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
        
        # Banner image generation flag
        self.generate_banner_image = generate_banner_image
    
    def generate_blog(self, topic):
        """
        Generate a blog post on the specified topic.
        
        Args:
            topic (str): The topic for the blog post.
            
        Returns:
            str: The generated blog post content.
        """
        result = self.crew.kickoff(inputs={"topic": topic})
        return result
    
    def generate_banner_image_for_blog(self, topic, blog_post):
        """
        Generate a banner image for the blog post.
        
        Args:
            topic (str): The topic of the blog post.
            
        Returns:
            str: Path to the generated image file.
        """
        prompt = f"Create a professional, visually striking banner image for a blog post about '{topic}'. The image should be modern, eye-catching, and suitable for a Medium-style article. Please create a banner image that is relevant to the blog post. Here is the blog post: {blog_post}"
        
        return generate_image(prompt, size="1792x1024", output_dir="blog_images")
    
    def save_blog_to_file(self, topic, output_file=None):
        """
        Generate a blog post and save it to a file.
        
        Args:
            topic (str): The topic for the blog post.
            output_file (str): Optional filename for the output file. If not provided,
                              a default name will be generated from the topic.
                              
        Returns:
            tuple: (str, str) Path to the saved blog file and path to the banner image (if generated).
        """
        result = self.generate_blog(topic)
        
        if output_file is None:
            # Create a filename from the topic
            output_file = f"{topic.lower().replace(' ', '_')}_blog.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(result))
        
        image_path = None
        if self.generate_banner_image:
            image_path = self.generate_banner_image_for_blog(topic, result)
            
            # Append image reference to the blog post if an image was successfully generated
            if image_path:
                with open(output_file, 'r+', encoding='utf-8') as f:
                    content = f.read()
                    f.seek(0, 0)
                    relative_path = os.path.relpath(image_path, os.path.dirname(output_file))
                    f.write(f"![Banner Image for {topic}]({relative_path})\n\n{content}")
        
        return output_file, image_path


def main():
    """Main function to run the blog writer system."""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent Blog Writer')
    parser.add_argument('--topic', type=str, required=True, help='Topic for the blog post')
    parser.add_argument('--output', type=str, help='Output file for the blog post')
    parser.add_argument('--custom-llm', action='store_true', help='Use custom Gemini LLM')
    parser.add_argument('--include-designer', action='store_true', help='Include designer agent')
    parser.add_argument('--no-image', action='store_true', help='Skip banner image generation')
    parser.add_argument('--image-size', type=str, default="1792x1024", 
                        choices=["1024x1024", "1792x1024", "1024x1792"],
                        help='Banner image size (default: 1792x1024)')
    
    args = parser.parse_args()
    
    # Check if OPENAI_API_KEY is set for image generation
    if not args.no_image and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Banner image generation will be disabled.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        args.no_image = True
    
    # Create the blog crew and generate the blog
    blog_crew = BlogCrew(
        use_custom_llm=args.custom_llm,
        include_designer=args.include_designer,
        generate_banner_image=not args.no_image
    )
    
    output_file, image_path = blog_crew.save_blog_to_file(args.topic, args.output)
    print(f"Blog post generated and saved to {output_file}")
    
    if image_path:
        print(f"Banner image generated and saved to {image_path}")


if __name__ == "__main__":
    main() 

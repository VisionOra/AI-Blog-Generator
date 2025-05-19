import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Load environment variables
load_dotenv()

# Define the base directory for this service module
SERVICE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTS_OUTPUT_DIR = os.path.join(SERVICE_BASE_DIR, 'posts')

@CrewBase
class LinkedInPostGenerator:
    """A crew for generating LinkedIn posts."""

    def __init__(self, use_custom_llm=False, topic=None):
        self.use_custom_llm = use_custom_llm
        self.topic = topic

        if use_custom_llm:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables for custom LLM.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=gemini_api_key,
                temperature=0.7,
            )
        else:
            # Default to OpenAI GPT 3.5 Turbo
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables for default LLM.")
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )

    @agent
    def linkedin_post_writer_agent(self):
        return Agent(
            role="LinkedIn Content Strategist",
            goal="""Craft a professional, elegant, and impactful LinkedIn post for the given topic.
            The post should be structured as two distinct, concise paragraphs, each offering some insight or value.
            It must incorporate SEO-friendly keywords naturally, judiciously use professional emojis to enhance readability and engagement,
            and conclude with relevant hashtags.""",
            backstory="""You are an expert LinkedIn content creator, skilled in writing concise and impactful posts
            that drive engagement. You understand the nuances of professional social media communication, SEO,
            and the subtle use of emojis to enhance a message, focusing on delivering value in a polished format.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    @task
    def linkedin_generation_task(self):
        topic_placeholder = self.topic if self.topic else "{topic}"
        
        return Task(
            description=f"""Generate a LinkedIn post for the topic: {topic_placeholder}.
            The post must adhere to the following criteria:
            - Tone: Professional and elegant.
            - Structure: Two distinct paragraphs. Each paragraph should be concise (e.g., 2-4 sentences long), well-developed, and focus on a key aspect or provide valuable insight related to the topic.
            - Content: Clean, incorporating relevant SEO-friendly keywords naturally.
            - Emojis: Thoughtfully integrate 1-2 professional and relevant emojis per paragraph where they enhance the message or readability. Avoid overuse and ensure they maintain an elegant tone.  مثلاً (For example: ✨, 🚀, 💡, 📈, 🤝, ✅)
            - Conclusion: End with 3-5 relevant hashtags.
            The output should be only the LinkedIn post content itself, formatted with a clear separation between the two paragraphs.""",
            expected_output="""A professional LinkedIn post composed of two well-defined, concise paragraphs, subtly enhanced with 1-2 professional emojis per paragraph, and followed by 3-5 relevant hashtags.
Example:
[Paragraph 1: 2-4 sentences developing a key point or introducing the topic with an insight. ✨ Maybe an emoji here.]

[Paragraph 2: 2-4 sentences expanding on another aspect, offering a takeaway, or a call to thought. 🚀 Perhaps another one here.]

#hashtag1 #hashtag2 #hashtag3 #hashtag4""",
            agent=self.linkedin_post_writer_agent()
        )

    def generate_post(self, topic: str):
        """Generates a LinkedIn post for a given topic and saves it to a file."""
        if not topic:
            raise ValueError("Topic must be provided for LinkedIn post generation.")

        inputs = {"topic": topic}
        self.topic = topic  # Update the instance topic

        linkedin_crew = Crew(
            agents=[self.linkedin_post_writer_agent()],
            tasks=[self.linkedin_generation_task()],
            process=Process.sequential,
            verbose=True
        )
        
        post_content_raw = str(linkedin_crew.kickoff(inputs=inputs)).strip()

        # Process content to lowercase all hashtags
        if post_content_raw:
            # Process each line individually to preserve line breaks
            processed_lines = []
            for line in post_content_raw.split('\n'):
                processed_words = []
                
                for word in line.split():
                    if word.startswith('#'):
                        # Convert standalone hashtag to lowercase
                        processed_words.append(word.lower())
                    elif '#' in word:
                        # Process word with embedded hashtag(s)
                        new_word = ""
                        i = 0
                        while i < len(word):
                            if word[i] == '#':
                                # Start of hashtag
                                hashtag_start = i
                                i += 1
                                # Find end of hashtag
                                while i < len(word) and (word[i].isalnum() or word[i] == '_'):
                                    i += 1
                                # Extract and lowercase the hashtag
                                hashtag = word[hashtag_start:i].lower()
                                new_word += hashtag
                            else:
                                new_word += word[i]
                                i += 1
                        processed_words.append(new_word)
                    else:
                        # Regular word, no change needed
                        processed_words.append(word)
                
                processed_lines.append(' '.join(processed_words))
            
            post_content = '\n'.join(processed_lines)
        else:
            post_content = ""

        if post_content:
            # Save the post to a file
            try:
                os.makedirs(POSTS_OUTPUT_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize topic for filename
                topic_slug = topic.lower().replace(' ', '_')
                topic_slug = "".join(c for c in topic_slug if c.isalnum() or c in ('_', '-')).rstrip()
                if not topic_slug:
                    topic_slug = "untitled_post"
                
                filename = f"linkedin_{topic_slug}_{timestamp}.md"
                filepath = os.path.join(POSTS_OUTPUT_DIR, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(post_content)
                return post_content, filepath
            except Exception:
                return post_content, None 
        
        return "", None

# Example Usage (for direct testing of this file)
if __name__ == '__main__':
    try:
        generator = LinkedInPostGenerator(use_custom_llm=False)
        sample_topic = "The Future of Renewable Energy"
        post, file_path = generator.generate_post(sample_topic)
        print("\n--- Generated LinkedIn Post ---")
        print(post)
        if file_path:
            print(f"\nSaved to: {file_path}")
        print("\n-----------------------------")
    except Exception as e:
        print(f"Error: {e}")

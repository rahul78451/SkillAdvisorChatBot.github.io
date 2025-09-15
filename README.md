CareerSage
CareerSage is an AI-powered chatbot designed to provide personalized career advice. It analyzes resumes, suggests job opportunities based on user skills and experiences, and offers valuable interview tips. Leveraging SerpApi, CareerSage fetches and scrapes career-related articles, building a rich knowledge base for delivering expert-backed career guidance. With integration from reputable career advice books, CareerSage ensures comprehensive support for career navigation.

Use Case
CareerSage has the potential to transform career counseling and job search platforms by:

Job Seekers: Providing resume feedback, job recommendations aligned with skills, and expert interview preparation.
Career Counselors and HR Professionals: Offering insights to effectively guide clients or candidates with well-rounded, up-to-date career advice.
CareerSage delivers accessible, personalized, and data-driven career guidance, helping users navigate their professional journeys with ease.

Technologies
Gemini Model
LangChain
Streamlit
Workflow
Generate Vector Embeddings: Use Gemini embeddings in LangChain.
Store Vector Embeddings: Save them in the Faiss Vector Database.
Fetch Relevant Chunk: Retrieve the most relevant chunk of the source based on user input using vector embeddings.
Create Prompt Template: Design a template for prompts.
Generate Response: Send user input and relevant chunk to Gemini.
Display Chat: Show the conversation on the Streamlit dashboard.
Applications
Educational Institutions: Integrate the chatbot into school and university portals to guide students towards their career paths.
Corporate Onboarding: Assist new employees in understanding their career growth and development opportunities within the company.
Remote Areas: Provide career advice in geographically isolated locations where traditional counseling is scarce.
Continuous Learning Platforms: Offer career advice based on skills acquired through online courses or MOOCs.
Getting Started

Install dependencies: pip install -r requirements.txt
Set the API keys for SerpApi and Gemini in your environment variables. Add the following lines to your .env file:
SERPAPI_API_KEY=your_serpapi_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
Run the application: streamlit run app.py

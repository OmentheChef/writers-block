import streamlit as st
import requests
import tempfile
import os
import base64
import io
import json
import re
from datetime import datetime
import urllib.parse

# Try to import document processing libraries
try:
    import pypdf
    from pypdf import PdfReader
    pdf_available = True
except ImportError:
    pdf_available = False

try:
    import docx2txt
    docx_available = True
except ImportError:
    docx_available = False

# Document processing functions
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file."""
    if not pdf_available:
        return "PDF processing is not available. Install pypdf with: pip install pypdf"
    
    try:
        pdf = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file."""
    if not docx_available:
        return "DOCX processing is not available. Install docx2txt with: pip install docx2txt"
    
    try:
        text = docx2txt.process(io.BytesIO(file_bytes))
        return text
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def extract_text_from_txt(file_bytes):
    """Extract text from TXT file."""
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        return f"Error extracting text from TXT: {str(e)}"

def process_document(uploaded_file):
    """Process uploaded document and extract text."""
    if uploaded_file is None:
        return ""
    
    # Get file bytes
    file_bytes = uploaded_file.getvalue()
    
    # Extract text based on file type
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif uploaded_file.name.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        return f"Unsupported file type: {uploaded_file.name}"

def simple_chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into chunks with overlap."""
    chunks = []
    if not text:
        return chunks
        
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Find the last period or newline to make cleaner breaks
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:  # Only use if it's not too far back
                end = break_point + 1
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position, considering overlap
        if end == text_length:
            break
        start = end - overlap
    
    return chunks

def search_web(query, num_results=3):
    """Search the web using multiple free methods."""
    try:
        # URL encode the query
        encoded_query = urllib.parse.quote(query)
        results = []
        
        # METHOD 1: DuckDuckGo API (free, no key required)
        try:
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the abstract
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", ""),
                        "snippet": data.get("Abstract", ""),
                        "url": data.get("AbstractURL", "")
                    })
                
                # Extract related topics
                for topic in data.get("RelatedTopics", [])[:num_results-len(results)]:
                    if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                        results.append({
                            "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", "")
                        })
        except Exception as e:
            st.error(f"DuckDuckGo search error: {str(e)}")
        
        # METHOD 2: Wikipedia API (free, no key required)
        if len(results) < num_results:
            try:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&srlimit={num_results}"
                wiki_response = requests.get(wiki_url)
                
                if wiki_response.status_code == 200:
                    wiki_data = wiki_response.json()
                    search_results = wiki_data.get("query", {}).get("search", [])
                    
                    for result in search_results:
                        # Remove HTML tags from snippet
                        snippet = re.sub(r'<[^>]+>', '', result.get("snippet", ""))
                        title = result.get("title", "")
                        
                        # Only add if we don't already have this result
                        if not any(r["title"] == title for r in results):
                            results.append({
                                "title": title,
                                "snippet": snippet,
                                "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                            })
                            
                            if len(results) >= num_results:
                                break
            except Exception as e:
                pass  # Silently continue to next method
        
        # METHOD 3: Try to use a free search API if available
        if len(results) < num_results:
            try:
                # If SerpAPI key is provided, use it
                if st.session_state.get('serpapi_key'):
                    serpapi_url = f"https://serpapi.com/search.json?q={encoded_query}&api_key={st.session_state.get('serpapi_key')}"
                    serpapi_response = requests.get(serpapi_url)
                    
                    if serpapi_response.status_code == 200:
                        serpapi_data = serpapi_response.json()
                        for result in serpapi_data.get("organic_results", [])[:num_results-len(results)]:
                            results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "url": result.get("link", "")
                            })
            except Exception:
                pass  # Silent fail, try next method
        
        # If still no results, create a generic search result
        if not results:
            results = [{
                "title": "Search Results",
                "snippet": f"Try searching for '{query}' directly on a search engine for more detailed information.",
                "url": f"https://www.google.com/search?q={encoded_query}"
            }]
        
        return results
    except Exception as e:
        return [{"title": "Search Error", "snippet": f"Error: {str(e)}", "url": ""}]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "combined_text" not in st.session_state:
    st.session_state.combined_text = ""
if "model" not in st.session_state:
    st.session_state.model = "anthropic/claude-3-5-sonnet"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "serpapi_key" not in st.session_state:
    st.session_state.serpapi_key = ""

# Streamlit app
st.title("🌟 Jakalas Chatbot")
st.subheader("Chat with your documents")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", 
                                    type=["pdf", "docx", "txt"], 
                                    accept_multiple_files=True)

    # OpenRouter configuration
    # Updated model list with correct OpenRouter IDs, including the speculative Claude 3.7
    model = st.selectbox(
        "Select Model",
        [
            "anthropic/claude-3-7-sonnet",   # Claude 3.7 (speculative ID)
            "anthropic/claude-3-5-sonnet",   # Claude 3.5 Sonnet
            "openai/gpt-4o",                 # OpenAI GPT-4o
            "anthropic/claude-3-opus",       # Claude 3 Opus
            "openai/gpt-4-turbo",            # OpenAI GPT-4 Turbo
            "anthropic/claude-3-haiku",      # Claude 3 Haiku
            "meta/llama-3-70b-instruct",     # Llama 3 70B
            "mistral/mistral-large-latest",  # Mistral Large
        ],
        index=0,  # Default to Claude 3.7 
        key="model_selector"
    )
    st.session_state.model = model

    # API key input
    api_key = st.text_input("Enter OpenRouter API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
    
    # Optional SerpAPI key for better web search
    serpapi_key = st.text_input("SerpAPI Key (Optional)", type="password")
    if serpapi_key:
        st.session_state.serpapi_key = serpapi_key

    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.number_input("Chunk Size", min_value=500, max_value=8000, value=2000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        
        # Source strictness setting
        st.markdown("### Source Fidelity")
        source_strictness = st.slider(
            "Adherence to Source Material",
            min_value=0.0, max_value=1.0, value=0.9, step=0.1,
            help="Higher values make the AI stick more strictly to source documents"
        )
    
    # Additional capabilities info
    st.markdown("### Capabilities")
    st.markdown("• Document analysis with high source fidelity")
    st.markdown("• Web search capability")
    st.markdown("• Screenplay writing")
    st.markdown("• Character development")

    # Process documents button
    if uploaded_files and st.button("Process Documents"):
        all_text = ""
        file_names = []
        
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                file_names.append(uploaded_file.name)
                text = process_document(uploaded_file)
                all_text += f"\n\n--- Document: {uploaded_file.name} ---\n\n{text}"
            
            st.session_state.combined_text = all_text
            st.session_state.document_chunks = simple_chunk_text(all_text, chunk_size, chunk_overlap)
            st.session_state.document_processed = True
            
            st.success(f"Processed {len(uploaded_files)} documents into {len(st.session_state.document_chunks)} chunks")
            
            # Add system message about the documents
            if "messages" not in st.session_state or len(st.session_state.messages) == 0:
                st.session_state.messages = [
                    {"role": "system", "content": f"The user has uploaded the following documents: {', '.join(file_names)}. Stick strictly to the facts in these documents."},
                    {"role": "assistant", "content": f"I've processed your documents: {', '.join(file_names)}. I'll help you with information from these files, sticking strictly to their content."}
                ]

# Main chat interface
st.header("Chat with your documents")

# Search toggle button - position this near the chat input
web_search_col1, web_search_col2 = st.columns([1, 10])
with web_search_col1:
    web_search_enabled = st.checkbox("🌐", value=st.session_state.web_search_enabled, 
                                     help="Enable web search for this query")
    st.session_state.web_search_enabled = web_search_enabled

# Display message history
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about your documents or request screenplay writing"):
    # Store current query for potential web search
    st.session_state.current_query = prompt
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if document has been processed or screenplay request
    is_screenplay_request = "screenplay" in prompt.lower() or "script" in prompt.lower() or "scene" in prompt.lower()
    
    # Determine if we need a document (only if it's not a screenplay request AND no documents)
    needs_document = not st.session_state.document_processed and not is_screenplay_request
    
    if needs_document and not st.session_state.web_search_enabled:
        with st.chat_message("assistant"):
            st.markdown("Please upload and process documents first for document-based questions, or ask me to write a screenplay or creative content.")
    elif not st.session_state.api_key:
        with st.chat_message("assistant"):
            st.markdown("Please provide an OpenRouter API key.")
    else:
        # Perform web search if enabled
        web_search_context = ""
        if st.session_state.web_search_enabled:
            with st.spinner("Searching the web..."):
                search_results = search_web(prompt)
                st.session_state.search_results = search_results
                
                # Build web search context
                if search_results:
                    web_search_context = "Web search results:\n\n"
                    for i, result in enumerate(search_results, 1):
                        web_search_context += f"{i}. {result['title']}\n"
                        web_search_context += f"   {result['snippet']}\n"
                        web_search_context += f"   URL: {result['url']}\n\n"
                else:
                    web_search_context = "Web search was performed but returned no results."
        
        # Prepare context based on if it's a document question or screenplay request
        if is_screenplay_request:
            # For screenplay requests, search the document for relevant context
            screenplay_chunks = []
            if st.session_state.document_processed:
                query_terms = prompt.lower().split()
                for chunk in st.session_state.document_chunks:
                    chunk_lower = chunk.lower()
                    # Simple relevance scoring for script requests too
                    relevance = sum(1 for term in query_terms if term in chunk_lower)
                    if relevance > 0:
                        screenplay_chunks.append((chunk, relevance))
                
                screenplay_chunks.sort(key=lambda x: x[1], reverse=True)
                top_screenplay_chunks = [chunk for chunk, _ in screenplay_chunks[:2]]
                
                if top_screenplay_chunks:
                    context = "The user is requesting screenplay writing assistance. Use these document excerpts as source material:\n\n"
                    context += "\n\n".join(top_screenplay_chunks)
                else:
                    context = "The user is requesting screenplay writing assistance. Respond with properly formatted screenplay content."
            else:
                context = "The user is requesting screenplay writing assistance. Respond with properly formatted screenplay content."
        else:
            # Find relevant chunks for the query from documents
            relevant_chunks = []
            query_terms = prompt.lower().split()
            
            for chunk in st.session_state.document_chunks:
                chunk_lower = chunk.lower()
                # Simple relevance scoring - count how many query terms appear in the chunk
                relevance = sum(1 for term in query_terms if term in chunk_lower)
                if relevance > 0:
                    relevant_chunks.append((chunk, relevance))
            
            # Sort by relevance score and take top chunks
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk for chunk, _ in relevant_chunks[:3]]
            
            # If no relevant chunks found, use some document content as context
            if not top_chunks and st.session_state.document_chunks:
                top_chunks = st.session_state.document_chunks[:2]
            
            # Build context from relevant chunks
            context = "\n\n".join(top_chunks) if top_chunks else "No relevant content found in the documents."
        
        # Add web search results to context if available
        if web_search_context:
            context = context + "\n\n" + web_search_context
        
        # Create system message with high source fidelity instructions
        system_content = f"""You are Jakalas Chatbot, a multi-functional assistant with EXTREMELY HIGH SOURCE FIDELITY. Your primary directives are:

1. STICK STRICTLY TO SOURCE DOCUMENTS - Do not hallucinate details or add information not present in the provided documents
2. ADMIT WHEN YOU DON'T KNOW - If information isn't in the sources, say "The documents don't mention this"
3. CITE YOUR SOURCES - Indicate which document or search result contains the information
4. ACKNOWLEDGE LIMITATIONS - Be transparent about gaps in document coverage

Your capabilities include:
- Document analysis with strict adherence to source material
- Web search integration when enabled by the user
- Screenplay writing based on document content
- Character development based on document details

When writing screenplays, use proper screenplay formatting with scene headings (INT/EXT), character names in ALL CAPS before dialogue, and action descriptions.

IMPORTANT: Always verify information against source documents before including it in your response.

Current date: {datetime.now().strftime('%Y-%m-%d')}

Document and search context:
{context}
"""
        
        # Prepare messages for OpenRouter API
        api_messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Add conversation history (excluding system messages)
        for message in st.session_state.messages:
            if message["role"] != "system":
                api_messages.append(message)
        
        # Get response from OpenRouter
        with st.chat_message("assistant"):
            with st.spinner("Generating response..." if is_screenplay_request else "Analyzing documents..."):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {st.session_state.api_key}",
                            "HTTP-Referer": "https://localhost:8501",  # Required by OpenRouter
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": st.session_state.model,
                            "messages": api_messages,
                            "temperature": temperature
                        }
                    )
                    
                    if response.status_code == 200:
                        assistant_response = response.json()["choices"][0]["message"]["content"]
                        
                        # Display any web search results used above the response
                        if st.session_state.web_search_enabled and st.session_state.search_results:
                            with st.expander("Web Search Results Used", expanded=False):
                                for i, result in enumerate(st.session_state.search_results, 1):
                                    st.markdown(f"**{i}. [{result['title']}]({result['url']})**")
                                    st.markdown(f"{result['snippet']}")
                                    st.markdown("---")
                        
                        # Display the AI response
                        st.markdown(assistant_response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        st.error(f"Error from OpenRouter API: {response.status_code}, {response.text}")
                        
                        # Helpful error message for invalid model IDs
                        if response.status_code == 400 and "not a valid model ID" in str(response.text):
                            st.info("It seems the selected model ID is invalid. Please select a different model from the dropdown menu.")
                            
                            # If Claude 3.7 was attempted but doesn't exist
                            if "claude-3-7" in st.session_state.model:
                                st.info("Note: Claude 3.7 might not be available yet on OpenRouter. Try Claude 3.5 Sonnet instead.")
                except Exception as e:
                    st.error(f"Error calling OpenRouter API: {str(e)}")

# Display document information if processed
if st.session_state.document_processed:
    with st.expander("Document Information", expanded=False):
        st.write(f"Total text length: {len(st.session_state.combined_text)} characters")
        st.write(f"Number of chunks: {len(st.session_state.document_chunks)}")
        if st.checkbox("Show first chunk sample"):
            if st.session_state.document_chunks:
                st.text_area("First chunk content:", st.session_state.document_chunks[0], height=200)

# Clear chat history button
if st.button("Clear Chat History"):
    # Preserve document processing but clear messages
    system_msg = next((m for m in st.session_state.messages if m["role"] == "system"), None)
    st.session_state.messages = [system_msg] if system_msg else []
    st.rerun()

# Footer
st.divider()
st.caption("Writers Chatbot - Chat with your documents with high source fidelity, search the web for research, and create screenplay content.")


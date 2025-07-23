import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os
import numpy as np
from openai import OpenAI
import json
import pandas as pd
import matplotlib.pyplot as plt
import ast
import zipfile
import os

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def load_embeddings():
    return np.load("word_embeddings.npy")

@st.cache_data
def load_corpus_df():
    with open("my_corpus.json", "r", encoding="utf-8") as f:
        structured_corpus = json.load(f)
    return pd.DataFrame(structured_corpus)

if not os.path.exists("word_embeddings.npy") and os.path.exists("word_embeddings.zip"):
    with zipfile.ZipFile("word_embeddings.zip", 'r') as zip_ref:
        zip_ref.extractall()

if not os.path.exists("my_corpus.json") and os.path.exists("my_corpus.zip"):
    with zipfile.ZipFile("my_corpus.zip", 'r') as zip_ref:
        zip_ref.extractall()

st.set_page_config(layout="wide")

Metadata_agent_instructions = (
    "You are a data assistant for a job postings dataset. "
    "Your ONLY job is to return a single valid JSON dictionary using double quotes and correct syntax (including commas between keys). "
    "The JSON describes the user's intent. You must infer the correct task, column, and any filters from the user's message. "
    "Supported tasks: 'count', 'plot', 'most_common'.\n\n"

    "You MUST return exactly one valid JSON object with keys: 'task', 'column', and optionally 'filter'. "
    "No explanations, no markdown, no comments — only the raw JSON object.\n\n"

    "Infer the task using these strict rules:\n"
    "- If the prompt includes: 'plot', 'bar chart', 'visualize', 'show distribution', return: \"task\": \"plot\"\n"
    "- If the prompt includes: 'how many', 'number of', 'count', return: \"task\": \"count\"\n"
    "- If the prompt includes: 'most common', 'frequent', 'mode', return: \"task\": \"most_common\"\n"
    "- Never guess the task. If uncertain, return: {\"task\": \"none\"}\n"
    "- If the user says 'plot authenticity', return: {\"task\": \"plot\", \"column\": \"Job Authenticity\"}\n\n"

    "Map user-friendly phrases to actual column names:\n"
    "- 'job title' or 'position' → 'Title'\n"
    "- 'job type' or 'type' → 'employment_type'\n"
    "- 'education level' → 'required_education'\n"
    "- 'experience' → 'required_experience'\n"
    "- 'location' → 'location'\n"
    "- 'authenticity', 'validity', 'real or fake', 'legitimacy', 'fraud status' → 'Job Authenticity'\n"
    "- 'industry' and 'function' remain unchanged\n\n"

    "Apply filters when 'real' or 'fake' modifies any of the following: job, title, function, industry, education, etc.:\n"
    "- If prompt contains 'real [thing]', add filter: {\"Job Authenticity\": \"real\"}\n"
    "- If prompt contains 'fake [thing]', add filter: {\"Job Authenticity\": \"fake\"}\n\n"

    "VALID COLUMNS:\n"
    "- Title\n- Job Authenticity\n- industry\n- function\n"
    "- required_education\n- required_experience\n- location\n- employment_type\n\n"

    "VALID TASKS:\n"
    "- count: how many jobs (optionally filtered)\n"
    "- plot: bar chart of top values in a column (optionally filtered)\n"
    "- most_common: return most frequent value in a column (optionally filtered)\n\n"

    "EXAMPLES:\n"
    "- 'plot authenticity' → {\"task\": \"plot\", \"column\": \"Job Authenticity\"}\n"
    "- 'how many fake jobs?' → {\"task\": \"count\", \"filter\": {\"Job Authenticity\": \"fake\"}}\n"
    "- 'most common real title' → {\"task\": \"most_common\", \"column\": \"Title\", \"filter\": {\"Job Authenticity\": \"real\"}}\n"
    "- 'most common industry for fake jobs' → {\"task\": \"most_common\", \"column\": \"industry\", \"filter\": {\"Job Authenticity\": \"fake\"}}\n\n"

    "IMPORTANT:\n"
    "- Always return proper JSON syntax with commas and double quotes.\n"
    "- Never default to 'Title'. Only use it if the prompt refers to job title or position.\n"
    "- If the task or column is unclear, return: {\"task\": \"none\"}")

def extract_first_json_block(text):
    try:
        # Try to parse the full text directly first
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # Try regex fallback below

    # Fallback: extract largest valid JSON object using greedy match
    matches = re.findall(r'{[^{}]*?(?:(?R)[^{}]*?)*}', text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found.")


def parse_metadata_prompt_with_llm(prompt):
    client = st.session_state["client"]
    model = st.session_state["my_llm_model"]
    
    messages = [
        {"role": "system", "content": Metadata_agent_instructions},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    
    raw_text = response.choices[0].message.content.strip()


    try:
        parsed = extract_first_json_block(raw_text)
        return parsed
    except Exception as e:
        raise ValueError(f"Failed to parse metadata JSON. Raw response: {raw_text}\n\nError: {e}")

def handle_metadata_query(prompt, df):
    try:
        parsed = parse_metadata_prompt_with_llm(prompt)
        st.subheader("Final Parsed Metadata JSON (Parsed version used by the app)")
        st.json(parsed, expanded=False)
    except Exception:
        return (
            "I'm a metadata agent. Please refer to the available functions:\n\n"
            "- **Counts**: e.g., *How many jobs are there?*\n"
            "- **Most Common**: e.g., *Most common industry among jobs?*\n"
            "- **Plots**: e.g., *Plot job functions*"
        )

    prompt_lower = prompt.lower()
    task = parsed.get("task")
    if "plot" in prompt_lower:
        task = "plot"
    elif "most common" in prompt_lower:
        task = "most_common"
    elif "how many" in prompt_lower or "count" in prompt_lower:
        task = "count"
    column = parsed.get("column")
    filters = parsed.get("filter", {}) or {}

    # --- Apply extra filters if mentioned in prompt ---
    real_synonyms = ["real", "genuine", "legit"]
    fake_synonyms = ["fake", "fraudulent", "scam", "phony"]

    if "Job Authenticity" not in filters:
        if any(word in prompt_lower for word in real_synonyms):
            filters["Job Authenticity"] = "real"
        elif any(word in prompt_lower for word in fake_synonyms):
            filters["Job Authenticity"] = "fake"

    # --- Infer task if not confidently set ---
    if not task:
        if "plot" in prompt_lower:
            task = "plot"
        elif "most common" in prompt_lower:
            task = "most_common"
        elif "how many" in prompt_lower or "count" in prompt_lower:
            task = "count"
        else:
            return (
                "I'm a metadata agent. Please refer to the available functions:\n\n"
                "- **Counts**: e.g., *How many jobs are there?*\n"
                "- **Most Common**: e.g., *Most common industry among jobs?*\n"
                "- **Plots**: e.g., *Plot job functions*"
            )

    # --- Clean up column synonyms ---
    if column and column.lower() in ["job title", "job", "position"]:
        column = "Title"

    # --- Final safety check ---
    valid_tasks = {"count", "plot", "most_common"}
    if task not in valid_tasks:
        return (
            "I'm a metadata agent. Please refer to the available functions:\n\n"
            "- **Counts**: e.g., *How many jobs are there?*\n"
            "- **Most Common**: e.g., *Most common industry among jobs?*\n"
            "- **Plots**: e.g., *Plot job functions*"
        )

    # --- Apply filters ---
    df_filtered = df.copy()
    for col, val in filters.items():
        if col not in df_filtered.columns:
            return f"Invalid filter column: `{col}`. Available columns: {list(df_filtered.columns)}"
        df_filtered = df_filtered[df_filtered[col].notna()]
        df_filtered = df_filtered[df_filtered[col].str.lower().str.contains(str(val).lower())]

    # --- Handle each task ---
    if task == "count":
        return f"The dataset contains {len(df_filtered):,} job postings."

    elif task == "most_common":
        if column not in df_filtered.columns:
            return f"Invalid column: `{column}` for most_common task."
        if df_filtered[column].dropna().empty:
            return f"No data available for column: `{column}` after filtering."
        most_common_value = df_filtered[column].value_counts().idxmax()
        return f"The most common value in column `{column}` is: **{most_common_value}**."

    elif task == "plot":
        if column not in df_filtered.columns:
            return f"Invalid column: `{column}` for plotting."
        valid_series = df_filtered[column].dropna().astype(str).str.strip()
        valid_series = valid_series[valid_series != ""]
        if valid_series.empty:
            return f"No valid values to plot in column: `{column}` after filtering."
        top_counts = valid_series.value_counts()
        top_counts.index = top_counts.index.str.capitalize()
        with st.spinner(f"Generating bar chart for `{column}`..."):
            top_counts = valid_series.value_counts().nlargest(10)
            fig, ax = plt.subplots(figsize=(10, 4))
            top_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Top 10 categories in '{column}'")
            ax.set_ylabel("Count")
            ax.set_xticklabels(top_counts.index, rotation=45, ha="right")
            for i, v in enumerate(top_counts.values):
                ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=8)
            # st.write("Filters applied:", filters)
            # st.write("Filtered dataset shape:", df_filtered.shape)
            st.write("Filtered value counts:", df_filtered["Job Authenticity"].value_counts())
            st.pyplot(fig)
            return f"Displayed bar chart for column: `{column}`."

    return "Unrecognized task. Try asking for 'count', 'plot', or 'most_common'."
    
# Load corpus if not already in session state
if "my_rag_text" not in st.session_state:
    df = load_corpus_df()
    st.session_state["structured_corpus_df"] = df

    structured_corpus = df.to_dict(orient="records")
    my_corpus_chunks = [
        f"Job Authenticity: {entry.get('Job Authenticity', '')}\n\n"
        f"Title: {entry.get('Title', '')}\n\n"
        f"Location: {entry.get('location', '')}\n\n"
        f"Employment Type: {entry.get('employment_type', '')}\n\n"
        f"Experience Level: {entry.get('required_experience', '')}\n\n"
        f"Education: {entry.get('required_education', '')}\n\n"
        f"Industry: {entry.get('industry', '')}\n\n"
        f"Function: {entry.get('function', '')}\n\n"
        f"Company Profile:\n{entry.get('company_profile', '')}\n\n"
        f"Description:\n{entry.get('description', '')}\n\n"
        f"Requirements:\n{entry.get('requirements', '')}\n\n"
        f"Benefits:\n{entry.get('benefits', '')}"
        for entry in structured_corpus
    ]
    my_initial_rag_text = "\n\n===JOB_ENTRY===\n\n".join(my_corpus_chunks)
    st.session_state["my_rag_text"] = my_initial_rag_text


# Load embeddings if not already in session state
if "word_embeddings" not in st.session_state:
    st.session_state["word_embeddings"] = load_embeddings()

# Check if the LLM model is not already in the session state
if "my_llm_model" not in st.session_state:
    # Set the default LLM model to "mistralai/Mistral-7B-Instruct-v0.3"
    st.session_state['my_llm_model'] = "mistralai/Mistral-7B-Instruct-v0.3"

# Check if the SPACE_ID environment variable is not already in the session state
if "my_space" not in st.session_state:
    st.session_state['my_space'] = os.environ.get("SPACE_ID") 

# Function to update the LLM model client
def update_llm_model():
    if st.session_state['my_llm_model'].startswith("gemini-"):
        # Initialize the client for gemini models. We use the OpenAI API to interact with gemini models.
        st.session_state['client'] = OpenAI(api_key = os.getenv("GOOGLE_API_KEY"),
                                            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/")
    elif st.session_state['my_llm_model'].startswith("gpt-"):
        # Initialize the client for openai models
        st.session_state['client'] = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
                                            # ,base_url = "https://eu.api.openai.com/" # gives error
    else:
        if st.session_state['my_space']:
            # Initialize the client with the model if SPACE_ID is available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'])
        else:
            # Initialize the client with the model and token if SPACE_ID is not available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=os.getenv("HF_TOKEN"))
        
# Check if the client is not already in the session state
if "client" not in st.session_state:
    update_llm_model()

# Check if the embeddings model is not already in the session state
if "embeddings_model" not in st.session_state:
    st.session_state['embeddings_model'] = load_embedding_model()

if "my_system_instructions" not in st.session_state:
    st.session_state["my_system_instructions"] = ("You are a helpful assistant that can answer questions only about the job postings in the corpus. Be brief and concise. "
                                                   "Provide your answers in 100 words or less. Only answer using the information in the context you are given. Do not guess or generalize."
                                                   "Job postings may be marked as real or fake (field: Job Authenticity). Include this when summarizing a posting or listing its characteristics.")

first_message = "Hello, how can I help you today?"

def delete_chat_messages():
    for key in st.session_state.keys():
        if key != "my_rag_text" and key != "my_system_instructions":
            del st.session_state[key]
    update_llm_model()
            
def create_sentences_rag():
    with rag_status_placeholder:
        pattern = r'\n+={3,}JOB_ENTRY={3,}\n+'
        st.session_state['my_sentences'] = [s.strip() for s in re.split(pattern, st.session_state['my_rag_text']) if s.strip()]
        
        with st.spinner(f"Loading {len(st.session_state['my_sentences'])} pre-chunked job entries..."):
            st.session_state['my_sentences_rag'] = st.session_state['my_sentences']
            st.session_state['my_sentences_rag_ids'] = [[i] for i in range(len(st.session_state['my_sentences']))]

            if len(st.session_state["word_embeddings"]) != len(st.session_state['my_sentences_rag']):
                st.error(f"Mismatch: {len(st.session_state['my_sentences_rag'])} chunks vs {len(st.session_state['word_embeddings'])} embeddings. Please fix this.")
                st.stop()
            st.session_state['my_embeddings'] = st.session_state["word_embeddings"]
        st.success(f"{len(st.session_state['my_sentences_rag'])} chunks loaded successfully.")
# Create two columns with a 1:2 ratio
column_1, column_2 = st.columns([1, 2])

# In the first column
if "show_disclaimer" not in st.session_state:

    st.session_state["show_disclaimer"] = False

with column_1:
    if st.button("View Disclaimer"):
        st.session_state["show_disclaimer"] = not st.session_state["show_disclaimer"]

    if st.session_state["show_disclaimer"]:
        st.markdown("""
        **Disclaimer**

        This application and code (hereafter referred to as the “Software”) is a proof of concept and is intended solely for research, experimentation, and educational purposes. It is not designed or approved for use in production environments or for making hiring decisions, job classifications, or any form of automated screening.

        The dataset used within the Software is derived from publicly available data on Kaggle with CC0: Public Domain copyright license (https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data), which contains both real and synthetic (fake) job postings. As such, the content and structure of the data may include inaccuracies, inconsistencies, or fabricated information that does not reflect real-world job listings. 
        
        The Software does not verify or validate the authenticity of the data.

        The Software is provided “as is,” without any warranties of any kind, expressed or implied. The user assumes full responsibility for how the Software and underlying data are used, interpreted, or implemented.

        The developers of the Software shall not be liable for any damages, losses, claims, or liabilities arising from its use. This includes, but is not limited to, errors in data interpretation, misuse of AI-generated outputs, third-party tool failures, data privacy breaches, intellectual property issues, legal or regulatory violations, or any indirect or consequential harm.

        Large Language Models (LLMs) integrated into the Software may generate incorrect, misleading, or biased responses. The outputs should not be considered factual or final without independent verification.

        By using the Software, you agree to:

        - Acknowledge the experimental and synthetic nature of the dataset.  
        - Accept the risks associated with using AI-generated content.  
        - Indemnify and hold harmless the developers from any legal or ethical claims resulting from your use of the Software.

        By continuing to use this Software, you accept the terms outlined in this disclaimer.
        """)
    st.markdown("### Choose Assistant Mode")

    agent_choice = st.radio(
        "Select which assistant you want to use:",
        ["Metadata Agent", "RAG Chatbot"],
        index=1,
        help="Select 'Metadata Agent' for general stats and plots. Choose 'RAG Chatbot' to search within job posting content using retrieval."
    )
    st.session_state["active_agent"] = agent_choice
    if st.session_state["active_agent"] == "Metadata Agent":
        st.markdown("### Available Metadata Questions")

        # Get column names
        df_columns = st.session_state["structured_corpus_df"].columns.tolist()
        plot_columns = [col for col in df_columns if col.lower() not in ["description", "company_profile", "benefits"]]
        formatted_columns = ", ".join(f"`{col}`" for col in plot_columns)

        st.markdown(f"""
        You can ask about:
        - **Counts**:
        - How many jobs are there?
        - How many fake or real jobs?
        - **Most Common**:
        - What is the most common industry?
        - What is the most common real title?
        - **Plots**:
        - Plot top categories in a column (bar plot)
        
        **Available columns for plotting:**  
        {formatted_columns}
        """)
          
        st.markdown("---")   
    else:
         st.markdown("### Example RAG Chatbot Questions")
         st.markdown(f"""
         You can ask questions like:
         - What is this dataset about?
         - What are the most common red flags in fake job descriptions?
         - Describe the benefits offered in marketing jobs.
         - Are there any postings for teachers?
         """)

    # Add a selectbox for model selection
    model_list_all = [  'mistralai/Mistral-7B-Instruct-v0.3',
                        'Qwen/Qwen2.5-72B-Instruct'] 
                        #'HuggingFaceH4/zephyr-7b-beta']
    if os.getenv("GOOGLE_API_KEY"):
        model_list_all.append('gemini-2.5-flash-preview-05-20')
    if os.getenv("OPENAI_API_KEY"):
        model_list_all.append('gpt-4.1-nano-2025-04-14')
    st.selectbox("Select the model to use:",
                model_list_all, 
                key="my_llm_model", 
                on_change=update_llm_model)
    
    # Add a text are for the system instructions
    # st.text_area(label="Please enter your system instructions here:", value=my_system_instructions, height=80, key="my_system_instructions", on_change=delete_chat_messages)
    
    # Placeholder right after text_area
    rag_status_placeholder = st.empty()
    # Add a text area for RAG text input
    # st.text_area(label="Please enter your RAG text here:", value=my_initial_rag_text, height=200, key="my_rag_text", on_change=delete_chat_messages)

# --- Window Size Settings ---
with column_1.expander("Window Size Settings"):
    st.slider(
        "Minimum window size in original sentences", 
        min_value=1, max_value=20, value=5, step=1, 
        key="min_window_size", 
        on_change=create_sentences_rag
    )
    st.caption("Minimum number of consecutive sentences to form a context chunk.")

    st.slider(
        "Maximum window size in original sentences", 
        min_value=1, max_value=20, value=10, step=1, 
        key="max_window_size", 
        on_change=create_sentences_rag
    )
    st.caption("Maximum number of sentences allowed in a chunk for retrieval.")

# --- Similarity & Filtering Settings ---
with column_1.expander("Similarity and Filtering Settings"):
    st.slider(
        "Similarity Threshold", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.01, 
        key="my_similarity_threshold"
    )
    st.caption("Controls how closely a chunk must match the user query. Raise this to include only highly relevant content (e.g., 0.3–0.5).")

    st.slider(
        "Number of original chunks to keep", 
        min_value=1, max_value=50, value=20, step=1, 
        key="nof_keep_sentences"
    )
    st.caption("Maximum number of relevant context chunks to retrieve for each query.")

# --- Prompt Splitting Settings ---
with column_1.expander("Sub-Prompt Splitting"):
    st.slider(
        "Minimum number of words in sub prompt split", 
        min_value=1, max_value=10, value=1, step=1, 
        key="nof_min_sub_prompts"
    )
    st.caption("Minimum number of words in a sub-prompt generated from the user query.")

    st.slider(
        "Maximum number of words in sub prompt split", 
        min_value=1, max_value=10, value=5, step=1, 
        key="nof_max_sub_prompts"
    )
    st.caption("Maximum number of words allowed in a sub-prompt from the user query.")

# Check if the chat messages are not already in the session state
if "my_chat_messages" not in st.session_state:
    # Initialize the chat messages list in the session state
    st.session_state['my_chat_messages'] = []
    # Add the system instructions to the chat messages
    st.session_state['my_chat_messages'].append({"role": "system", "content": st.session_state['my_system_instructions']})

# print(100*"-")
# Check if the sentences are not already in the session state
if "my_sentences_rag" not in st.session_state or "my_embeddings" not in st.session_state:
    create_sentences_rag()

with column_2:
    # Create a container for the messages with a specified height
    messages_container = st.container(height=500) 
    
    # Display the first message from the assistant
    messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)
    
    # Iterate through the chat messages stored in the session state
    for message in st.session_state['my_chat_messages']:
        if message['role'] == "user":
            # Display user messages with a specific avatar - https://fonts.google.com/icons
            messages_container.chat_message(message['role'], avatar=":material/psychology_alt:").markdown(message['content'])
        elif message['role'] == "assistant":
            # Display assistant messages with a specific avatar
            messages_container.chat_message(message['role'], avatar=":material/robot_2:").markdown(message['content'])

    # Check if there is a new prompt from the user
    if prompt := st.chat_input("you may ask here your questions"):
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)
        if st.session_state["active_agent"] == "Metadata Agent":
            response = handle_metadata_query(prompt, st.session_state["structured_corpus_df"])
            messages_container.chat_message("assistant", avatar=":material/robot_2:").markdown(response)
            st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
            st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})
        else:
            # Split the prompt into words
            split_prompt = prompt.split(" ")
            all_sub_prompts = []
        # Generate sub-prompts based on the specified range
            for jj in range(st.session_state['nof_min_sub_prompts'], st.session_state['nof_max_sub_prompts']+1):
                for ii in range(len(split_prompt)):
                    # Create sub-prompt by joining words
                    i_split = " ".join(split_prompt[ii:ii+jj]).strip()
                    if i_split:
                        all_sub_prompts.append(i_split)
        
            similarities_to_question = np.zeros(len(st.session_state['my_embeddings']))
            for sub_prompt in all_sub_prompts:
                # Encode the user's prompt to get its embedding
                my_question_embedding = st.session_state.embeddings_model.encode([sub_prompt])
            
                # Calculate the cosine similarity between the prompt embedding and stored embeddings
                similarities_to_question += cosine_similarity(my_question_embedding, st.session_state['my_embeddings']).flatten()
            similarities_to_question /= len(all_sub_prompts)
        
            # Get the indices of the top similar sentences
            bottom_col1, bottom_col2 = st.columns([1, 1])
            sorted_indices_rag = similarities_to_question.argsort()[::-1]
            sorted_indices_sentences = []
            max_similarity = 0
            # for irag in range(st.session_state['nof_keep_sentences']):
            irag = 0
            while len(set(sorted_indices_sentences))<st.session_state['nof_keep_sentences'] and irag<len(sorted_indices_rag):
                sorted_indices_sentences.extend(st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]])
                max_similarity = max(max_similarity, similarities_to_question[sorted_indices_rag[irag]])
                with bottom_col1:
                    str_conf = f"Confidence: {similarities_to_question[sorted_indices_rag[irag]]:.5f}, Sentences IDs: {st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]]}"
                    with st.expander(f"Chunk: {str(irag+1)} {str_conf}"):
                        for idx in st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]]:
                            st.write(f"{st.session_state['my_sentences'][idx]}")
                irag += 1

            sorted_indices_sentences = sorted(list(set(sorted_indices_sentences)))
            
            # Create an empty container for the streaming response from the assistant
            with messages_container.chat_message("ai", avatar=":material/robot_2:"):
                response_placeholder = st.empty()
                if max_similarity > st.session_state['my_similarity_threshold']:
                    # Construct the augmented prompt with the similar sentences
                    augmented_prompt = "This is my context:" + "\n\n" + 20*"-" + "\n\n" 
                    augmented_prompt += "\n".join([st.session_state['my_sentences'][idx] for idx in sorted_indices_sentences])
                    augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "If the above context is not relevant to the prompt, ignore the context and reply based only on the prompt."
                    augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "If the above context is relevant to the prompt, reply based on the context and the prompt."
                    augmented_prompt += "\n\n" + 20*"-" + "\n\n" + "The prompt is:"
                    augmented_prompt += "\n\n" + f"\n\n{prompt}"
                    # Append the augmented prompt to the chat messages in the session state
                    st.session_state['my_chat_messages'].append({"role": "user", "content": augmented_prompt})
                    # Stream the response from the assistant and update the placeholder
                    response = ""
                    for chunk in st.session_state['client'].chat.completions.create(messages = st.session_state['my_chat_messages'], 
                                                                                    model = st.session_state['my_llm_model'],
                                                                                    stream = True, 
                                                                                    max_tokens = 1024):
                        if chunk.choices[0].delta.content:
                            response += chunk.choices[0].delta.content
                            # Use markdown to update the response placeholder with the streamed content
                            response_placeholder.markdown(response)
                    # Remove the last message from the chat messages in the session state
                    st.session_state['my_chat_messages'].pop()
                else:
                    augmented_prompt = ""
                    response = f"This question is not about the job postings in the dataset. The maximum similarity found in the context is: {100*max_similarity:.2f}%."
                    response_placeholder.markdown(response)

            # Append the user's original prompt to the chat messages in the session state
            st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
            # Append the assistant's response to the chat messages in the session state
            st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})


            if len(st.session_state['my_chat_messages'])>10:
                # Keep the first message which is the system instructions, remove the 2nd and 3rd messages which are the first user and assistant messages
                st.session_state['my_chat_messages'] = st.session_state['my_chat_messages'][:1] + st.session_state['my_chat_messages'][3:]
        
                

            with bottom_col2:
                # Display the augmented prompt used for generating the response
                st.write("Augmented prompt:")
                st.json({"max_similarity": max_similarity, "augmented_prompt": augmented_prompt}, expanded=False)

                # Display the chat messages history
                st.write("Messages History All:")
                st.json(st.session_state['my_chat_messages'], expanded=False)
